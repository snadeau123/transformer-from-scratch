"""
Training Utilities from Scratch

This module implements:
- CrossEntropyLoss: Loss function for next-token prediction
- SGD: Stochastic Gradient Descent optimizer with momentum
- Gradient clipping: Prevents exploding gradients
- Training loop: Puts it all together

These are the fundamental components needed to train any neural network.
"""

import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy Loss with Softmax.

    For classification/next-token prediction, we use cross-entropy loss:
        L = -log(p_correct)

    Where p_correct is the predicted probability of the correct class.

    We combine softmax and cross-entropy for numerical stability:
    1. Computing softmax then log can cause numerical issues
    2. Combined form is more stable: log(softmax(x)) = x - log(sum(exp(x)))

    For language modeling:
    - Logits: (batch, seq_len, vocab_size) - raw model outputs
    - Targets: (batch, seq_len) - correct token indices
    - Loss: scalar - average negative log-probability of correct tokens

    Backward:
        The gradient of softmax + cross-entropy has a beautiful simple form:
        dL/d(logits) = predicted_probs - one_hot(targets)

        This is the difference between what we predicted and what was correct.
    """

    def __init__(self, eps=1e-8):
        """
        Initialize loss function.

        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps

        # Cache for backward pass
        self.probs = None
        self.targets = None
        self.logits_shape = None

    def forward(self, logits, targets):
        """
        Compute cross-entropy loss.

        Args:
            logits: Model outputs, shape (batch, seq_len, vocab_size)
                    These are raw scores (unnormalized log-probabilities)
            targets: Target token indices, shape (batch, seq_len)
                     Values in [0, vocab_size)

        Returns:
            Scalar loss value (average negative log-probability)
        """
        self.logits_shape = logits.shape
        self.targets = targets

        batch_size, seq_len, vocab_size = logits.shape

        # =====================================================================
        # Step 1: Compute softmax (numerically stable)
        # =====================================================================
        # Subtract max for numerical stability before exp
        logits_max = np.max(logits, axis=-1, keepdims=True)
        logits_shifted = logits - logits_max

        # Compute exp and normalize
        exp_logits = np.exp(logits_shifted)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        self.probs = exp_logits / sum_exp
        # Shape: (batch, seq_len, vocab_size)

        # =====================================================================
        # Step 2: Get probability of correct tokens
        # =====================================================================
        # We need probs[b, s, targets[b, s]] for each position
        # Using advanced indexing:

        # Create batch and sequence indices
        batch_idx = np.arange(batch_size)[:, None]  # (batch, 1)
        seq_idx = np.arange(seq_len)[None, :]  # (1, seq)

        # Get probabilities of correct tokens
        correct_probs = self.probs[batch_idx, seq_idx, targets]
        # Shape: (batch, seq_len)

        # =====================================================================
        # Step 3: Compute cross-entropy loss
        # =====================================================================
        # Cross-entropy = -log(p_correct)
        # Average over all tokens
        loss = -np.mean(np.log(correct_probs + self.eps))

        return loss

    def backward(self):
        """
        Compute gradient of loss w.r.t. logits.

        The gradient of softmax + cross-entropy is remarkably simple:
            dL/d(logits_i) = probs_i - indicator(i == target)

        In other words:
        - For the correct class: gradient = prob - 1 (we want higher prob)
        - For other classes: gradient = prob (we want lower prob)

        This pushes the model to increase probability of correct class
        and decrease probability of incorrect classes.

        Returns:
            Gradient w.r.t. logits, shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len, vocab_size = self.logits_shape

        # Start with the predicted probabilities
        grad = self.probs.copy()

        # Subtract 1 from the correct class positions
        batch_idx = np.arange(batch_size)[:, None]
        seq_idx = np.arange(seq_len)[None, :]

        grad[batch_idx, seq_idx, self.targets] -= 1

        # Average over all tokens (to match the mean in forward)
        grad /= (batch_size * seq_len)

        return grad


class SGD:
    """
    Stochastic Gradient Descent with Momentum.

    The basic update rule is:
        param = param - learning_rate * gradient

    With momentum, we maintain a velocity for each parameter:
        velocity = momentum * velocity - learning_rate * gradient
        param = param + velocity

    Momentum helps in two ways:
    1. Accelerates convergence by building up speed in consistent directions
    2. Smooths out noisy gradients, reducing oscillations

    Think of it like a ball rolling down a hill:
    - Without momentum: ball stops immediately when gradient is zero
    - With momentum: ball has inertia and can roll through flat spots

    Typical momentum values: 0.9 or 0.99
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize optimizer.

        Args:
            learning_rate: Step size for parameter updates
            momentum: Momentum coefficient (0 = no momentum)
        """
        self.lr = learning_rate
        self.momentum = momentum

        # Velocity for each parameter (lazy initialization)
        self.velocities = {}

    def step(self, params_and_grads):
        """
        Update parameters using gradients.

        Args:
            params_and_grads: List of (parameter, gradient) tuples
                              Parameters are numpy arrays (modified in-place)
                              Gradients are numpy arrays (same shape as params)
        """
        for i, (param, grad) in enumerate(params_and_grads):
            # Skip if gradient is None (e.g., for embedding input)
            if grad is None:
                continue

            if self.momentum > 0:
                # ============================================================
                # SGD with Momentum
                # ============================================================

                # Initialize velocity on first use
                if i not in self.velocities:
                    self.velocities[i] = np.zeros_like(param)

                # Update velocity: v = momentum * v - lr * grad
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad

                # Update parameter: param += v
                param += self.velocities[i]

            else:
                # ============================================================
                # Vanilla SGD (no momentum)
                # ============================================================
                param -= self.lr * grad


def clip_gradients(params_and_grads, max_norm=1.0):
    """
    Clip gradients to prevent exploding gradients.

    Gradient clipping is essential for training transformers.
    Without it, gradients can become extremely large, causing
    unstable training or NaN values.

    We use global norm clipping:
    1. Compute the total norm of all gradients
    2. If total norm > max_norm, scale all gradients down proportionally

    This preserves the direction of gradients while limiting their magnitude.

    Args:
        params_and_grads: List of (parameter, gradient) tuples
        max_norm: Maximum allowed gradient norm

    Returns:
        The total norm before clipping (useful for monitoring)
    """
    # =========================================================================
    # Step 1: Compute total gradient norm
    # =========================================================================
    total_norm_sq = 0.0
    for param, grad in params_and_grads:
        if grad is not None:
            total_norm_sq += np.sum(grad ** 2)

    total_norm = np.sqrt(total_norm_sq)

    # =========================================================================
    # Step 2: Clip if necessary
    # =========================================================================
    if total_norm > max_norm:
        # Scale factor to bring norm down to max_norm
        scale = max_norm / (total_norm + 1e-8)

        for param, grad in params_and_grads:
            if grad is not None:
                grad *= scale

    return total_norm


def train_epoch(model, data_loader, loss_fn, optimizer, config):
    """
    Train for one epoch.

    An epoch is one complete pass through the training data.

    Args:
        model: TransformerLM model
        data_loader: DataLoader yielding (input, target) batches
        loss_fn: CrossEntropyLoss
        optimizer: SGD optimizer
        config: Configuration dictionary

    Returns:
        Average loss for the epoch
    """
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in data_loader:
        # =================================================================
        # Forward pass
        # =================================================================
        # Compute model predictions
        logits = model.forward(inputs)

        # Compute loss
        loss = loss_fn.forward(logits, targets)
        total_loss += loss
        num_batches += 1

        # =================================================================
        # Backward pass
        # =================================================================
        # Compute gradient of loss w.r.t. logits
        grad_logits = loss_fn.backward()

        # Backpropagate through model
        model.backward(grad_logits)

        # =================================================================
        # Gradient clipping
        # =================================================================
        params_and_grads = model.get_params_and_grads()
        clip_gradients(params_and_grads, config["max_grad_norm"])

        # =================================================================
        # Parameter update
        # =================================================================
        optimizer.step(params_and_grads)

    return total_loss / num_batches


def evaluate(model, data_loader, loss_fn):
    """
    Evaluate model on data.

    Similar to training, but no gradient computation or parameter updates.

    Args:
        model: TransformerLM model
        data_loader: DataLoader yielding (input, target) batches
        loss_fn: CrossEntropyLoss

    Returns:
        Average loss
    """
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in data_loader:
        # Forward pass only
        logits = model.forward(inputs)
        loss = loss_fn.forward(logits, targets)

        total_loss += loss
        num_batches += 1

    return total_loss / num_batches


def generate(model, start_tokens, max_len, idx_to_word, temperature=1.0):
    """
    Generate text autoregressively.

    Starting from seed tokens, repeatedly:
    1. Run forward pass to get next-token probabilities
    2. Sample from the distribution
    3. Append sampled token
    4. Repeat

    Args:
        model: Trained TransformerLM model
        start_tokens: List of starting token IDs
        max_len: Maximum number of tokens to generate
        idx_to_word: Vocabulary mapping (ID -> word)
        temperature: Sampling temperature
                     - Lower (e.g., 0.5): more deterministic, picks likely tokens
                     - Higher (e.g., 1.5): more random, more diverse outputs
                     - 1.0: sample from raw probabilities

    Returns:
        List of generated words
    """
    tokens = list(start_tokens)
    max_seq_len = model.config["max_seq_len"]

    for _ in range(max_len):
        # Get last max_seq_len tokens (model has limited context)
        context = tokens[-max_seq_len:]

        # Convert to array with batch dimension
        input_array = np.array([context])

        # Forward pass to get logits
        logits = model.forward(input_array)

        # Get logits for last position (next token prediction)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply temperature
        # Higher temperature = flatter distribution = more random
        # Lower temperature = sharper distribution = more deterministic
        next_logits = next_logits / temperature

        # Convert to probabilities with stable softmax
        next_logits = next_logits - np.max(next_logits)
        probs = np.exp(next_logits) / np.sum(np.exp(next_logits))

        # Sample from distribution
        next_token = np.random.choice(len(probs), p=probs)

        tokens.append(next_token)

    # Convert to words
    words = [idx_to_word.get(t, "<unk>") for t in tokens]

    return words


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Testing CrossEntropyLoss...")

    loss_fn = CrossEntropyLoss()

    # Create dummy data
    batch_size, seq_len, vocab_size = 2, 4, 10
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # Forward
    loss = loss_fn.forward(logits, targets)
    print(f"  Loss: {loss:.4f}")

    # Backward
    grad = loss_fn.backward()
    print(f"  Gradient shape: {grad.shape}")
    print(f"  Gradient range: [{grad.min():.4f}, {grad.max():.4f}]")

    # Verify: gradient should sum to ~0 for each position
    grad_sum = np.sum(grad, axis=-1)
    print(f"  Gradient sum per position (should be ~0): max abs = {np.max(np.abs(grad_sum)):.6f}")

    print("\nTesting SGD optimizer...")
    optimizer = SGD(learning_rate=0.1, momentum=0.9)

    # Create dummy parameter and gradient
    param = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.2, 0.3])

    print(f"  Before update: {param}")
    optimizer.step([(param, grad)])
    print(f"  After update: {param}")

    print("\nTesting gradient clipping...")
    params_and_grads = [
        (np.zeros(3), np.array([10.0, 0.0, 0.0])),
        (np.zeros(3), np.array([0.0, 10.0, 0.0])),
    ]
    norm_before = clip_gradients(params_and_grads, max_norm=1.0)
    print(f"  Norm before clipping: {norm_before:.4f}")

    # Check that norm is now <= 1.0
    norm_after = np.sqrt(sum(np.sum(g**2) for p, g in params_and_grads if g is not None))
    print(f"  Norm after clipping: {norm_after:.4f}")
