"""
Core Neural Network Layers from Scratch

This module implements fundamental layers used in transformers:
- Linear: Fully connected layer (matrix multiplication + bias)
- LayerNorm: Layer normalization for stable training
- Embedding: Token to vector lookup table
- PositionalEncoding: Adds position information to embeddings

Each layer implements:
- forward(x): Compute output from input
- backward(grad_output): Compute gradient w.r.t. input and parameters
- get_params_and_grads(): Return list of (parameter, gradient) tuples for optimizer
"""

import numpy as np


class Linear:
    """
    Fully Connected (Dense) Layer.

    Forward:
        y = x @ W + b

    Where:
        x: input of shape (batch, seq_len, in_features) or (batch, in_features)
        W: weight matrix of shape (in_features, out_features)
        b: bias vector of shape (out_features,)
        y: output of shape (batch, seq_len, out_features) or (batch, out_features)

    This is the fundamental building block of neural networks.
    It performs a linear transformation followed by an optional bias addition.

    Backward:
        Given dL/dy (gradient of loss w.r.t. output):
        - dL/dx = dL/dy @ W^T  (to propagate to previous layer)
        - dL/dW = x^T @ dL/dy  (to update weights)
        - dL/db = sum(dL/dy)   (to update bias)

    Weight Initialization:
        We use Xavier/Glorot initialization to maintain variance across layers:
        W ~ Normal(0, sqrt(2 / (fan_in + fan_out)))

        This prevents gradients from vanishing or exploding during training.
    """

    def __init__(self, in_features, out_features):
        """
        Initialize the linear layer.

        Args:
            in_features: Size of input dimension
            out_features: Size of output dimension
        """
        # Xavier initialization for stable training
        # The scale factor keeps the variance of activations roughly constant
        scale = np.sqrt(2.0 / (in_features + out_features))

        # Weight matrix: transforms in_features -> out_features
        self.W = np.random.randn(in_features, out_features) * scale

        # Bias vector: one bias per output feature, initialized to zero
        # Zero init for bias is standard practice
        self.b = np.zeros(out_features)

        # Gradient accumulators - will be set during backward pass
        self.grad_W = None
        self.grad_b = None

        # Cache for backward pass
        self.x = None

    def forward(self, x):
        """
        Compute linear transformation.

        Args:
            x: Input array of shape (..., in_features)
               Typically (batch, seq_len, in_features) for transformers

        Returns:
            Output of shape (..., out_features)
        """
        # Cache input for backward pass
        self.x = x

        # Matrix multiplication broadcasts over leading dimensions
        # (batch, seq, in) @ (in, out) -> (batch, seq, out)
        # Bias broadcasts: (out,) adds to each position
        return x @ self.W + self.b

    def backward(self, grad_output):
        """
        Compute gradients for backpropagation.

        Args:
            grad_output: Gradient of loss w.r.t. output (dL/dy)
                         Shape: (..., out_features)

        Returns:
            Gradient of loss w.r.t. input (dL/dx)
            Shape: (..., in_features)
        """
        # =====================================================================
        # Gradient w.r.t. input: dL/dx = dL/dy @ W^T
        # =====================================================================
        # This is what we pass to the previous layer
        # (batch, seq, out) @ (out, in) -> (batch, seq, in)
        grad_input = grad_output @ self.W.T

        # =====================================================================
        # Gradient w.r.t. weights: dL/dW = x^T @ dL/dy
        # =====================================================================
        # We need to sum over batch and sequence dimensions
        # Reshape to 2D for matrix multiplication
        x_2d = self.x.reshape(-1, self.x.shape[-1])  # (batch*seq, in)
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])  # (batch*seq, out)

        # (in, batch*seq) @ (batch*seq, out) -> (in, out)
        self.grad_W = x_2d.T @ grad_2d

        # =====================================================================
        # Gradient w.r.t. bias: dL/db = sum(dL/dy)
        # =====================================================================
        # Sum over all positions where bias was added
        # Bias affects every position equally, so we sum all gradients
        if grad_output.ndim == 3:
            self.grad_b = np.sum(grad_output, axis=(0, 1))  # Sum over batch and seq
        elif grad_output.ndim == 2:
            self.grad_b = np.sum(grad_output, axis=0)  # Sum over batch
        else:
            self.grad_b = grad_output.copy()

        return grad_input

    def get_params_and_grads(self):
        """Return parameters and their gradients for optimizer."""
        return [(self.W, self.grad_W), (self.b, self.grad_b)]


class LayerNorm:
    """
    Layer Normalization.

    Forward:
        y = gamma * (x - mean) / sqrt(var + eps) + beta

    Where:
        mean = mean(x, axis=-1)  (mean over features)
        var = var(x, axis=-1)   (variance over features)
        gamma: learnable scale parameter (initialized to 1)
        beta: learnable shift parameter (initialized to 0)
        eps: small constant for numerical stability

    Layer normalization normalizes each sample independently across its features.
    This stabilizes training by:
    1. Reducing internal covariate shift
    2. Making the network less sensitive to weight initialization
    3. Acting as a form of regularization

    Unlike BatchNorm, LayerNorm:
    - Normalizes across features, not batch
    - Works the same during training and inference
    - Is suitable for variable-length sequences (RNNs, Transformers)

    Backward:
        The gradient computation is complex because mean and variance
        are functions of all inputs. We must account for:
        1. Direct path through (x - mean) / std
        2. Indirect path through mean (affects all outputs)
        3. Indirect path through variance (affects all outputs)
    """

    def __init__(self, normalized_shape, eps=1e-8):
        """
        Initialize layer normalization.

        Args:
            normalized_shape: Size of the last dimension (features)
            eps: Small constant for numerical stability in division
        """
        self.eps = eps

        # Learnable parameters
        # gamma (scale): initialized to 1, allows network to scale normalized output
        self.gamma = np.ones(normalized_shape)

        # beta (shift): initialized to 0, allows network to shift normalized output
        self.beta = np.zeros(normalized_shape)

        # Gradient accumulators
        self.grad_gamma = None
        self.grad_beta = None

        # Cache for backward pass
        self.x = None
        self.mean = None
        self.var = None
        self.x_norm = None
        self.std_inv = None

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x: Input of shape (..., normalized_shape)
               Typically (batch, seq_len, embed_dim) for transformers

        Returns:
            Normalized output of same shape
        """
        # Cache input
        self.x = x

        # Step 1: Compute mean across last dimension (features)
        self.mean = np.mean(x, axis=-1, keepdims=True)

        # Step 2: Compute variance across last dimension
        self.var = np.var(x, axis=-1, keepdims=True)

        # Step 3: Compute inverse standard deviation (for efficiency)
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)

        # Step 4: Normalize: zero mean, unit variance
        self.x_norm = (x - self.mean) * self.std_inv

        # Step 5: Scale and shift with learnable parameters
        # This allows the network to undo normalization if needed
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output):
        """
        Compute gradients through layer normalization.

        This is mathematically involved because changing any input x_i
        affects the mean and variance, which in turn affects all outputs.

        Args:
            grad_output: Gradient of loss w.r.t. output (dL/dy)

        Returns:
            Gradient of loss w.r.t. input (dL/dx)
        """
        N = self.x.shape[-1]  # Number of features

        # =====================================================================
        # Gradients w.r.t. learnable parameters
        # =====================================================================

        # dL/d_gamma = sum(dL/dy * x_norm)
        # gamma scales each feature, so gradient is sum of (grad * normalized value)
        if grad_output.ndim == 3:
            self.grad_gamma = np.sum(grad_output * self.x_norm, axis=(0, 1))
            self.grad_beta = np.sum(grad_output, axis=(0, 1))
        else:
            self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0)
            self.grad_beta = np.sum(grad_output, axis=0)

        # =====================================================================
        # Gradient w.r.t. input (the complex part)
        # =====================================================================

        # First, compute gradient w.r.t. normalized x
        # dL/dx_norm = dL/dy * gamma (chain rule through the scaling)
        dx_norm = grad_output * self.gamma

        # Now we need dL/dx given dL/dx_norm
        # x_norm = (x - mean) * std_inv
        # This involves:
        # 1. Direct path: x -> x_norm
        # 2. Through mean: x -> mean -> x_norm
        # 3. Through variance: x -> var -> std_inv -> x_norm

        # Gradient through variance path:
        # d(std_inv)/d(var) = -0.5 * (var + eps)^(-3/2)
        # d(var)/d(x_i) = 2*(x_i - mean)/N
        x_centered = self.x - self.mean
        dvar = np.sum(dx_norm * x_centered * -0.5 * self.std_inv**3, axis=-1, keepdims=True)

        # Gradient through mean path:
        # d(mean)/d(x_i) = 1/N for all i
        # d(x_norm)/d(mean) = -std_inv + contribution from variance
        dmean = np.sum(dx_norm * -self.std_inv, axis=-1, keepdims=True)
        dmean += dvar * np.mean(-2.0 * x_centered, axis=-1, keepdims=True)

        # Combine all paths:
        # Direct path: dx_norm * std_inv
        # Mean path: dmean / N
        # Variance path: dvar * 2 * (x - mean) / N
        grad_input = dx_norm * self.std_inv
        grad_input += dvar * 2.0 * x_centered / N
        grad_input += dmean / N

        return grad_input

    def get_params_and_grads(self):
        """Return parameters and their gradients for optimizer."""
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]


class Embedding:
    """
    Token Embedding Layer.

    Maps discrete token indices to continuous vector representations.

    Forward:
        y = embedding_matrix[token_indices]

    This is simply a lookup table. For each token ID, we retrieve its
    corresponding embedding vector. No computation, just indexing.

    Why embeddings?
        - Tokens (words/characters) are discrete symbols
        - Neural networks need continuous inputs
        - Embeddings learn meaningful representations where
          similar tokens have similar vectors

    Backward:
        The gradient for an embedding is the sum of all gradients
        that "looked up" that embedding. We use scatter-add to
        accumulate gradients for each token.
    """

    def __init__(self, vocab_size, embed_dim):
        """
        Initialize embedding layer.

        Args:
            vocab_size: Number of unique tokens
            embed_dim: Dimension of embedding vectors
        """
        # Initialize embeddings with small random values
        # Small initialization prevents large initial activations
        self.weight = np.random.randn(vocab_size, embed_dim) * 0.02

        # Gradient accumulator
        self.grad_weight = None

        # Cache for backward pass
        self.x = None

    def forward(self, x):
        """
        Look up embeddings for token indices.

        Args:
            x: Integer array of token indices, shape (batch, seq_len)
               Values should be in [0, vocab_size)

        Returns:
            Embedding vectors, shape (batch, seq_len, embed_dim)
        """
        # Cache indices for backward pass
        self.x = x

        # Simple indexing: weight[x] gathers rows from the embedding matrix
        # If x has shape (batch, seq), result has shape (batch, seq, embed_dim)
        return self.weight[x]

    def backward(self, grad_output):
        """
        Compute gradient for embedding weights.

        Args:
            grad_output: Gradient of loss w.r.t. embeddings
                         Shape: (batch, seq_len, embed_dim)

        Returns:
            None (no gradient to propagate - input is discrete indices)
        """
        # Initialize gradient to zeros
        self.grad_weight = np.zeros_like(self.weight)

        # Scatter-add: accumulate gradients for each token
        # If token i appears multiple times, its gradient is the sum
        # of all positions where it appeared
        #
        # np.add.at performs unbuffered in-place addition:
        # for each (batch, seq) position, add the gradient to the
        # corresponding row in grad_weight
        np.add.at(self.grad_weight, self.x, grad_output)

        # No gradient to return - can't backprop through discrete indices
        return None

    def get_params_and_grads(self):
        """Return parameters and their gradients for optimizer."""
        return [(self.weight, self.grad_weight)]


class PositionalEncoding:
    """
    Sinusoidal Positional Encoding.

    Adds position information to token embeddings. Without this,
    the transformer would be permutation-invariant and couldn't
    distinguish "the cat sat" from "sat the cat".

    Forward:
        y = x + PE

    Where PE is pre-computed using sinusoidal functions:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sinusoidal?
        1. Unique encoding for each position
        2. Bounded values (between -1 and 1)
        3. Relative positions can be computed as linear functions
        4. Generalizes to longer sequences than seen during training

    The wavelengths form a geometric progression from 2*pi to 10000*2*pi.
    Lower dimensions capture fine-grained position info (nearby positions differ),
    higher dimensions capture coarse position info (changes slowly).

    This is NOT learned - it's fixed. No parameters, no gradients to track.
    """

    def __init__(self, max_seq_len, embed_dim):
        """
        Pre-compute positional encodings.

        Args:
            max_seq_len: Maximum sequence length to support
            embed_dim: Dimension of embeddings (must match token embeddings)
        """
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        position = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)

        # Create dimension indices for the exponential term
        # We use pairs (0, 1), (2, 3), ... for sin/cos
        dim_indices = np.arange(0, embed_dim, 2)  # [0, 2, 4, ...]

        # Compute the denominator: 10000^(2i/d_model)
        # Using log for numerical stability: 10000^x = exp(x * log(10000))
        div_term = np.exp(dim_indices * -(np.log(10000.0) / embed_dim))

        # Pre-compute positional encoding matrix
        self.pe = np.zeros((max_seq_len, embed_dim))

        # Even dimensions: sine
        self.pe[:, 0::2] = np.sin(position * div_term)

        # Odd dimensions: cosine
        # Handle case where embed_dim is odd
        if embed_dim % 2 == 0:
            self.pe[:, 1::2] = np.cos(position * div_term)
        else:
            self.pe[:, 1::2] = np.cos(position * div_term[:-1])

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Token embeddings, shape (batch, seq_len, embed_dim)

        Returns:
            Embeddings + positional encoding, same shape
        """
        seq_len = x.shape[1]

        # Add positional encoding (broadcasts over batch dimension)
        # We only use the first seq_len positions
        return x + self.pe[:seq_len]

    def backward(self, grad_output):
        """
        Pass gradient through unchanged.

        Positional encoding is a fixed addition, so the gradient
        just passes through. d(x + PE)/dx = 1.
        """
        return grad_output


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    from activations import check_gradient

    print("Testing Linear layer...")
    linear = Linear(8, 4)
    x = np.random.randn(2, 3, 8)  # (batch=2, seq=3, features=8)
    y = linear.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    check_gradient(linear, x.copy())

    print("\nTesting LayerNorm...")
    ln = LayerNorm(8)
    x = np.random.randn(2, 3, 8)
    y = ln.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output mean per position (should be ~0): {np.mean(y, axis=-1)}")
    print(f"  Output var per position (should be ~1): {np.var(y, axis=-1)}")
    check_gradient(ln, x.copy())

    print("\nTesting Embedding...")
    embed = Embedding(vocab_size=10, embed_dim=4)
    x = np.array([[0, 1, 2], [3, 4, 5]])  # (batch=2, seq=3)
    y = embed.forward(x)
    print(f"  Token indices shape: {x.shape}")
    print(f"  Embeddings shape: {y.shape}")
    print(f"  First token embedding: {y[0, 0]}")

    print("\nTesting PositionalEncoding...")
    pe = PositionalEncoding(max_seq_len=10, embed_dim=8)
    x = np.zeros((2, 5, 8))  # (batch=2, seq=5, embed=8)
    y = pe.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Position 0 encoding: {pe.pe[0]}")
    print(f"  Position 1 encoding: {pe.pe[1]}")
