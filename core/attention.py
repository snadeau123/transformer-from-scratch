"""
Self-Attention Mechanism from Scratch

This module implements the core attention mechanism of transformers:
- SingleHeadAttention: Basic scaled dot-product attention
- MultiHeadAttention: Multiple parallel attention heads

Attention is the key innovation that allows transformers to:
1. Look at all positions in a sequence simultaneously
2. Learn which positions are relevant to each other
3. Create context-aware representations

The formula:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Where:
    Q = Query: "What am I looking for?"
    K = Key: "What do I contain?"
    V = Value: "What information do I provide?"
    d_k = Key dimension (for scaling)
"""

import numpy as np
from .layers import Linear
from .activations import Softmax


class SingleHeadAttention:
    """
    Single-Head Scaled Dot-Product Attention.

    The attention mechanism works like a soft dictionary lookup:
    1. Query asks "what am I looking for?"
    2. Keys advertise "what do I have?"
    3. Attention scores = how well each key matches the query
    4. Output = weighted sum of values based on attention scores

    Step by step:
        1. Project input to Q, K, V:  Q = X @ W_q, K = X @ W_k, V = X @ W_v
        2. Compute attention scores:   scores = Q @ K^T
        3. Scale by sqrt(d_k):         scores = scores / sqrt(d_k)
        4. Apply mask (optional):      scores = scores + mask
        5. Apply softmax:              attn = softmax(scores)
        6. Apply to values:            output = attn @ V

    Why scale by sqrt(d_k)?
        Without scaling, the dot products grow large as d_k increases,
        pushing softmax into regions with tiny gradients.
        Scaling by sqrt(d_k) keeps the variance of the dot products at 1.

    Causal mask:
        For autoregressive generation, we mask future positions so
        position i can only attend to positions j <= i.
        We do this by adding -inf to masked positions before softmax.
    """

    def __init__(self, embed_dim, head_dim, eps=1e-8):
        """
        Initialize single attention head.

        Args:
            embed_dim: Input embedding dimension
            head_dim: Dimension of this attention head (typically embed_dim // num_heads)
            eps: Numerical stability constant
        """
        self.head_dim = head_dim
        self.eps = eps

        # Scaling factor: 1/sqrt(d_k) to normalize dot products
        self.scale = 1.0 / np.sqrt(head_dim)

        # Linear projections for Query, Key, Value
        # Each projects from embed_dim -> head_dim
        self.W_q = Linear(embed_dim, head_dim)
        self.W_k = Linear(embed_dim, head_dim)
        self.W_v = Linear(embed_dim, head_dim)

        # Softmax for attention weights
        self.softmax = Softmax(eps)

        # Cache for backward pass
        self.Q = None
        self.K = None
        self.V = None
        self.scores = None
        self.attn_weights = None

    def forward(self, x, mask=None):
        """
        Compute attention output.

        Args:
            x: Input embeddings, shape (batch, seq_len, embed_dim)
            mask: Optional attention mask, shape (seq_len, seq_len)
                  Contains 0 for allowed positions, -inf for masked positions

        Returns:
            Attention output, shape (batch, seq_len, head_dim)
        """
        # =====================================================================
        # Step 1: Compute Query, Key, Value projections
        # =====================================================================
        # Each position in the sequence gets its own Q, K, V vector
        # Q: "What am I looking for?"
        # K: "What do I contain that others might want?"
        # V: "What information should I contribute if attended to?"

        self.Q = self.W_q.forward(x)  # (batch, seq_len, head_dim)
        self.K = self.W_k.forward(x)  # (batch, seq_len, head_dim)
        self.V = self.W_v.forward(x)  # (batch, seq_len, head_dim)

        # =====================================================================
        # Step 2: Compute attention scores
        # =====================================================================
        # Score(i, j) = Q_i dot K_j = how much position i attends to position j
        # Q @ K^T: (batch, seq, head_dim) @ (batch, head_dim, seq) -> (batch, seq, seq)

        # Transpose K: swap last two dimensions
        K_T = self.K.transpose(0, 2, 1)  # (batch, head_dim, seq_len)

        # Batch matrix multiplication
        self.scores = np.matmul(self.Q, K_T)  # (batch, seq_len, seq_len)

        # =====================================================================
        # Step 3: Scale scores
        # =====================================================================
        # Without scaling, variance of scores grows with head_dim
        # This would make softmax very peaked (near one-hot), killing gradients
        self.scores = self.scores * self.scale

        # =====================================================================
        # Step 4: Apply mask (for causal attention)
        # =====================================================================
        # Mask is (seq_len, seq_len) with -inf where attention is forbidden
        # Adding -inf makes softmax output 0 for those positions
        if mask is not None:
            self.scores = self.scores + mask

        # =====================================================================
        # Step 5: Apply softmax to get attention weights
        # =====================================================================
        # Softmax along last axis: each query position gets a distribution over keys
        # attn_weights[b, i, :] sums to 1 (it's a probability distribution)
        self.attn_weights = self.softmax.forward(self.scores, axis=-1)
        # Shape: (batch, seq_len, seq_len)
        # attn_weights[b, i, j] = how much position i attends to position j

        # =====================================================================
        # Step 6: Apply attention to values
        # =====================================================================
        # Weighted sum of value vectors based on attention weights
        # output[i] = sum_j(attn_weights[i,j] * V[j])
        # (batch, seq, seq) @ (batch, seq, head_dim) -> (batch, seq, head_dim)
        output = np.matmul(self.attn_weights, self.V)

        return output

    def backward(self, grad_output):
        """
        Backpropagate through attention.

        This is complex because attention involves several matrix operations.
        We work backwards through the forward pass.

        Args:
            grad_output: Gradient w.r.t. attention output
                         Shape: (batch, seq_len, head_dim)

        Returns:
            Gradient w.r.t. input x
            Shape: (batch, seq_len, embed_dim)
        """
        # =====================================================================
        # Step 6 backward: attn_weights @ V -> output
        # =====================================================================
        # output = attn_weights @ V
        # d(output)/d(attn_weights) = grad_output @ V^T
        # d(output)/d(V) = attn_weights^T @ grad_output

        V_T = self.V.transpose(0, 2, 1)  # (batch, head_dim, seq)
        grad_attn = np.matmul(grad_output, V_T)  # (batch, seq, seq)

        attn_T = self.attn_weights.transpose(0, 2, 1)  # (batch, seq, seq)
        grad_V = np.matmul(attn_T, grad_output)  # (batch, seq, head_dim)

        # =====================================================================
        # Step 5 backward: softmax
        # =====================================================================
        grad_scores = self.softmax.backward(grad_attn)  # (batch, seq, seq)

        # =====================================================================
        # Step 4 backward: mask (if used)
        # =====================================================================
        # Mask is additive, gradient passes through unchanged
        # (masked positions had -inf, their gradients are 0 from softmax)

        # =====================================================================
        # Step 3 backward: scaling
        # =====================================================================
        grad_scores = grad_scores * self.scale

        # =====================================================================
        # Step 2 backward: Q @ K^T -> scores
        # =====================================================================
        # scores = Q @ K^T
        # d(scores)/d(Q) = grad_scores @ K
        # d(scores)/d(K) = grad_scores^T @ Q = Q^T @ grad_scores (then needs transpose)

        grad_Q = np.matmul(grad_scores, self.K)  # (batch, seq, head_dim)

        grad_scores_T = grad_scores.transpose(0, 2, 1)  # (batch, seq, seq)
        grad_K = np.matmul(grad_scores_T, self.Q)  # (batch, seq, head_dim)

        # =====================================================================
        # Step 1 backward: linear projections
        # =====================================================================
        # Q = x @ W_q, K = x @ W_k, V = x @ W_v
        # All three projections share the same input x
        # Total gradient = sum of gradients from Q, K, V paths

        grad_x_q = self.W_q.backward(grad_Q)  # (batch, seq, embed_dim)
        grad_x_k = self.W_k.backward(grad_K)  # (batch, seq, embed_dim)
        grad_x_v = self.W_v.backward(grad_V)  # (batch, seq, embed_dim)

        # Sum gradients because they all flow from the same input
        grad_input = grad_x_q + grad_x_k + grad_x_v

        return grad_input

    def get_params_and_grads(self):
        """Collect all parameters and gradients from projections."""
        params = []
        params.extend(self.W_q.get_params_and_grads())
        params.extend(self.W_k.get_params_and_grads())
        params.extend(self.W_v.get_params_and_grads())
        return params


class MultiHeadAttention:
    """
    Multi-Head Attention.

    Instead of one attention function, we run multiple "heads" in parallel.
    Each head can learn to focus on different types of relationships.

    For example:
        - Head 1 might learn syntactic relationships (subject-verb)
        - Head 2 might learn semantic relationships (synonyms)
        - Head 3 might learn positional patterns (nearby words)

    Architecture:
        1. Project input to h separate (Q, K, V) triples
        2. Run attention on each head independently
        3. Concatenate all head outputs
        4. Project concatenated output back to embed_dim

    Math:
        head_i = Attention(Q_i, K_i, V_i)
        MultiHead = Concat(head_1, ..., head_h) @ W_o

    Note: In optimized implementations, all heads are computed in a single
    batched operation. For clarity, we compute heads sequentially here.
    """

    def __init__(self, embed_dim, num_heads, eps=1e-8):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Input/output embedding dimension
            num_heads: Number of attention heads
            eps: Numerical stability constant

        Note: embed_dim must be divisible by num_heads
        """
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Create separate attention head for each head
        # Each head projects to head_dim dimensions
        self.heads = [
            SingleHeadAttention(embed_dim, self.head_dim, eps)
            for _ in range(num_heads)
        ]

        # Output projection: combines all heads back to embed_dim
        # Takes concatenated heads (num_heads * head_dim = embed_dim)
        # and projects to embed_dim
        self.W_o = Linear(embed_dim, embed_dim)

        # Cache for backward pass
        self.concat = None

    def forward(self, x, mask=None):
        """
        Compute multi-head attention.

        Args:
            x: Input embeddings, shape (batch, seq_len, embed_dim)
            mask: Optional attention mask, shape (seq_len, seq_len)

        Returns:
            Output embeddings, shape (batch, seq_len, embed_dim)
        """
        # =====================================================================
        # Run each attention head independently
        # =====================================================================
        # Each head sees the full input but has its own Q, K, V projections
        head_outputs = []
        for head in self.heads:
            # Each head output: (batch, seq_len, head_dim)
            head_out = head.forward(x, mask)
            head_outputs.append(head_out)

        # =====================================================================
        # Concatenate all heads along the feature dimension
        # =====================================================================
        # (batch, seq, head_dim) * num_heads -> (batch, seq, embed_dim)
        self.concat = np.concatenate(head_outputs, axis=-1)

        # =====================================================================
        # Project back to embed_dim
        # =====================================================================
        # This allows heads to interact and produces final output
        output = self.W_o.forward(self.concat)

        return output

    def backward(self, grad_output):
        """
        Backpropagate through multi-head attention.

        Args:
            grad_output: Gradient w.r.t. output
                         Shape: (batch, seq_len, embed_dim)

        Returns:
            Gradient w.r.t. input
            Shape: (batch, seq_len, embed_dim)
        """
        # =====================================================================
        # Backward through output projection
        # =====================================================================
        grad_concat = self.W_o.backward(grad_output)  # (batch, seq, embed_dim)

        # =====================================================================
        # Split gradient to each head
        # =====================================================================
        # grad_concat is (batch, seq, num_heads * head_dim)
        # Split into num_heads pieces of (batch, seq, head_dim) each
        grad_splits = np.split(grad_concat, self.num_heads, axis=-1)

        # =====================================================================
        # Backward through each head
        # =====================================================================
        grad_inputs = []
        for head, grad_split in zip(self.heads, grad_splits):
            grad_x = head.backward(grad_split)  # (batch, seq, embed_dim)
            grad_inputs.append(grad_x)

        # =====================================================================
        # Sum gradients from all heads
        # =====================================================================
        # All heads received the same input, so gradients add
        grad_input = sum(grad_inputs)

        return grad_input

    def get_params_and_grads(self):
        """Collect all parameters and gradients from all heads and output projection."""
        params = []
        for head in self.heads:
            params.extend(head.get_params_and_grads())
        params.extend(self.W_o.get_params_and_grads())
        return params


def create_causal_mask(seq_len):
    """
    Create causal (autoregressive) attention mask.

    This mask ensures that position i can only attend to positions j <= i.
    This is essential for training language models, where we predict
    the next token given previous tokens.

    The mask works by adding -inf to forbidden attention positions.
    After softmax, -inf becomes 0, effectively masking those positions.

    Args:
        seq_len: Length of the sequence

    Returns:
        Mask of shape (seq_len, seq_len)
        Lower triangular part is 0, upper triangular is -inf

    Example for seq_len=4:
        [[  0, -inf, -inf, -inf],
         [  0,    0, -inf, -inf],
         [  0,    0,    0, -inf],
         [  0,    0,    0,    0]]

        Row i shows which positions position i can attend to.
        Position 0 can only attend to itself.
        Position 3 can attend to all positions.
    """
    # np.triu creates upper triangular matrix
    # k=1 means start from first diagonal above main diagonal
    # Result: 1s above diagonal, 0s on and below
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)

    # Replace 1s with -inf using np.where to avoid 0 * -inf = nan
    mask = np.where(mask == 1, -np.inf, 0.0)

    return mask


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("Testing SingleHeadAttention...")
    attn = SingleHeadAttention(embed_dim=8, head_dim=4)
    x = np.random.randn(2, 5, 8)  # (batch=2, seq=5, embed=8)

    # Test without mask
    y = attn.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Attention weights shape: {attn.attn_weights.shape}")
    print(f"  Attention weights sum per query (should be 1): {attn.attn_weights[0].sum(axis=-1)}")

    # Test with causal mask
    mask = create_causal_mask(5)
    y_masked = attn.forward(x, mask)
    print(f"\n  With causal mask:")
    print(f"  Mask:\n{mask}")
    print(f"  Attention weights row 0 (can only attend to pos 0): {attn.attn_weights[0, 0]}")
    print(f"  Attention weights row 4 (can attend to all): {attn.attn_weights[0, 4]}")

    print("\nTesting MultiHeadAttention...")
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)
    x = np.random.randn(2, 5, 8)
    y = mha.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    # Test backward pass
    print("\n  Testing backward pass...")
    grad_output = np.random.randn(*y.shape)
    grad_input = mha.backward(grad_output)
    print(f"  Grad output shape: {grad_output.shape}")
    print(f"  Grad input shape: {grad_input.shape}")

    # Count parameters
    params = mha.get_params_and_grads()
    total_params = sum(p.size for p, g in params)
    print(f"  Total parameters: {total_params}")
