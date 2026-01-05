"""
Transformer Block and Language Model from Scratch

This module implements:
- TransformerBlock: Single transformer layer (attention + FFN + residuals + norms)
- TransformerLM: Complete language model for next-token prediction

The transformer architecture was introduced in "Attention Is All You Need" (2017).
Key innovations:
1. Self-attention allows every position to directly attend to every other position
2. Residual connections help gradient flow through deep networks
3. Layer normalization stabilizes training
4. Position-wise feed-forward networks add non-linear transformations

We use the Pre-LN (Pre-LayerNorm) variant, which applies layer normalization
BEFORE each sub-layer rather than after. Pre-LN is more stable for training.
"""

import numpy as np
from .layers import Linear, LayerNorm, Embedding, PositionalEncoding
from .attention import MultiHeadAttention, create_causal_mask
from .activations import ReLU


class TransformerBlock:
    """
    Single Transformer Block (Layer).

    Architecture (Pre-LN variant):

        x ─────────────────────────────┐
        │                              │ (Residual Connection 1)
        ▼                              │
    [LayerNorm] ──► [MultiHeadAttn] ───+──► y1
        │                              │
        │                              │
        y1 ────────────────────────────┐
        │                              │ (Residual Connection 2)
        ▼                              │
    [LayerNorm] ──► [FFN] ─────────────+──► output

    Where FFN is:
        FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Residual Connections:
        Instead of output = f(input), we compute output = input + f(input)
        This helps gradients flow directly through the network without vanishing.
        Each layer can learn to make small modifications to the identity mapping.

    Layer Normalization:
        Applied before each sub-layer (Pre-LN) to stabilize training.
        Normalizes activations to have zero mean and unit variance.

    Why this structure works:
        1. Attention allows each position to gather information from all positions
        2. FFN allows each position to process information independently
        3. Residuals ensure information can flow unimpeded
        4. Stacking multiple blocks allows increasingly complex representations
    """

    def __init__(self, embed_dim, num_heads, ffn_dim, eps=1e-8):
        """
        Initialize transformer block.

        Args:
            embed_dim: Dimension of embeddings (d_model)
            num_heads: Number of attention heads
            ffn_dim: Hidden dimension of feed-forward network
            eps: Numerical stability constant for LayerNorm
        """
        # =====================================================================
        # First sub-layer: Multi-head attention
        # =====================================================================
        self.ln1 = LayerNorm(embed_dim, eps)
        self.attn = MultiHeadAttention(embed_dim, num_heads, eps)

        # =====================================================================
        # Second sub-layer: Position-wise Feed-Forward Network
        # =====================================================================
        self.ln2 = LayerNorm(embed_dim, eps)

        # FFN: Two linear layers with ReLU in between
        # First layer expands: embed_dim -> ffn_dim
        # Second layer projects back: ffn_dim -> embed_dim
        self.ffn_linear1 = Linear(embed_dim, ffn_dim)
        self.ffn_activation = ReLU()
        self.ffn_linear2 = Linear(ffn_dim, embed_dim)

        # Cache for backward pass
        self.x = None  # Original input
        self.x1 = None  # After first residual
        self.x_norm1 = None  # After first layer norm
        self.x_norm2 = None  # After second layer norm
        self.ffn_hidden = None  # After first FFN linear

    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.

        Args:
            x: Input embeddings, shape (batch, seq_len, embed_dim)
            mask: Optional attention mask, shape (seq_len, seq_len)

        Returns:
            Output embeddings, shape (batch, seq_len, embed_dim)
        """
        # Cache input for backward pass
        self.x = x

        # =====================================================================
        # Sub-layer 1: Attention with residual connection
        # =====================================================================

        # Pre-LayerNorm: normalize before attention
        self.x_norm1 = self.ln1.forward(x)

        # Multi-head self-attention
        attn_out = self.attn.forward(self.x_norm1, mask)

        # Residual connection: add input to attention output
        # This allows the network to learn "how much to change" from identity
        self.x1 = x + attn_out

        # =====================================================================
        # Sub-layer 2: Feed-forward network with residual connection
        # =====================================================================

        # Pre-LayerNorm: normalize before FFN
        self.x_norm2 = self.ln2.forward(self.x1)

        # Feed-forward network:
        # 1. Expand to higher dimension
        self.ffn_hidden = self.ffn_linear1.forward(self.x_norm2)

        # 2. Apply non-linearity (ReLU)
        ffn_act = self.ffn_activation.forward(self.ffn_hidden)

        # 3. Project back to embed_dim
        ffn_out = self.ffn_linear2.forward(ffn_act)

        # Residual connection
        output = self.x1 + ffn_out

        return output

    def backward(self, grad_output):
        """
        Backward pass through transformer block.

        We work backwards through the forward pass, carefully handling
        the residual connections (which split and merge gradients).

        Args:
            grad_output: Gradient w.r.t. output
                         Shape: (batch, seq_len, embed_dim)

        Returns:
            Gradient w.r.t. input
            Shape: (batch, seq_len, embed_dim)
        """
        # =====================================================================
        # Backward through second residual connection
        # =====================================================================
        # output = x1 + ffn_out
        # Gradient splits: flows to both x1 and ffn_out
        grad_ffn_out = grad_output  # Gradient to FFN path
        grad_x1_residual = grad_output  # Gradient to residual path

        # =====================================================================
        # Backward through FFN
        # =====================================================================
        # ffn_out = ffn_linear2(ffn_activation(ffn_linear1(x_norm2)))

        # Backward through ffn_linear2
        grad_ffn_act = self.ffn_linear2.backward(grad_ffn_out)

        # Backward through ReLU
        grad_ffn_hidden = self.ffn_activation.backward(grad_ffn_act)

        # Backward through ffn_linear1
        grad_x_norm2 = self.ffn_linear1.backward(grad_ffn_hidden)

        # Backward through ln2
        grad_x1_ln = self.ln2.backward(grad_x_norm2)

        # =====================================================================
        # Combine gradients at x1
        # =====================================================================
        # x1 receives gradient from both paths: residual and through ln2
        grad_x1 = grad_x1_residual + grad_x1_ln

        # =====================================================================
        # Backward through first residual connection
        # =====================================================================
        # x1 = x + attn_out
        # Gradient splits again
        grad_attn_out = grad_x1
        grad_x_residual = grad_x1

        # =====================================================================
        # Backward through attention
        # =====================================================================
        grad_x_norm1 = self.attn.backward(grad_attn_out)

        # Backward through ln1
        grad_x_ln = self.ln1.backward(grad_x_norm1)

        # =====================================================================
        # Combine gradients at input x
        # =====================================================================
        grad_input = grad_x_residual + grad_x_ln

        return grad_input

    def get_params_and_grads(self):
        """Collect all parameters and gradients."""
        params = []
        params.extend(self.ln1.get_params_and_grads())
        params.extend(self.attn.get_params_and_grads())
        params.extend(self.ln2.get_params_and_grads())
        params.extend(self.ffn_linear1.get_params_and_grads())
        params.extend(self.ffn_linear2.get_params_and_grads())
        return params


class TransformerLM:
    """
    Transformer Language Model for Next-Token Prediction.

    Complete architecture:

        Token IDs ──► [Embedding] ──► [PositionalEncoding] ──► [TransformerBlock] x N
                                                                      │
                                                                      ▼
                                         Logits ◄── [Linear] ◄── [LayerNorm]

    This is a decoder-only transformer (like GPT), designed to predict
    the next token given previous tokens.

    Training:
        Input:  [w1, w2, w3, w4]
        Target: [w2, w3, w4, w5]
        Loss:   Cross-entropy between predicted and actual next tokens

    Generation:
        Start with seed tokens, predict next token, append, repeat.

    The causal mask ensures position i only attends to positions j <= i,
    which is essential for autoregressive generation.
    """

    def __init__(self, config):
        """
        Initialize transformer language model.

        Args:
            config: Dictionary with model hyperparameters
                - vocab_size: Number of unique tokens
                - max_seq_len: Maximum sequence length
                - embed_dim: Embedding dimension
                - num_heads: Number of attention heads
                - num_layers: Number of transformer blocks
                - ffn_dim: FFN hidden dimension
                - eps: Numerical stability constant
        """
        self.config = config

        # =====================================================================
        # Token Embedding: Map discrete tokens to continuous vectors
        # =====================================================================
        self.embedding = Embedding(config["vocab_size"], config["embed_dim"])

        # =====================================================================
        # Positional Encoding: Add position information
        # =====================================================================
        self.pos_encoding = PositionalEncoding(
            config["max_seq_len"],
            config["embed_dim"]
        )

        # =====================================================================
        # Transformer Blocks: Stack of attention + FFN layers
        # =====================================================================
        self.blocks = [
            TransformerBlock(
                config["embed_dim"],
                config["num_heads"],
                config["ffn_dim"],
                config["eps"]
            )
            for _ in range(config["num_layers"])
        ]

        # =====================================================================
        # Final Layer Norm: Normalize before output projection
        # =====================================================================
        self.ln_final = LayerNorm(config["embed_dim"], config["eps"])

        # =====================================================================
        # Output Projection: Project to vocabulary size for prediction
        # =====================================================================
        # This produces logits (unnormalized log-probabilities) for each token
        self.output_proj = Linear(config["embed_dim"], config["vocab_size"])

        # =====================================================================
        # Causal Mask: Pre-compute for efficiency
        # =====================================================================
        self.causal_mask = create_causal_mask(config["max_seq_len"])

    def forward(self, x):
        """
        Forward pass: Compute logits for next-token prediction.

        Args:
            x: Token indices, shape (batch, seq_len)
               Values in [0, vocab_size)

        Returns:
            Logits, shape (batch, seq_len, vocab_size)
            logits[b, i, :] is the distribution for predicting position i+1
        """
        seq_len = x.shape[1]

        # =====================================================================
        # Step 1: Token Embedding
        # =====================================================================
        # Convert token IDs to embedding vectors
        h = self.embedding.forward(x)  # (batch, seq, embed_dim)

        # =====================================================================
        # Step 2: Add Positional Encoding
        # =====================================================================
        # Add position information so model knows token order
        h = self.pos_encoding.forward(h)  # (batch, seq, embed_dim)

        # =====================================================================
        # Step 3: Get causal mask for this sequence length
        # =====================================================================
        mask = self.causal_mask[:seq_len, :seq_len]

        # =====================================================================
        # Step 4: Pass through transformer blocks
        # =====================================================================
        # Each block refines the representations
        for block in self.blocks:
            h = block.forward(h, mask)  # (batch, seq, embed_dim)

        # =====================================================================
        # Step 5: Final layer normalization
        # =====================================================================
        h = self.ln_final.forward(h)  # (batch, seq, embed_dim)

        # =====================================================================
        # Step 6: Project to vocabulary
        # =====================================================================
        # Produce logits for each token in vocabulary
        logits = self.output_proj.forward(h)  # (batch, seq, vocab_size)

        return logits

    def backward(self, grad_logits):
        """
        Backward pass: Compute gradients for all parameters.

        Args:
            grad_logits: Gradient w.r.t. logits
                         Shape: (batch, seq_len, vocab_size)
        """
        # =====================================================================
        # Backward through output projection
        # =====================================================================
        grad = self.output_proj.backward(grad_logits)

        # =====================================================================
        # Backward through final layer norm
        # =====================================================================
        grad = self.ln_final.backward(grad)

        # =====================================================================
        # Backward through transformer blocks (in reverse order)
        # =====================================================================
        for block in reversed(self.blocks):
            grad = block.backward(grad)

        # =====================================================================
        # Backward through positional encoding
        # =====================================================================
        # Positional encoding is additive, gradient passes through
        grad = self.pos_encoding.backward(grad)

        # =====================================================================
        # Backward through embedding
        # =====================================================================
        # This updates embedding gradients but returns None
        # (can't backprop through discrete token indices)
        self.embedding.backward(grad)

    def get_params_and_grads(self):
        """Collect all parameters and gradients for optimizer."""
        params = []

        # Embedding parameters
        params.extend(self.embedding.get_params_and_grads())

        # Transformer block parameters
        for block in self.blocks:
            params.extend(block.get_params_and_grads())

        # Final layer norm parameters
        params.extend(self.ln_final.get_params_and_grads())

        # Output projection parameters
        params.extend(self.output_proj.get_params_and_grads())

        return params

    def count_parameters(self):
        """Count total number of trainable parameters."""
        total = 0
        for param, _ in self.get_params_and_grads():
            total += param.size
        return total


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test configuration
    config = {
        "vocab_size": 20,
        "max_seq_len": 10,
        "embed_dim": 16,
        "num_heads": 2,
        "num_layers": 2,
        "ffn_dim": 32,
        "eps": 1e-8,
    }

    print("Testing TransformerBlock...")
    block = TransformerBlock(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        eps=config["eps"]
    )

    x = np.random.randn(2, 5, config["embed_dim"])
    mask = create_causal_mask(5)

    y = block.forward(x, mask)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    # Test backward
    grad_output = np.random.randn(*y.shape)
    grad_input = block.backward(grad_output)
    print(f"  Gradient shape: {grad_input.shape}")

    print("\nTesting TransformerLM...")
    model = TransformerLM(config)

    # Create dummy input (token indices)
    x = np.random.randint(0, config["vocab_size"], size=(2, 8))
    print(f"  Input tokens: {x}")

    # Forward pass
    logits = model.forward(x)
    print(f"  Logits shape: {logits.shape}")  # Should be (2, 8, 20)

    # Check that logits are reasonable
    print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # Test backward
    grad_logits = np.random.randn(*logits.shape)
    model.backward(grad_logits)

    # Count parameters
    num_params = model.count_parameters()
    print(f"  Total parameters: {num_params}")

    # Check that gradients were computed
    params = model.get_params_and_grads()
    has_grads = all(g is not None for p, g in params)
    print(f"  All gradients computed: {has_grads}")
