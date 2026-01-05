"""
Configuration for the from-scratch transformer.

These hyperparameters are intentionally small to allow training on CPU
while still demonstrating all key transformer concepts.
"""

CONFIG = {
    # ==========================================================================
    # MODEL ARCHITECTURE
    # ==========================================================================

    # Vocabulary size: number of unique tokens the model can handle
    # We use a small vocab (~20-30 words) from our training poem
    "vocab_size": 30,

    # Maximum sequence length the model can process
    # Longer sequences = more memory and computation
    "max_seq_len": 16,

    # Embedding dimension (d_model in the paper)
    # This is the size of the vector representing each token
    # All layers operate on vectors of this dimension
    "embed_dim": 32,

    # Number of attention heads
    # Multi-head attention allows the model to attend to different
    # types of relationships simultaneously
    # embed_dim must be divisible by num_heads
    "num_heads": 2,

    # Dimension of each attention head
    # head_dim = embed_dim // num_heads
    "head_dim": 16,

    # Number of transformer blocks (layers)
    # Each block contains: attention -> FFN (with residuals and layer norm)
    # More layers = deeper representations but slower training
    "num_layers": 2,

    # Feed-forward network hidden dimension
    # The FFN expands to this size then projects back to embed_dim
    # Typically 4x embed_dim in real transformers, we use 2x for simplicity
    "ffn_dim": 64,

    # ==========================================================================
    # TRAINING HYPERPARAMETERS
    # ==========================================================================

    # Learning rate for gradient descent
    # Too high = unstable training, too low = slow convergence
    "learning_rate": 0.01,

    # Momentum for SGD optimizer
    # Helps smooth out gradient updates and escape local minima
    "momentum": 0.9,

    # Number of training epochs
    # We need many epochs to memorize our small poem
    "epochs": 2000,

    # Batch size (number of sequences per update)
    # We use 1 for simplicity - processes one sequence at a time
    "batch_size": 1,

    # ==========================================================================
    # NUMERICAL STABILITY
    # ==========================================================================

    # Small epsilon to prevent division by zero
    # Used in LayerNorm and Softmax
    "eps": 1e-8,

    # Maximum gradient norm for clipping
    # Prevents exploding gradients during training
    "max_grad_norm": 1.0,
}
