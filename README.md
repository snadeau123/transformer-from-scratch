# Transformer from Scratch

A complete transformer/attention architecture implemented from scratch using only Python and NumPy. No ML frameworks - every component (neurons, activations, backpropagation) is built from the ground up for educational purposes.

## Features

- **From-scratch implementation**: Linear layers, LayerNorm, Embeddings, Positional Encoding
- **Full attention mechanism**: Scaled dot-product attention with causal masking
- **Multi-head attention**: Parallel attention heads
- **Complete backpropagation**: Gradients flow through all components
- **Training utilities**: Cross-entropy loss, SGD with momentum, gradient clipping

## Architecture

```
Token IDs → [Embedding] → [PositionalEncoding] → [TransformerBlock x 2] → [LayerNorm] → [Linear] → Logits
```

Each TransformerBlock contains:
- Multi-head self-attention (2 heads)
- Feed-forward network (ReLU activation)
- Residual connections and layer normalization (Pre-LN)

## Quick Start

```bash
# Requires only NumPy
pip install numpy

# Run training
python main.py
```

## Project Structure

```
├── core/
│   ├── activations.py   # ReLU, Softmax
│   ├── layers.py        # Linear, LayerNorm, Embedding, PositionalEncoding
│   ├── attention.py     # SingleHeadAttention, MultiHeadAttention
│   └── transformer.py   # TransformerBlock, TransformerLM
├── utils/
│   └── data.py          # Vocabulary, tokenization
├── config.py            # Hyperparameters
├── train.py             # Loss, optimizer, training loop
└── main.py              # Entry point
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedding dim | 32 |
| Attention heads | 2 |
| Layers | 2 |
| FFN dim | 64 |
| Parameters | ~18K |

## Results

The model learns to complete a simple poem:

```
Training accuracy: 97.5%
Loss: 2.63 → 0.04 (98.5% improvement)
```

## License

MIT
