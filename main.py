#!/usr/bin/env python3
"""
Transformer from Scratch - Main Entry Point

This script trains a small transformer language model on a simple poem
to demonstrate that our from-scratch implementation works.

The goal is educational: understand every component of a transformer
by implementing it without any ML frameworks.

What this demonstrates:
1. Token embedding and positional encoding
2. Multi-head self-attention mechanism
3. Feed-forward networks with residual connections
4. Layer normalization
5. Backpropagation through all components
6. Next-token prediction training
7. Autoregressive text generation

Usage:
    python main.py

The model will:
1. Create a vocabulary from the training poem
2. Build a small transformer (2 layers, 2 heads, 32-dim embeddings)
3. Train for 2000 epochs to memorize the poem
4. Generate text to verify it learned

Expected output:
- Loss should decrease from ~3.0 to <0.5
- Generated text should resemble the training poem
"""

import numpy as np

# Our from-scratch implementations
from config import CONFIG
from core.transformer import TransformerLM
from utils.data import POEM, create_vocabulary, create_training_data, DataLoader, detokenize
from train import CrossEntropyLoss, SGD, train_epoch, generate, clip_gradients


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def main():
    """Main training and evaluation loop."""

    print_separator("TRANSFORMER FROM SCRATCH")
    print("Training a small transformer to memorize a poem")
    print("Using only Python and NumPy - no ML frameworks!")

    # =========================================================================
    # Set random seed for reproducibility
    # =========================================================================
    np.random.seed(42)

    # =========================================================================
    # Step 1: Prepare Data
    # =========================================================================
    print_separator("STEP 1: Preparing Data")

    print(f"\nTraining text:\n{POEM}")

    # Create vocabulary
    word_to_idx, idx_to_word = create_vocabulary(POEM)
    vocab_size = len(word_to_idx)

    print(f"\nVocabulary ({vocab_size} words):")
    for word, idx in sorted(word_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: '{word}'")

    # Update config with actual vocab size
    config = CONFIG.copy()
    config["vocab_size"] = vocab_size

    # Create training sequences
    seq_len = min(config["max_seq_len"], 8)  # Use shorter sequences for this small task
    inputs, targets = create_training_data(POEM, word_to_idx, seq_len)

    print(f"\nTraining sequences (seq_len={seq_len}):")
    print(f"  Number of sequences: {len(inputs)}")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")

    # Show a few examples
    print("\nExample input -> target pairs:")
    for i in range(min(3, len(inputs))):
        input_text = detokenize(inputs[i], idx_to_word)
        target_text = detokenize(targets[i], idx_to_word)
        print(f"  {i+1}. '{input_text}' -> '{target_text}'")

    # Create data loader
    data_loader = DataLoader(inputs, targets, batch_size=config["batch_size"], shuffle=True)

    # =========================================================================
    # Step 2: Build Model
    # =========================================================================
    print_separator("STEP 2: Building Model")

    model = TransformerLM(config)

    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {config['vocab_size']}")
    print(f"  Embedding dimension: {config['embed_dim']}")
    print(f"  Number of heads: {config['num_heads']}")
    print(f"  Number of layers: {config['num_layers']}")
    print(f"  FFN dimension: {config['ffn_dim']}")
    print(f"  Max sequence length: {config['max_seq_len']}")

    num_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {num_params:,}")

    # =========================================================================
    # Step 3: Initialize Training
    # =========================================================================
    print_separator("STEP 3: Initializing Training")

    loss_fn = CrossEntropyLoss(config["eps"])
    optimizer = SGD(learning_rate=config["learning_rate"], momentum=config["momentum"])

    print(f"\nTraining Configuration:")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Momentum: {config['momentum']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient clipping: {config['max_grad_norm']}")

    # =========================================================================
    # Step 4: Training Loop
    # =========================================================================
    print_separator("STEP 4: Training")

    print("\nTraining progress:")
    print("-" * 40)

    losses = []
    log_every = config["epochs"] // 20  # Log 20 times during training

    for epoch in range(config["epochs"]):
        # Train for one epoch
        avg_loss = train_epoch(model, data_loader, loss_fn, optimizer, config)
        losses.append(avg_loss)

        # Log progress
        if epoch % log_every == 0 or epoch == config["epochs"] - 1:
            print(f"Epoch {epoch:4d}/{config['epochs']}  |  Loss: {avg_loss:.4f}")

            # Generate a sample every few logs to monitor progress
            if epoch > 0 and epoch % (log_every * 5) == 0:
                start_tokens = [word_to_idx.get("roses", 0)]
                generated = generate(model, start_tokens, max_len=12, idx_to_word=idx_to_word, temperature=0.8)
                print(f"              |  Sample: {' '.join(generated)}")

    print("-" * 40)
    print(f"\nTraining complete!")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

    # =========================================================================
    # Step 5: Evaluation and Generation
    # =========================================================================
    print_separator("STEP 5: Evaluation and Generation")

    print("\nGenerating text with different starting words:\n")

    # Test with different starting words
    test_starts = ["roses", "violets", "sugar", "and"]

    for start_word in test_starts:
        if start_word in word_to_idx:
            start_tokens = [word_to_idx[start_word]]

            # Generate with low temperature (more deterministic)
            generated_low = generate(model, start_tokens, max_len=10, idx_to_word=idx_to_word, temperature=0.5)

            # Generate with normal temperature
            generated_mid = generate(model, start_tokens, max_len=10, idx_to_word=idx_to_word, temperature=1.0)

            print(f"  Start: '{start_word}'")
            print(f"    temp=0.5: {' '.join(generated_low)}")
            print(f"    temp=1.0: {' '.join(generated_mid)}")
            print()

    # =========================================================================
    # Step 6: Verify Next-Token Prediction
    # =========================================================================
    print_separator("STEP 6: Verification")

    print("\nNext-token prediction accuracy:\n")

    # Check predictions on training data
    correct = 0
    total = 0

    for input_seq, target_seq in zip(inputs, targets):
        # Forward pass
        logits = model.forward(np.array([input_seq]))

        # Get predictions (argmax of logits)
        predictions = np.argmax(logits[0], axis=-1)

        # Count correct predictions
        correct += np.sum(predictions == target_seq)
        total += len(target_seq)

    accuracy = correct / total * 100
    print(f"  Training accuracy: {accuracy:.1f}% ({correct}/{total} tokens correct)")

    if accuracy > 90:
        print("\n  The model has successfully memorized the poem!")
    elif accuracy > 50:
        print("\n  The model is learning but could use more training.")
    else:
        print("\n  The model needs more training or hyperparameter tuning.")

    # Show some specific predictions
    print("\nSample predictions:")
    test_input = inputs[0:1]
    test_target = targets[0]
    logits = model.forward(test_input)
    predictions = np.argmax(logits[0], axis=-1)

    input_words = [idx_to_word[i] for i in test_input[0]]
    target_words = [idx_to_word[i] for i in test_target]
    pred_words = [idx_to_word[i] for i in predictions]

    print(f"  Input:      {' '.join(input_words)}")
    print(f"  Target:     {' '.join(target_words)}")
    print(f"  Predicted:  {' '.join(pred_words)}")

    print_separator("COMPLETE")
    print("\nYou've built a transformer from scratch!")
    print("Explore the code to understand how attention and backpropagation work.")


if __name__ == "__main__":
    main()
