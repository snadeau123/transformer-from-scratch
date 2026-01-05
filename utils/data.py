"""
Data Utilities for Transformer Training

This module handles:
- Vocabulary creation (word-to-index and index-to-word mappings)
- Training data preparation (input-target pairs for next-token prediction)
- Sample poem for training

For a production system, you would use more sophisticated tokenization
(like BPE or WordPiece). Here we use simple word-level tokenization
for clarity.
"""

import numpy as np


# =============================================================================
# TRAINING DATA: A Simple Poem
# =============================================================================
# We'll train the model to memorize and complete this short poem.
# With ~20 tokens, a small transformer should be able to memorize it
# and predict next words given context.

POEM = """
roses are red
violets are blue
sugar is sweet
and so are you
"""


# =============================================================================
# VOCABULARY
# =============================================================================

def create_vocabulary(text):
    """
    Create vocabulary mappings from text.

    This creates:
    - word_to_idx: Dictionary mapping words to integer IDs
    - idx_to_word: Dictionary mapping IDs back to words

    We add special tokens:
    - <pad>: Padding token (index 0)
    - <unk>: Unknown token for words not in vocabulary (index 1)

    Args:
        text: String of training text

    Returns:
        word_to_idx: Dict mapping word -> integer ID
        idx_to_word: Dict mapping integer ID -> word
    """
    # Clean and tokenize: lowercase, split on whitespace
    words = text.lower().split()

    # Get unique words, maintaining a consistent order
    unique_words = sorted(list(set(words)))

    # Special tokens at the beginning
    special_tokens = ["<pad>", "<unk>"]

    # Build vocabulary: special tokens + unique words
    vocab = special_tokens + unique_words

    # Create bidirectional mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return word_to_idx, idx_to_word


def tokenize(text, word_to_idx):
    """
    Convert text to token IDs.

    Args:
        text: String to tokenize
        word_to_idx: Vocabulary mapping

    Returns:
        List of integer token IDs
    """
    words = text.lower().split()
    # Use <unk> token for words not in vocabulary
    unk_idx = word_to_idx["<unk>"]
    return [word_to_idx.get(word, unk_idx) for word in words]


def detokenize(token_ids, idx_to_word):
    """
    Convert token IDs back to text.

    Args:
        token_ids: List or array of integer token IDs
        idx_to_word: Reverse vocabulary mapping

    Returns:
        String of words joined by spaces
    """
    words = [idx_to_word.get(idx, "<unk>") for idx in token_ids]
    return " ".join(words)


# =============================================================================
# TRAINING DATA CREATION
# =============================================================================

def create_training_data(text, word_to_idx, seq_len):
    """
    Create input-target pairs for next-token prediction.

    For language modeling, we predict the next token at each position:
        Input:  [w1, w2, w3, w4]
        Target: [w2, w3, w4, w5]

    We create overlapping sequences from the text using a sliding window.

    Args:
        text: Training text
        word_to_idx: Vocabulary mapping
        seq_len: Length of each training sequence

    Returns:
        inputs: Array of shape (num_sequences, seq_len)
        targets: Array of shape (num_sequences, seq_len)

    Example:
        Text: "roses are red violets are blue"
        seq_len: 3

        Creates pairs like:
            Input:  [roses, are, red]      Target: [are, red, violets]
            Input:  [are, red, violets]    Target: [red, violets, are]
            Input:  [red, violets, are]    Target: [violets, are, blue]
    """
    # Tokenize the text
    token_ids = tokenize(text, word_to_idx)

    # Create sequences using sliding window
    inputs = []
    targets = []

    # We need at least seq_len + 1 tokens for one input-target pair
    for i in range(len(token_ids) - seq_len):
        # Input: tokens from i to i + seq_len
        input_seq = token_ids[i:i + seq_len]

        # Target: tokens from i+1 to i + seq_len + 1 (shifted by 1)
        target_seq = token_ids[i + 1:i + seq_len + 1]

        inputs.append(input_seq)
        targets.append(target_seq)

    return np.array(inputs), np.array(targets)


class DataLoader:
    """
    Simple data loader for iterating over training data.

    This handles:
    - Batching (grouping sequences together)
    - Shuffling (randomizing order each epoch)

    For our small example, we typically use batch_size=1.
    """

    def __init__(self, inputs, targets, batch_size=1, shuffle=True):
        """
        Initialize data loader.

        Args:
            inputs: Array of input sequences, shape (num_sequences, seq_len)
            targets: Array of target sequences, shape (num_sequences, seq_len)
            batch_size: Number of sequences per batch
            shuffle: Whether to shuffle data each epoch
        """
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(inputs)

    def __iter__(self):
        """Iterate over batches."""
        # Create indices
        indices = np.arange(self.num_samples)

        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(indices)

        # Yield batches
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            batch_inputs = self.inputs[batch_indices]
            batch_targets = self.targets[batch_indices]

            yield batch_inputs, batch_targets

    def __len__(self):
        """Number of batches per epoch."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Utilities")
    print("=" * 60)

    print("\nOriginal poem:")
    print(POEM)

    # Create vocabulary
    word_to_idx, idx_to_word = create_vocabulary(POEM)
    print(f"\nVocabulary size: {len(word_to_idx)}")
    print(f"Vocabulary: {word_to_idx}")

    # Tokenize
    tokens = tokenize(POEM, word_to_idx)
    print(f"\nTokenized: {tokens}")

    # Detokenize
    text = detokenize(tokens, idx_to_word)
    print(f"Detokenized: {text}")

    # Create training data
    seq_len = 4
    inputs, targets = create_training_data(POEM, word_to_idx, seq_len)
    print(f"\nTraining data (seq_len={seq_len}):")
    print(f"  Number of sequences: {len(inputs)}")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")

    # Show a few examples
    print("\nSample input-target pairs:")
    for i in range(min(3, len(inputs))):
        input_text = detokenize(inputs[i], idx_to_word)
        target_text = detokenize(targets[i], idx_to_word)
        print(f"  Input:  {input_text}")
        print(f"  Target: {target_text}")
        print()

    # Test DataLoader
    print("Testing DataLoader:")
    loader = DataLoader(inputs, targets, batch_size=2, shuffle=False)
    print(f"  Number of batches: {len(loader)}")

    for batch_idx, (batch_in, batch_out) in enumerate(loader):
        print(f"  Batch {batch_idx}: input shape {batch_in.shape}, target shape {batch_out.shape}")
        if batch_idx >= 1:
            break
