# Understanding Transformers: A Complete Walkthrough

This guide explains the transformer architecture from first principles and walks you through our implementation. No prior knowledge of the original paper is required.

## Table of Contents

1. [What is a Transformer?](#what-is-a-transformer)
2. [The Big Picture](#the-big-picture)
3. [Step 1: Turning Words into Numbers](#step-1-turning-words-into-numbers)
4. [Step 2: Adding Position Information](#step-2-adding-position-information)
5. [Step 3: The Attention Mechanism](#step-3-the-attention-mechanism)
6. [Step 4: Multi-Head Attention](#step-4-multi-head-attention)
7. [Step 5: Feed-Forward Networks](#step-5-feed-forward-networks)
8. [Step 6: Residual Connections & Layer Normalization](#step-6-residual-connections--layer-normalization)
9. [Step 7: The Complete Transformer Block](#step-7-the-complete-transformer-block)
10. [Step 8: Stacking Blocks](#step-8-stacking-blocks)
11. [Step 9: Making Predictions](#step-9-making-predictions)
12. [Step 10: Training the Model](#step-10-training-the-model)
13. [Step 11: Generating Text](#step-11-generating-text)
14. [Putting It All Together](#putting-it-all-together)

---

## What is a Transformer?

A transformer is a neural network architecture designed to process sequences (like sentences). Before transformers, we used recurrent neural networks (RNNs) which processed words one at a time, left to right. This was slow and made it hard to learn relationships between distant words.

Transformers changed everything with one key innovation: **attention**. Instead of processing words sequentially, a transformer looks at all words simultaneously and learns which words are relevant to each other.

For example, in "The cat sat on the mat because it was tired":
- To understand "it", we need to connect it to "cat"
- Attention learns this connection automatically

### Why "Transformer"?

The name comes from how it *transforms* input representations. Each layer takes the input and transforms it into increasingly useful representations for the task at hand.

---

## The Big Picture

Here's what happens when we process the sentence "roses are red":

```
"roses are red"
       ↓
┌─────────────────────────────────────────┐
│  1. TOKENIZATION                        │
│     "roses" → 7, "are" → 3, "red" → 6   │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  2. EMBEDDING                           │
│     7 → [0.12, -0.34, 0.56, ...]       │
│     Each token becomes a 32-dim vector  │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  3. POSITIONAL ENCODING                 │
│     Add position info to embeddings     │
│     So model knows word order           │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  4. TRANSFORMER BLOCKS (×2)             │
│     ┌─────────────────────────────────┐ │
│     │ Attention: words look at each   │ │
│     │ other to gather context         │ │
│     ├─────────────────────────────────┤ │
│     │ Feed-Forward: process each      │ │
│     │ position independently          │ │
│     └─────────────────────────────────┘ │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  5. OUTPUT PROJECTION                   │
│     Convert to vocabulary-sized scores  │
│     Predict next word probabilities     │
└─────────────────────────────────────────┘
       ↓
"violets" (predicted next word)
```

---

## Step 1: Turning Words into Numbers

Neural networks only understand numbers, so we need to convert words to numerical representations.

### Vocabulary

First, we create a vocabulary - a mapping from words to unique integer IDs:

```python
# From utils/data.py, lines 45-67

def create_vocabulary(text):
    # Clean and tokenize: lowercase, split on whitespace
    words = text.lower().split()

    # Get unique words
    unique_words = sorted(list(set(words)))

    # Special tokens at the beginning
    special_tokens = ["<pad>", "<unk>"]

    # Build vocabulary
    vocab = special_tokens + unique_words

    # Create mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return word_to_idx, idx_to_word
```

For our poem, this creates:
```
<pad>: 0, <unk>: 1, and: 2, are: 3, blue: 4, is: 5,
red: 6, roses: 7, so: 8, sugar: 9, sweet: 10, violets: 11, you: 12
```

### Embeddings

Raw token IDs (like 7 for "roses") don't carry meaning. We need to convert them to dense vectors where similar words have similar representations.

```python
# From core/layers.py, lines 188-230

class Embedding:
    """
    Maps discrete token indices to continuous vector representations.
    This is simply a lookup table.
    """

    def __init__(self, vocab_size, embed_dim):
        # Initialize embeddings with small random values
        self.weight = np.random.randn(vocab_size, embed_dim) * 0.02

    def forward(self, x):
        """
        x: integer indices of shape (batch, seq_len)
        Returns: (batch, seq_len, embed_dim)
        """
        # Simple indexing: weight[x] gathers rows from the embedding matrix
        return self.weight[x]
```

**How it works:**
- We have a matrix of shape `(vocab_size, embed_dim)` = `(13, 32)`
- Each row is the embedding for one word
- Looking up token 7 ("roses") gives us row 7: a 32-dimensional vector

**Why it's learnable:**
- Initially, embeddings are random
- During training, backpropagation updates them
- Words that appear in similar contexts develop similar embeddings

```python
# From core/layers.py, lines 232-248

def backward(self, grad_output):
    """
    Accumulate gradients for each embedding vector.
    """
    self.grad_weight = np.zeros_like(self.weight)

    # Scatter-add: accumulate gradients for each token
    # If token i appears multiple times, its gradient is the sum
    np.add.at(self.grad_weight, self.x, grad_output)

    return None  # Can't backprop through discrete indices
```

---

## Step 2: Adding Position Information

Attention treats all positions equally - it doesn't know that "roses" comes before "are". We fix this by adding positional information to embeddings.

### The Problem

Consider these sentences:
- "The cat chased the dog"
- "The dog chased the cat"

Same words, completely different meanings! Position matters.

### The Solution: Sinusoidal Encoding

We add a unique pattern to each position using sine and cosine waves:

```python
# From core/layers.py, lines 251-298

class PositionalEncoding:
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, max_seq_len, embed_dim):
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        position = np.arange(max_seq_len)[:, np.newaxis]

        # Compute wavelengths
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

        # Pre-compute positional encoding matrix
        self.pe = np.zeros((max_seq_len, embed_dim))
        self.pe[:, 0::2] = np.sin(position * div_term)  # Even dimensions
        self.pe[:, 1::2] = np.cos(position * div_term)  # Odd dimensions

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]  # Add position info
```

**Why sine and cosine?**
1. **Bounded values**: Always between -1 and 1
2. **Unique patterns**: Each position has a distinct encoding
3. **Relative positions**: The model can learn to compute relative positions as linear functions

**Visualization:**
```
Position 0: [sin(0), cos(0), sin(0), cos(0), ...]  = [0, 1, 0, 1, ...]
Position 1: [sin(1), cos(1), sin(0.1), cos(0.1), ...] = [0.84, 0.54, 0.10, 0.99, ...]
Position 2: [sin(2), cos(2), sin(0.2), cos(0.2), ...] = [0.91, -0.42, 0.20, 0.98, ...]
```

Different dimensions use different frequencies - low dimensions change fast (capturing fine position), high dimensions change slowly (capturing coarse position).

---

## Step 3: The Attention Mechanism

This is the heart of the transformer. Attention allows each word to "look at" every other word and decide how much to focus on each.

### Intuition: A Soft Dictionary Lookup

Imagine you're reading and encounter the word "it". To understand what "it" refers to, you look back at previous words. Attention automates this process:

1. **Query (Q)**: "What am I looking for?" (the current word asking a question)
2. **Key (K)**: "What do I contain?" (each word advertising its content)
3. **Value (V)**: "What information can I provide?" (the actual content to retrieve)

### The Math

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

Let's break this down step by step:

```python
# From core/attention.py, lines 60-130

class SingleHeadAttention:
    def __init__(self, embed_dim, head_dim):
        self.scale = 1.0 / np.sqrt(head_dim)

        # Linear projections for Q, K, V
        self.W_q = Linear(embed_dim, head_dim)
        self.W_k = Linear(embed_dim, head_dim)
        self.W_v = Linear(embed_dim, head_dim)

    def forward(self, x, mask=None):
        # Step 1: Project input to Q, K, V
        Q = self.W_q.forward(x)  # (batch, seq, head_dim)
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)

        # Step 2: Compute attention scores
        # Q × K^T gives similarity between all pairs of positions
        scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq, seq)

        # Step 3: Scale by √d_k
        # Without scaling, dot products grow large, making softmax very peaked
        scores = scores * self.scale

        # Step 4: Apply causal mask (for autoregressive generation)
        if mask is not None:
            scores = scores + mask  # mask has -inf for future positions

        # Step 5: Softmax converts scores to probabilities
        attn_weights = softmax(scores, axis=-1)  # (batch, seq, seq)

        # Step 6: Weighted sum of values
        output = np.matmul(attn_weights, V)  # (batch, seq, head_dim)

        return output
```

### Worked Example

Let's trace through attention for "the cat sat":

**Step 1: Create Q, K, V** (simplified to 2D for illustration)
```
Word    | Query (what I want) | Key (what I have) | Value (my content)
--------|---------------------|-------------------|-------------------
"the"   | [0.2, 0.8]         | [0.1, 0.9]        | [0.5, 0.3]
"cat"   | [0.9, 0.1]         | [0.8, 0.2]        | [0.7, 0.4]
"sat"   | [0.7, 0.3]         | [0.6, 0.4]        | [0.2, 0.9]
```

**Step 2: Compute scores** (Q × K^T)
For simplicity, we'll skip the √d_k scaling here; it just rescales the numbers.
```
Scores matrix (who attends to whom):
              "the"  "cat"  "sat"
    "the"  [  0.74,  0.32,  0.44 ]   ← "the" mostly attends to itself
    "cat"  [  0.18,  0.74,  0.58 ]   ← "cat" attends to itself and "sat"
    "sat"  [  0.34,  0.62,  0.54 ]   ← "sat" attends to "cat"
```

**Step 3: Apply softmax** (convert to probabilities)
```
Attention weights (rows sum to 1):
              "the"  "cat"  "sat"
    "the"  [  0.42,  0.27,  0.31 ]
    "cat"  [  0.24,  0.41,  0.35 ]
    "sat"  [  0.28,  0.37,  0.34 ]
```

**Step 4: Weighted sum of values**
```
Output for "cat" = 0.24 × V_the + 0.41 × V_cat + 0.35 × V_sat
                 = 0.24 × [0.5, 0.3] + 0.41 × [0.7, 0.4] + 0.35 × [0.2, 0.9]
                 = [0.48, 0.55]
```

The output for "cat" is now a blend of all words, weighted by relevance!

### Causal Masking

For language modeling, we can't let words see the future. We mask future positions:

```python
# From core/attention.py, lines 383-418

def create_causal_mask(seq_len):
    """
    Create mask where position i can only attend to positions j <= i.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask == 1, -np.inf, 0.0)
    return mask
```

This creates:
```
Position 0 sees: [  0, -∞, -∞ ]  → only itself
Position 1 sees: [  0,  0, -∞ ]  → positions 0 and 1
Position 2 sees: [  0,  0,  0 ]  → all positions
```

Adding `-∞` before softmax makes those positions have probability 0.

---

## Step 4: Multi-Head Attention

One attention head can only focus on one type of relationship. Multi-head attention runs several attention heads in parallel, each learning different patterns.

```python
# From core/attention.py, lines 230-280

class MultiHeadAttention:
    """
    Multiple attention heads in parallel.
    - Head 1 might learn syntactic relationships (subject-verb)
    - Head 2 might learn semantic relationships (synonyms)
    """

    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Split embedding across heads

        # Each head has its own Q, K, V projections
        self.heads = [
            SingleHeadAttention(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ]

        # Combine heads back to embed_dim
        self.W_o = Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # Run each head independently
        head_outputs = [head.forward(x, mask) for head in self.heads]

        # Concatenate: (batch, seq, head_dim) × num_heads → (batch, seq, embed_dim)
        concat = np.concatenate(head_outputs, axis=-1)

        # Project back
        return self.W_o.forward(concat)
```

**Why multiple heads?**

Different heads can specialize:
- Head 1: "For nouns, attend to their adjectives"
- Head 2: "For verbs, attend to their subjects"

The final output combines insights from all heads.

---

## Step 5: Feed-Forward Networks

After attention, each position is processed independently through a feed-forward network (FFN):

```python
# Part of TransformerBlock in core/transformer.py

# FFN architecture
self.ffn_linear1 = Linear(embed_dim, ffn_dim)   # Expand: 32 → 64
self.ffn_activation = ReLU()
self.ffn_linear2 = Linear(ffn_dim, embed_dim)   # Contract: 64 → 32
```

**The pattern: Expand → Non-linearity → Contract**

```
Input:  [batch, seq, 32]
         ↓
Linear1: [batch, seq, 64]  ← Expand to higher dimension
         ↓
ReLU:    [batch, seq, 64]  ← Apply non-linearity
         ↓
Linear2: [batch, seq, 32]  ← Contract back
```

**Why this helps:**
1. **More capacity**: The expanded dimension allows learning complex patterns
2. **Non-linearity**: ReLU lets the model learn non-linear relationships
3. **Position-wise**: Each position is processed independently (no mixing)

### ReLU Activation

```python
# From core/activations.py, lines 25-60

class ReLU:
    """
    y = max(0, x)

    Simple but effective:
    - Positive inputs pass through unchanged
    - Negative inputs become 0
    """

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        # Gradient is 1 for positive, 0 for negative
        return grad_output * (self.cache > 0)
```

---

## Step 6: Residual Connections & Layer Normalization

Two techniques that make deep networks trainable:

### Residual Connections

Instead of `output = f(input)`, we use `output = input + f(input)`:

```python
# From core/transformer.py, lines 85-95

# In the forward pass:
attn_out = self.attn.forward(x_norm1, mask)
self.x1 = x + attn_out  # ← Residual connection!

ffn_out = self.ffn_linear2.forward(ffn_act)
output = self.x1 + ffn_out  # ← Another residual!
```

**Why residuals help:**
1. **Gradient flow**: Gradients can flow directly through the addition
2. **Easy to learn**: The network can learn to make small modifications to identity
3. **Depth**: Enables training much deeper networks

### Layer Normalization

Normalizes activations to have zero mean and unit variance:

```python
# From core/layers.py, lines 85-145

class LayerNorm:
    """
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def forward(self, x):
        # Compute mean and variance across features
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)

        # Scale and shift with learnable parameters
        return self.gamma * self.x_norm + self.beta
```

**Why normalize?**
1. **Stable training**: Keeps activations in a reasonable range
2. **Faster convergence**: Normalized inputs are easier to optimize
3. **Less sensitivity**: Reduces dependence on initialization

### Pre-LN vs Post-LN

We use **Pre-LN** (normalize before each sub-layer):

```
x → LayerNorm → Attention → + → LayerNorm → FFN → +
    ↑________________________|   ↑_________________|
         residual                     residual
```

Pre-LN is more stable than Post-LN (normalize after) for training.

---

## Step 7: The Complete Transformer Block

Now let's put it all together:

```python
# From core/transformer.py, lines 40-115

class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ffn_dim):
        # Sub-layer 1: Attention
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)

        # Sub-layer 2: Feed-Forward
        self.ln2 = LayerNorm(embed_dim)
        self.ffn_linear1 = Linear(embed_dim, ffn_dim)
        self.ffn_activation = ReLU()
        self.ffn_linear2 = Linear(ffn_dim, embed_dim)

    def forward(self, x, mask=None):
        # ===== Sub-layer 1: Attention =====
        x_norm1 = self.ln1.forward(x)           # Normalize
        attn_out = self.attn.forward(x_norm1, mask)  # Attend
        x1 = x + attn_out                       # Residual

        # ===== Sub-layer 2: FFN =====
        x_norm2 = self.ln2.forward(x1)          # Normalize
        ffn_hidden = self.ffn_linear1.forward(x_norm2)  # Expand
        ffn_act = self.ffn_activation.forward(ffn_hidden)  # ReLU
        ffn_out = self.ffn_linear2.forward(ffn_act)  # Contract
        output = x1 + ffn_out                   # Residual

        return output
```

**Data flow through one block:**
```
Input [batch, seq, 32]
    ↓
LayerNorm
    ↓
MultiHeadAttention (2 heads, each 16-dim)
    ↓
Add residual ←───────┐
    ↓                │
LayerNorm            │
    ↓                │
Linear (32 → 64)     │
    ↓                │
ReLU                 │
    ↓                │
Linear (64 → 32)     │
    ↓                │
Add residual ←───────┘
    ↓
Output [batch, seq, 32]
```

---

## Step 8: Stacking Blocks

We stack 2 transformer blocks. Each block refines the representations:

```python
# From core/transformer.py, lines 175-185

# In TransformerLM.__init__:
self.blocks = [
    TransformerBlock(embed_dim, num_heads, ffn_dim)
    for _ in range(num_layers)  # num_layers = 2
]

# In forward:
for block in self.blocks:
    h = block.forward(h, mask)
```

**What each layer learns:**
- **Layer 1**: Basic patterns (word types, simple relationships)
- **Layer 2**: More complex patterns (built on Layer 1's representations)

In larger models (GPT-3 has 96 layers!), early layers learn syntax, middle layers learn semantics, and late layers learn task-specific features.

---

## Step 9: Making Predictions

After all transformer blocks, we predict the next word:

```python
# From core/transformer.py, lines 190-220

class TransformerLM:
    def __init__(self, config):
        # ... embedding, blocks ...

        # Final layer norm
        self.ln_final = LayerNorm(config["embed_dim"])

        # Project to vocabulary size
        self.output_proj = Linear(config["embed_dim"], config["vocab_size"])

    def forward(self, x):
        # 1. Embed tokens
        h = self.embedding.forward(x)

        # 2. Add positions
        h = self.pos_encoding.forward(h)

        # 3. Pass through transformer blocks
        for block in self.blocks:
            h = block.forward(h, mask)

        # 4. Final normalization
        h = self.ln_final.forward(h)

        # 5. Project to vocabulary
        logits = self.output_proj.forward(h)

        return logits  # Shape: (batch, seq, vocab_size)
```

**The output logits:**
- Shape: `(batch, seq_len, vocab_size)` = `(1, 8, 13)`
- `logits[0, 5, :]` = scores for predicting the word after position 5
- Higher score = model thinks that word is more likely

---

## Step 10: Training the Model

### The Loss Function

We use cross-entropy loss: "how surprised are we by the correct answer?"

```python
# From train.py, lines 25-100

class CrossEntropyLoss:
    def forward(self, logits, targets):
        # 1. Convert logits to probabilities (softmax)
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # 2. Get probability assigned to correct tokens
        correct_probs = self.probs[batch_idx, seq_idx, targets]

        # 3. Loss = negative log probability
        loss = -np.mean(np.log(correct_probs + eps))

        return loss

    def backward(self):
        # Gradient has a beautiful simple form:
        # dL/d(logits) = predicted_probs - one_hot(targets)
        grad = self.probs.copy()
        grad[batch_idx, seq_idx, self.targets] -= 1
        grad /= (batch_size * seq_len)
        return grad
```

**Example:**
- Model predicts: `{"roses": 0.1, "are": 0.7, "blue": 0.2}`
- Correct answer: "are"
- Loss = `-log(0.7)` = `0.36`

If model was wrong:
- Model predicts: `{"roses": 0.7, "are": 0.1, "blue": 0.2}`
- Correct answer: "are"
- Loss = `-log(0.1)` = `2.30` (much higher!)

### Backpropagation

We compute gradients by working backwards through every operation:

```python
# From core/transformer.py, lines 135-170

def backward(self, grad_output):
    # Work backwards through the block

    # FFN backward
    grad = self.ffn_linear2.backward(grad_output)
    grad = self.ffn_activation.backward(grad)
    grad = self.ffn_linear1.backward(grad)
    grad = self.ln2.backward(grad)

    # Residual: gradient splits and adds
    grad_x1 = grad_output + grad

    # Attention backward
    grad = self.attn.backward(grad_x1)
    grad = self.ln1.backward(grad)

    # Residual
    grad_input = grad_x1 + grad

    return grad_input
```

### The Optimizer

SGD with momentum updates parameters:

```python
# From train.py, lines 140-180

class SGD:
    def step(self, params_and_grads):
        for i, (param, grad) in enumerate(params_and_grads):
            if self.momentum > 0:
                # Momentum: accumulate velocity
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                param += self.velocities[i]
            else:
                # Vanilla SGD
                param -= self.lr * grad
```

### Gradient Clipping

Prevents exploding gradients:

```python
# From train.py, lines 185-210

def clip_gradients(params_and_grads, max_norm=1.0):
    # Compute total gradient norm
    total_norm = np.sqrt(sum(np.sum(g**2) for p, g in params_and_grads if g is not None))

    # Scale down if too large
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for param, grad in params_and_grads:
            grad *= scale
```

### Training Loop

```python
# From train.py, lines 215-250

def train_epoch(model, data_loader, loss_fn, optimizer, config):
    for inputs, targets in data_loader:
        # Forward pass
        logits = model.forward(inputs)
        loss = loss_fn.forward(logits, targets)

        # Backward pass
        grad_logits = loss_fn.backward()
        model.backward(grad_logits)

        # Clip gradients
        params_and_grads = model.get_params_and_grads()
        clip_gradients(params_and_grads, config["max_grad_norm"])

        # Update parameters
        optimizer.step(params_and_grads)
```

---

## Step 11: Generating Text

Once trained, we generate text autoregressively:

```python
# From train.py, lines 260-310

def generate(model, start_tokens, max_len, idx_to_word, temperature=1.0):
    tokens = list(start_tokens)

    for _ in range(max_len):
        # 1. Get context (last max_seq_len tokens)
        context = tokens[-max_seq_len:]
        input_array = np.array([context])

        # 2. Forward pass
        logits = model.forward(input_array)

        # 3. Get logits for next token (last position)
        next_logits = logits[0, -1, :]

        # 4. Apply temperature
        next_logits = next_logits / temperature

        # 5. Convert to probabilities
        probs = softmax(next_logits)

        # 6. Sample from distribution
        next_token = np.random.choice(len(probs), p=probs)

        tokens.append(next_token)

    return [idx_to_word[t] for t in tokens]
```

**Temperature:**
- `temperature < 1`: Sharper distribution → more deterministic
- `temperature > 1`: Flatter distribution → more random
- `temperature = 1`: Use raw probabilities

**Example generation:**
```
Start: "roses"
Step 1: predict next → "are" (p=0.9)
Step 2: predict next → "red" (p=0.85)
Step 3: predict next → "violets" (p=0.7)
...
Output: "roses are red violets are blue sugar is sweet..."
```

---

## Putting It All Together

Here's the complete flow in `main.py`:

```python
# From main.py

def main():
    # 1. Prepare data
    word_to_idx, idx_to_word = create_vocabulary(POEM)
    inputs, targets = create_training_data(POEM, word_to_idx, seq_len=8)
    data_loader = DataLoader(inputs, targets)

    # 2. Create model
    model = TransformerLM(config)

    # 3. Training setup
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(learning_rate=0.01, momentum=0.9)

    # 4. Train
    for epoch in range(2000):
        loss = train_epoch(model, data_loader, loss_fn, optimizer, config)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # 5. Generate
    start = [word_to_idx["roses"]]
    generated = generate(model, start, max_len=10, idx_to_word)
    print(" ".join(generated))
```

**Results:**
```
Epoch    0: Loss = 2.63
Epoch  500: Loss = 0.04
Epoch 1999: Loss = 0.04

Generated: "roses are red violets are blue sugar is sweet and so"
Accuracy: 97.5%
```

---

## Summary

| Component | Purpose | File Location |
|-----------|---------|---------------|
| Embedding | Words → Vectors | `core/layers.py:188` |
| PositionalEncoding | Add position info | `core/layers.py:251` |
| SingleHeadAttention | Learn word relationships | `core/attention.py:40` |
| MultiHeadAttention | Multiple relationship types | `core/attention.py:230` |
| Linear | Matrix multiplication + bias | `core/layers.py:30` |
| LayerNorm | Normalize activations | `core/layers.py:85` |
| ReLU | Non-linear activation | `core/activations.py:25` |
| TransformerBlock | Attention + FFN + residuals | `core/transformer.py:40` |
| TransformerLM | Complete language model | `core/transformer.py:150` |
| CrossEntropyLoss | Training objective | `train.py:25` |
| SGD | Parameter optimization | `train.py:140` |

---

## Further Reading

Now that you understand the basics:
1. Read the original paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
2. Explore larger models: GPT, BERT, T5
3. Try modifying this code: Add more layers, change dimensions, try different tasks

The concepts here scale directly to models with billions of parameters!
