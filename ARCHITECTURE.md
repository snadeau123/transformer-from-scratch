# Transformer Architecture

Visual diagrams of the transformer implementation.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Input
        tokens[/"Token IDs<br/>[batch, seq_len]"/]
    end

    subgraph Embedding Layer
        embed[Embedding Lookup]
        pos[Positional Encoding]
        embed_out(("+"))
    end

    subgraph Transformer Blocks
        block1[TransformerBlock 1]
        block2[TransformerBlock 2]
    end

    subgraph Output Layer
        ln_final[Layer Norm]
        proj[Linear Projection]
        logits[/"Logits<br/>[batch, seq_len, vocab_size]"/]
    end

    tokens --> embed
    embed --> embed_out
    pos --> embed_out
    embed_out --> block1
    block1 --> block2
    block2 --> ln_final
    ln_final --> proj
    proj --> logits
```

## Transformer Block (Pre-LayerNorm)

```mermaid
flowchart TB
    input[/"Input<br/>[batch, seq, embed_dim]"/]

    subgraph "Sub-layer 1: Attention"
        ln1[LayerNorm]
        mha[Multi-Head<br/>Attention]
        add1(("+"))
    end

    subgraph "Sub-layer 2: Feed-Forward"
        ln2[LayerNorm]
        ffn1[Linear<br/>embed→ffn_dim]
        relu[ReLU]
        ffn2[Linear<br/>ffn_dim→embed]
        add2(("+"))
    end

    output[/"Output<br/>[batch, seq, embed_dim]"/]

    input --> ln1
    input --> add1
    ln1 --> mha
    mha --> add1
    add1 --> ln2
    add1 --> add2
    ln2 --> ffn1
    ffn1 --> relu
    relu --> ffn2
    ffn2 --> add2
    add2 --> output
```

## Multi-Head Attention

```mermaid
flowchart TB
    input[/"Input X<br/>[batch, seq, embed_dim]"/]

    subgraph "Head 1"
        q1[W_q]
        k1[W_k]
        v1[W_v]
        attn1[Scaled Dot-Product<br/>Attention]
    end

    subgraph "Head 2"
        q2[W_q]
        k2[W_k]
        v2[W_v]
        attn2[Scaled Dot-Product<br/>Attention]
    end

    concat[Concatenate]
    wo[W_o Linear]
    output[/"Output<br/>[batch, seq, embed_dim]"/]

    input --> q1 & k1 & v1
    input --> q2 & k2 & v2
    q1 & k1 & v1 --> attn1
    q2 & k2 & v2 --> attn2
    attn1 --> concat
    attn2 --> concat
    concat --> wo
    wo --> output
```

## Scaled Dot-Product Attention

```mermaid
flowchart TB
    q[/"Query Q<br/>[batch, seq, head_dim]"/]
    k[/"Key K<br/>[batch, seq, head_dim]"/]
    v[/"Value V<br/>[batch, seq, head_dim]"/]

    matmul1["MatMul<br/>Q × K^T"]
    scale["Scale<br/>÷ √d_k"]
    mask["Causal Mask<br/>(optional)"]
    softmax[Softmax]
    matmul2["MatMul<br/>weights × V"]

    output[/"Output<br/>[batch, seq, head_dim]"/]

    q --> matmul1
    k --> matmul1
    matmul1 --> scale
    scale --> mask
    mask --> softmax
    softmax --> matmul2
    v --> matmul2
    matmul2 --> output
```

## Causal Mask

Prevents attending to future positions during autoregressive generation:

```mermaid
flowchart LR
    subgraph "Attention Scores + Mask"
        matrix["
        Position →  0    1    2    3
        Query 0:  [ 0,  -∞,  -∞,  -∞ ]
        Query 1:  [ 0,   0,  -∞,  -∞ ]
        Query 2:  [ 0,   0,   0,  -∞ ]
        Query 3:  [ 0,   0,   0,   0 ]
        "]
    end
```

## Training Data Flow

```mermaid
flowchart LR
    subgraph Data Preparation
        poem["Poem Text"]
        vocab["Vocabulary<br/>word ↔ idx"]
        seqs["Sequences<br/>[input, target]"]
    end

    subgraph Forward Pass
        model["TransformerLM"]
        logits["Logits"]
        loss["CrossEntropy<br/>Loss"]
    end

    subgraph Backward Pass
        grad_loss["∂L/∂logits"]
        grad_model["Backprop through<br/>all layers"]
        params["Parameter<br/>Gradients"]
    end

    subgraph Update
        clip["Gradient<br/>Clipping"]
        sgd["SGD + Momentum"]
        update["Update<br/>Parameters"]
    end

    poem --> vocab --> seqs
    seqs --> model --> logits --> loss
    loss --> grad_loss --> grad_model --> params
    params --> clip --> sgd --> update
    update -.-> model
```

## Layer Components

```mermaid
flowchart TB
    subgraph "Linear Layer"
        lin_in[/"x"/]
        lin_w["W, b"]
        lin_op["y = x @ W + b"]
        lin_out[/"y"/]
        lin_in --> lin_op
        lin_w --> lin_op
        lin_op --> lin_out
    end

    subgraph "Layer Norm"
        ln_in[/"x"/]
        ln_params["γ, β"]
        ln_op["y = γ · (x - μ) / σ + β"]
        ln_out[/"y"/]
        ln_in --> ln_op
        ln_params --> ln_op
        ln_op --> ln_out
    end

    subgraph "Embedding"
        emb_in[/"token_id"/]
        emb_table["Embedding<br/>Matrix"]
        emb_op["lookup"]
        emb_out[/"vector"/]
        emb_in --> emb_op
        emb_table --> emb_op
        emb_op --> emb_out
    end
```

## File Structure

```mermaid
flowchart TB
    subgraph "core/"
        activations["activations.py<br/>ReLU, Softmax"]
        layers["layers.py<br/>Linear, LayerNorm,<br/>Embedding, PosEnc"]
        attention["attention.py<br/>SingleHead, MultiHead"]
        transformer["transformer.py<br/>Block, TransformerLM"]
    end

    subgraph "utils/"
        data["data.py<br/>Vocabulary, DataLoader"]
    end

    subgraph "root"
        config["config.py"]
        train["train.py<br/>Loss, SGD, Training"]
        main["main.py"]
    end

    activations --> layers
    layers --> attention
    attention --> transformer
    data --> main
    config --> main
    train --> main
    transformer --> main
```

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Embedding | vocab_size × embed_dim = 13 × 32 = 416 |
| Per Attention Head | 3 × (embed_dim × head_dim + head_dim) = 3 × (32 × 16 + 16) = 1,584 |
| Multi-Head (2 heads) | 2 × 1,584 + (embed_dim × embed_dim + embed_dim) = 4,224 |
| FFN per Block | 2 × (embed_dim × ffn_dim + ffn_dim) = 2 × (32 × 64 + 64) = 4,224 |
| LayerNorm per Block | 2 × 2 × embed_dim = 128 |
| Per Block Total | 4,224 + 4,224 + 128 = 8,576 |
| 2 Blocks | 17,152 |
| Final LayerNorm | 64 |
| Output Projection | embed_dim × vocab_size + vocab_size = 32 × 13 + 13 = 429 |
| **Total** | **~18,000** |
