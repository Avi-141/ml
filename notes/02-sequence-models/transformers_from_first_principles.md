# Transformers From First Principles

## Intro And Concepts

The transformer is a sequence architecture built around self-attention rather
than recurrence.

Its central claim is:

> Sequence modeling does not require processing tokens one step at a time if
> pairwise contextual interaction can be learned directly.

This is the architectural shift that distinguishes transformers from classical
RNN/LSTM/GRU models.

### The Core Problem

A sequence model must combine:

- token identity
- contextual information
- long-range dependencies
- efficient learning

RNNs solve this with hidden-state recurrence.
Transformers solve it with repeated layers of:

- self-attention
- feedforward transformation
- residual connections
- normalization

### Minimal Transformer Block

A transformer block contains:

1. multi-head self-attention
2. residual connection
3. normalization
4. position-wise feedforward network
5. another residual connection
6. another normalization

Very roughly:

```text
token embeddings + positional information
        ->
self-attention
        ->
feedforward mixing
        ->
repeat many times
```

### Why Positional Information Is Needed

Self-attention alone is permutation-equivariant over the input set of tokens.
Without positional information, the model would know which tokens are present but
not their order.

So transformers add positional information through:

- sinusoidal encodings
- learned position embeddings
- relative position mechanisms

This restores sequence order information.

### Why Transformers Worked So Well

Transformers solved major issues in sequence modeling:

- shorter path between distant tokens
- better parallelism than recurrence
- scalable self-attention as contextual mixing
- strong compatibility with large-scale pretraining

This combination made them dominant in:

- NLP
- vision
- multimodal learning
- code models
- speech models

## Deep Dive

### Input Representation

Suppose the input tokens are:

`x_1, ..., x_n`

Each token is embedded:

`e_i = Embedding(x_i)`

Then positional information is added:

`h_i^(0) = e_i + p_i`

The initial sequence matrix is:

`H^(0) in R^(n x d_model)`

where each row corresponds to one contextualizable token representation.

### Self-Attention Inside A Transformer

Given hidden states `H`, compute:

`Q = H W_Q`

`K = H W_K`

`V = H W_V`

Then:

`Attention(H) = softmax(QK^T / sqrt(d_k)) V`

This means each token representation becomes a weighted mixture of value vectors
from across the sequence, where weights are determined by learned query-key
compatibility.

### Multi-Head Attention

Instead of one attention map, use several heads:

`head_h = Attention(H W_Q^(h), H W_K^(h), H W_V^(h))`

Then concatenate:

`MultiHead(H) = Concat(head_1, ..., head_m) W_O`

This allows the model to learn multiple relational subspaces in parallel.

### Feedforward Network

After attention, each token position passes through the same feedforward
subnetwork independently:

`FFN(x) = W_2 sigma(W_1 x + b_1) + b_2`

This is position-wise:

- attention mixes information across positions
- FFN transforms each position's representation nonlinearly

So the transformer alternates:

- cross-position interaction
- within-position transformation

### Residual Connections And Normalization

Residual connections help preserve gradient flow and stabilize deep stacking:

`x -> x + Sublayer(x)`

Normalization helps stabilize scale and optimization.

Together they make very deep sequence models trainable.

This is one reason transformers scale so well.

### Encoder, Decoder, And Encoder-Decoder Variants

#### Encoder-Only

Used for understanding tasks:

- classification
- retrieval
- token labeling

Examples:

- BERT
- RoBERTa

#### Decoder-Only

Uses causal self-attention and predicts the next token.

Examples:

- GPT-style models

#### Encoder-Decoder

Uses:

- encoder self-attention
- decoder masked self-attention
- cross-attention from decoder to encoder

Examples:

- T5
- BART

### Why Decoder-Only Models Suit Language Modeling

Autoregressive language modeling wants:

`p(x_1, ..., x_T) = prod_t p(x_t | x_<t)`

Masked causal self-attention implements exactly this by preventing access to
future tokens while still allowing rich conditioning on all past tokens.

This gives the modern GPT-style training setup.

### Why Transformers Beat RNNs At Scale

#### Shorter Dependency Paths

Two tokens can interact directly in one attention layer.

#### Better Parallelism

All positions can be processed simultaneously during training.

#### Flexible Contextual Mixing

Information routing is content-dependent, not locked to sequential recurrence.

#### Large-Scale Pretraining Compatibility

Transformers scale well with huge datasets, huge models, and distributed
training.

These four reasons are the main story.

### What The Transformer Actually Learns

A transformer layer can be thought of as learning:

- what each token should ask
- where it should look
- what information it should retrieve
- how to rewrite its representation after retrieval

Across many layers, this yields increasingly abstract contextual
representations.

### Why Transformers Still Need Feedforward Layers

Attention alone only mixes information.
It does not provide enough expressive nonlinear transformation by itself.

The feedforward block gives each position additional nonlinear processing
capacity after contextual information has been integrated.

So the architecture needs both:

- relational mixing
- nonlinear pointwise transformation

### Cross-Attention

In encoder-decoder models, the decoder attends not only to its own prior tokens
but also to encoder outputs:

`CrossAttention(Q_decoder, K_encoder, V_encoder)`

This lets the decoder dynamically retrieve source information while generating
the target.

This is the generalized attention mechanism used for translation and many
conditional generation tasks.

### Transformer Limitations

Transformers are powerful, but not free.

Standard self-attention scales quadratically in sequence length:

- memory `O(n^2)`
- compute `O(n^2)`

This becomes expensive for long contexts.

Transformers also often require:

- large data
- large compute
- careful optimization

So their dominance comes from scale and flexibility, not from being universally
cheap.

### Historical Arc

A useful sequence of ideas is:

```text
feedforward networks
    ->
recurrent sequence models
    ->
attention-augmented RNNs
    ->
self-attention
    ->
transformers
```

Understanding transformers is much easier once you see them as the architectural
endpoint of this progression.

### Why Transformers Became Foundation Models

Transformers are especially suitable for foundation-model training because they
combine:

- scalable optimization
- strong representation learning
- flexible input-output behavior
- architectural reuse across domains

This is why the same family appears in:

- BERT
- GPT
- T5
- ViT
- CLIP-style systems
- multimodal LLMs

### Final Synthesis

Transformers replace recurrence with self-attention-based contextual interaction.
They process sequences by repeatedly:

- comparing tokens to tokens
- routing information through attention
- transforming each position with feedforward layers

Their success comes from combining expressive long-range interaction with
parallelizable computation, which made them the dominant architecture of modern
deep learning.

## Further Reading

- `Canonical paper`: Vaswani et al., "Attention Is All You Need" (2017).
- `Best intuition resource`: Jay Alammar, *The Illustrated Transformer*.
- `Best practical code resource`: `karpathy/nanoGPT` on GitHub and Harvard NLP's *The Annotated Transformer*.
