# Attention And Self-Attention

## Intro And Concepts

Attention is a mechanism for dynamically selecting which pieces of information
should matter most for a computation at a given step.

The high-level pattern is:

```text
score relevance
    ->
normalize scores
    ->
take a weighted combination of values
```

This idea became central because sequence models faced a bottleneck: trying to
compress all relevant past information into a single fixed hidden state or
context vector.

### Why Attention Was Needed

In encoder-decoder RNNs, the decoder originally relied on one vector summarizing
the entire source sequence.

That creates problems when:

- source sequences are long
- multiple distant positions matter differently for each decoding step
- alignment between source and target is dynamic

Attention solves this by letting the decoder look back at all encoder states and
decide, at each step, which ones matter most.

### Basic Attention Pattern

Suppose we have:

- a query `q`
- keys `k_i`
- values `v_i`

Then attention computes:

1. scores: `s_i = score(q, k_i)`
2. weights: `a_i = softmax(s)_i`
3. output: `context = sum_i a_i v_i`

Interpretation:

- the query asks what is needed now
- keys describe what each memory slot offers
- values contain the actual information to retrieve

### Why Softmax Appears Here

Softmax turns arbitrary similarity scores into a probability-like distribution
over positions.

This gives:

- non-negative weights
- normalized weights summing to 1
- differentiable competition among candidate positions

So in attention, softmax is not producing class probabilities. It is producing
attention weights over positions or memory slots.

### Self-Attention

Self-attention is the special case where:

- queries, keys, and values come from the same sequence

So each token can attend to other tokens in the sequence, including possibly
itself.

This allows the model to build contextual representations by directly mixing
information across positions rather than relying on recurrent state propagation.

### Why Self-Attention Was A Big Deal

Self-attention changes the way dependencies are modeled.

RNN:

- information flows step by step through hidden state

Self-attention:

- any token can directly interact with any other token in one layer

This reduces path length between distant positions and greatly improves
parallelism.

## Deep Dive

### Classical Encoder-Decoder Attention

Suppose encoder hidden states are:

`h_1, ..., h_T`

At decoder step `t`, we have decoder state `s_t`.

We score each encoder position:

`e_(t,i) = score(s_t, h_i)`

Then normalize:

`alpha_(t,i) = exp(e_(t,i)) / sum_j exp(e_(t,j))`

Then form a context vector:

`c_t = sum_i alpha_(t,i) h_i`

The decoder now uses `c_t` in addition to its own hidden state.

This solves the fixed-context bottleneck because different decoding steps can
focus on different source positions.

### Alignment Interpretation

Attention can be interpreted as soft alignment.

In translation, for example:

- while generating one target word
- the decoder assigns higher attention weight to source words most relevant to
  that target word

This is more flexible than forcing all source information into one vector.

### Query-Key-Value Formalism

Modern attention is usually written with queries, keys, and values.

Given matrices:

- `Q` for queries
- `K` for keys
- `V` for values

attention is:

`Attention(Q, K, V) = softmax(score(Q, K)) V`

The most common score is scaled dot product:

`score(Q, K) = QK^T / sqrt(d_k)`

so:

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`

This is the canonical transformer attention equation.

### Why Divide By `sqrt(d_k)`

If query and key dimension `d_k` is large, raw dot products can grow in
magnitude. Large logits fed into softmax create overly peaked attention weights
and unstable gradients.

Dividing by:

`sqrt(d_k)`

controls variance and keeps the score scale in a more stable range.

This is a subtle but important normalization step.

### Self-Attention As Contextual Mixing

Let token representations be:

`X = [x_1, ..., x_n]`

Then:

`Q = XW_Q`

`K = XW_K`

`V = XW_V`

Each token produces:

- a query: what it is looking for
- a key: how it can be matched
- a value: what content it contributes

Then every token receives a weighted mixture of other tokens' values according
to query-key compatibility.

This means self-attention is a learned, content-dependent message passing
mechanism.

### Geometry Of Attention

The matrix:

`QK^T`

contains pairwise similarities between query positions and key positions.

After softmax, each row becomes a distribution over where one position attends.

So attention can be viewed as:

```text
pairwise similarity matrix
        ->
row-wise normalized routing weights
        ->
weighted recombination of values
```

This is a very different computational style from recurrence.

### Why Attention Helps Long-Range Dependencies

In an RNN, information from token 1 to token 100 must pass through many
intermediate states.

In self-attention, token 100 can attend directly to token 1 in one layer.

This shortens dependency path length dramatically.

That helps with:

- long-range context
- gradient flow
- easier learning of distant relationships

### Multi-Head Attention

Instead of using a single attention map, transformers use multiple heads.

Each head has separate projection matrices:

- `W_Q^(h)`
- `W_K^(h)`
- `W_V^(h)`

This allows the model to learn different relation types in parallel.

One head may focus on:

- local syntax
- positional relations
- subject-verb agreement
- long-range coreference
- semantic dependency

The outputs of all heads are concatenated and projected.

### Why Multiple Heads Help

If there were only one head, all relational structure would be forced through a
single attention pattern.

Multiple heads increase expressivity by allowing the model to attend in multiple
subspaces simultaneously.

### Causal Self-Attention

For autoregressive language models, a token should not attend to future tokens.

So a causal mask is applied before softmax:

- future positions get `-inf` score
- softmax turns their attention weight into zero

This gives:

`p(x_t | x_<t)`

while still using self-attention.

This is the core mechanism behind GPT-style decoders.

### Attention Versus Convolution Versus Recurrence

#### Recurrence

- strong temporal inductive bias
- sequential computation
- long dependency path

#### Convolution

- local receptive fields
- strong locality bias
- easier parallelism than recurrence

#### Attention

- content-dependent global interaction
- direct long-range access
- highly parallelizable

This is why attention became so dominant in large-scale sequence modeling.

### Attention Is Not Memory In The Same Way As RNN State

RNN memory is compressed into hidden state.
Attention-based memory is often externalized across token representations.

That means:

- RNNs carry a compressed evolving summary
- attention models retain many token-level states and learn to retrieve among
  them dynamically

This is a major conceptual shift.

### Computational Trade-Off

Self-attention has a cost:

- pairwise interactions across positions
- `O(n^2)` memory and compute in sequence length for standard full attention

So while it solves long-range dependency and parallelism issues, it introduces
scaling issues for very long contexts.

This is why efficient attention variants continue to matter.

### Why Attention Was The Bridge To Transformers

Historically:

1. RNNs handled sequences through recurrence
2. attention improved encoder-decoder RNNs
3. self-attention became strong enough to replace recurrence entirely

That last step is the conceptual leap into transformers.

So attention is not just another layer. It is the mechanism that made sequence
modeling less dependent on recurrent state propagation.

### Final Synthesis

Attention is a differentiable relevance-weighted retrieval mechanism.
Self-attention applies that mechanism within a sequence itself.

It matters because it replaces the idea of "store everything in one evolving
hidden state" with "store representations across positions and dynamically route
information between them."

That is why attention is the immediate conceptual precursor to transformers.

## Further Reading

- `Canonical paper`: Bahdanau, Cho, and Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014).
- `Best intuition resource`: Lilian Weng, *Attention? Attention!*, and Jay Alammar, *The Illustrated Transformer*.
- `Best practical code resource`: Harvard NLP's *The Annotated Transformer*.
