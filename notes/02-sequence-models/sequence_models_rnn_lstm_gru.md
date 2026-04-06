# Sequence Models: RNN, LSTM, And GRU

## Intro And Concepts

Sequence models are neural architectures designed for data where order matters.

Examples:

- text
- speech
- time series
- DNA and protein sequences
- event streams

The central issue is that a sequence is not just a bag of tokens or values. The
position and history matter.

A sequence model therefore aims to represent:

`p(x_t | x_1, ..., x_(t-1))`

or more generally:

`p(y_t | x_1, ..., x_T, y_1, ..., y_(t-1))`

depending on the task.

### Why Ordinary Feedforward Networks Are Not Enough

A standard feedforward network assumes fixed-size input and does not naturally
share structure across positions in time.

For sequences we need:

- variable-length handling
- parameter sharing across time
- a mechanism to carry information from earlier steps to later ones

This is what recurrence was designed to provide.

### Recurrent Neural Networks

A basic recurrent neural network, or RNN, updates a hidden state:

`h_t = phi(W_x x_t + W_h h_(t-1) + b)`

and may produce an output:

`y_t = g(W_y h_t + c)`

The hidden state `h_t` acts as a compressed summary of the past.

This gives the right intuition:

```text
new hidden state = current input + previous memory -> updated representation
```

### Why Recurrence Is Powerful

RNNs impose temporal structure through repeated application of the same
transition:

```text
x_1 -> h_1
x_2 + h_1 -> h_2
x_3 + h_2 -> h_3
...
```

This parameter sharing means:

- the same mechanism processes every time step
- sequence length can vary
- the model can, in principle, carry information arbitrarily far

### Why Plain RNNs Struggle

The hidden state must do two incompatible jobs:

- preserve useful old information
- rapidly incorporate new information

This tension causes difficulty with long-range dependencies.

Training also suffers because backpropagation through time multiplies many
Jacobians, leading to:

- vanishing gradients
- exploding gradients

This is the main reason LSTMs and GRUs were invented.

### LSTM Intuition

Long Short-Term Memory networks separate the idea of:

- memory storage
- memory reading
- memory updating

An LSTM maintains:

- a cell state `c_t`
- a hidden state `h_t`

and uses gates to control information flow.

The rough structure is:

- forget gate: what to erase
- input gate: what new information to write
- output gate: what to expose

This gives a path through time that is easier for gradients to preserve.

### GRU Intuition

The Gated Recurrent Unit is a simplified alternative to the LSTM.

It uses:

- update gate
- reset gate

instead of separate cell and hidden states.

GRUs often work well in practice because they preserve the main gating idea
while being simpler and cheaper than LSTMs.

### Why Sequence Models Matter Historically

RNNs, LSTMs, and GRUs were the dominant neural sequence models before
transformers.

They were central in:

- language modeling
- machine translation
- speech recognition
- sequence labeling

Attention was originally introduced not to replace sequence models entirely, but
to help them overcome their bottlenecks.

## Deep Dive

### RNN Mathematics

A simple RNN uses:

`h_t = tanh(W_x x_t + W_h h_(t-1) + b_h)`

`o_t = W_o h_t + b_o`

The same matrices are reused at every time step. This weight sharing is both the
strength and the limitation of recurrence.

Strength:

- elegant temporal structure
- parameter efficiency

Limitation:

- repeated multiplication through the same transition can destabilize gradients

### Hidden State As Compressed Sufficient Summary

An RNN attempts to map the entire history into a fixed-dimensional hidden state.

Conceptually:

`h_t approx summary(x_1, ..., x_t)`

This is powerful but restrictive. A fixed-size vector must retain all relevant
past information regardless of sequence length.

This becomes a bottleneck when:

- sequences are long
- multiple distant dependencies matter
- information needed later is subtle

### Backpropagation Through Time

To train an RNN, the recurrent computation is unrolled:

```text
h_1 -> h_2 -> h_3 -> ... -> h_T
```

Then ordinary backpropagation is applied through the unrolled graph.

The gradient with respect to an earlier hidden state includes products such as:

`prod_k d h_k / d h_(k-1)`

This is where the instability enters.

If the norms of these Jacobians are mostly:

- less than 1 -> gradients vanish
- greater than 1 -> gradients explode

This is a spectral and dynamical-systems issue, not just a coding issue.

### Why Vanishing Gradients Hurt Long-Term Memory

Suppose information from time step `t=3` matters at `t=100`.
If the gradient signal from later time steps back to early time steps decays
exponentially, then the model receives almost no learning signal telling it to
preserve that early information.

This is why plain RNNs often learn short-term dependencies but struggle with
long-term ones.

### LSTM Equations

An LSTM introduces gated updates:

`f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)`   forget gate

`i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)`   input gate

`g_t = tanh(W_g [h_(t-1), x_t] + b_g)`      candidate content

`o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)`   output gate

`c_t = f_t * c_(t-1) + i_t * g_t`

`h_t = o_t * tanh(c_t)`

The key innovation is the cell state recurrence:

`c_t = f_t * c_(t-1) + ...`

This gives a more direct additive path through time, making gradient preservation
easier than in purely multiplicative recurrent dynamics.

### Why The LSTM Gates Help

#### Forget Gate

Controls whether old memory is preserved or erased.

#### Input Gate

Controls whether new information is written.

#### Output Gate

Controls what part of memory becomes visible as the hidden state.

This division of labor solves the earlier tension in plain RNNs:

- memory maintenance
- memory update
- memory exposure

are no longer forced into a single ungated transition.

### GRU Equations

A GRU typically uses:

`z_t = sigmoid(W_z [h_(t-1), x_t] + b_z)`   update gate

`r_t = sigmoid(W_r [h_(t-1), x_t] + b_r)`   reset gate

`h_t_tilde = tanh(W_h [r_t * h_(t-1), x_t] + b_h)`

`h_t = (1 - z_t) * h_(t-1) + z_t * h_t_tilde`

The update gate determines how much of the old state is kept versus replaced.
The reset gate controls how much past information is used when forming the new
candidate state.

This simpler design often performs competitively with LSTMs.

### LSTM Versus GRU

LSTM:

- more expressive gating structure
- separate cell state and hidden state
- sometimes stronger for complex long dependencies

GRU:

- fewer parameters
- simpler update equations
- often faster to train

In practice the best choice is empirical and task-dependent.

### Sequence Model Use Cases

#### Many-To-One

Example:

- sentiment classification from an entire sentence

```text
sequence -> final hidden state -> class prediction
```

#### One-To-Many

Example:

- conditioned generation from one input representation

#### Many-To-Many

Examples:

- translation
- tagging
- speech-to-text

### Why Attention Was Introduced

Encoder-decoder RNNs originally tried to compress an entire source sequence into
a single context vector. This created a bottleneck.

Attention was introduced so the decoder could directly access different encoder
states rather than relying only on one fixed summary.

So attention should be understood historically as:

- first, an augmentation to recurrent sequence models
- later, a mechanism strong enough to replace recurrence itself in transformers

### Computational Limitations Of Recurrence

RNN-style models process time steps sequentially:

`h_t` depends on `h_(t-1)`

This implies:

- low parallelism across sequence positions
- long dependency paths
- slower training at scale

These are major reasons attention-based models eventually overtook them.

### Final Synthesis

RNNs introduced the core neural idea for sequences: carry state through time.
LSTMs and GRUs refined that idea with gating so memory could be preserved and
updated more effectively.

They matter because they solve the original sequence problem and make clear why
attention and transformers were later such a major step forward.

## Further Reading

- `Canonical papers`: Elman, "Finding Structure in Time" (1990); Hochreiter and Schmidhuber, "Long Short-Term Memory" (1997); Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014).
- `Best intuition resource`: Chris Olah, *Understanding LSTM Networks*, and Andrej Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks*.
- `Best practical code resource`: `karpathy/char-rnn` on GitHub.
