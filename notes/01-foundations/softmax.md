# Softmax

## Intro And Concepts

Softmax is the standard mapping used to convert a vector of raw class scores,
called logits, into a probability distribution over mutually exclusive classes.

If a model outputs:

`z = [z_1, z_2, ..., z_K]`

then softmax produces:

`softmax(z)_i = exp(z_i) / sum_j exp(z_j)`

for `i = 1, ..., K`.

This definition does three things at once:

- forces every output to be non-negative
- makes all outputs sum to 1
- preserves relative score comparisons in a smooth way

So softmax is not merely an activation function in the usual elementwise sense.
It is a vector-valued normalization operator acting across the entire output
layer.

### The Basic Problem Softmax Solves

Suppose a classifier must choose exactly one class from `K` possibilities.

Examples:

- image classification: cat, dog, horse, car
- sentiment classification: positive, neutral, negative
- language modeling: next token among the whole vocabulary
- speech recognition: next symbol among many possible units

The model naturally produces raw evidence for each class. These raw values are
the logits:

```text
logits = [2.1, 0.3, -1.7, 4.2]
```

But logits are not probabilities:

- they can be negative
- they do not sum to 1
- their scale is arbitrary

Softmax turns them into something probabilistic:

```text
probabilities = [0.10, 0.02, 0.00, 0.88]
```

### What A Logit Really Means

A logit is a raw score before normalization.

In a neural network classifier, the final linear layer typically computes:

`z = W h + b`

where:

- `h` is the hidden representation
- `W` is the output weight matrix
- `b` is the bias vector
- `z` is the logit vector

Each `z_i` can be interpreted as unnormalized evidence for class `i`.

Softmax then converts relative evidence into relative probability mass.

### Why Softmax Is Used For Multi-Class Classification

Softmax is appropriate when the classes are:

- mutually exclusive
- collectively competing for probability mass

This is different from multi-label classification, where multiple labels may be
true at once. In that case, independent sigmoids are usually used instead.

So:

- softmax -> choose one class among many
- sigmoid -> independently score each label

### Intuition: Competition Across Outputs

Softmax works across all outputs together.

That is its key conceptual difference from activations such as:

- ReLU
- sigmoid
- tanh

Those are applied elementwise. Softmax is coupled across coordinates.

If one logit increases while the others stay fixed:

- its probability increases
- the others must decrease because the outputs must still sum to 1

So softmax creates competition between classes.

Visualization:

```text
larger score for one class
        ->
larger share of total probability mass
        ->
smaller shares for competing classes
```

### The Exponential Matters

Why use exponentials?

Because exponentials:

- make all values positive
- amplify score differences smoothly
- preserve ranking

If one logit is much larger than another, its exponential is disproportionately
larger, so softmax strongly prefers it.

Example:

```text
logits      = [1, 2, 5]
exp(logits) = [e^1, e^2, e^5]
```

The largest class dominates after exponentiation, but the final vector is still
normalized into a valid distribution.

### Translation Invariance

Softmax has an important property:

`softmax(z) = softmax(z + c)`

for any constant `c` added to every coordinate.

Why:

`exp(z_i + c) / sum_j exp(z_j + c)`

equals:

`exp(c) exp(z_i) / exp(c) sum_j exp(z_j)`

and `exp(c)` cancels out.

This matters deeply for numerical stability and interpretation:

- only relative differences between logits matter
- absolute offset does not matter

### Argmax Is Preserved

Softmax does not change which class is largest:

`argmax_i z_i = argmax_i softmax(z)_i`

So for prediction, the winning class is the same whether you use:

- raw logits
- softmax probabilities

Softmax is therefore mainly needed for:

- probabilistic interpretation
- probabilistic losses such as cross-entropy
- calibration-sensitive use cases

### Why The Last Layer Is Often Linear

In modern deep learning code, the final layer for multi-class classification is
often:

`Dense(K, activation='linear')`

and the loss is configured with:

`SparseCategoricalCrossentropy(from_logits=True)`

This is the standard training setup because it is numerically more stable.

Conceptually:

1. the network outputs logits
2. the loss internally applies a stable softmax-plus-log operation
3. optimization is done directly from logits

This avoids unstable explicit computation of:

- `softmax(z)`
- then `log(softmax(z))`

when logits are very large or very negative.

### Why Numerical Stability Matters

If logits are large, exponentials can overflow:

```text
exp(1000)  -> enormous
exp(-1000) -> effectively zero
```

A naive softmax implementation can therefore become unstable.

A standard stable version uses:

`softmax(z)_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))`

because subtracting the maximum logit does not change the output distribution
but keeps exponentials in a safe numeric range.

This is why frameworks prefer logits directly: they can fuse the operations in a
stable way.

### Typical Keras Pattern

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation="relu"),
    tf.keras.layers.Dense(15, activation="relu"),
    tf.keras.layers.Dense(4, activation="linear"),
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
```

Then for probabilities during inference:

```python
logits = model(x)
probs = tf.nn.softmax(logits)
```

### One Important Rule

Do not combine:

- `activation='softmax'`
- and `from_logits=True`

That is inconsistent because `from_logits=True` tells the loss to expect raw
unnormalized logits, not already-softmaxed probabilities.

Valid pairings are:

```text
linear last layer + from_logits=True
```

or

```text
softmax last layer + from_logits=False
```

The first is usually preferred for training stability.

### Where Softmax Appears

Softmax is used across many neural architectures.

#### Standard Feedforward Networks

For ordinary tabular or feature-based multi-class classification:

```text
input -> hidden layers -> logits -> softmax -> class probabilities
```

#### Convolutional Neural Networks

For image classification:

```text
image -> conv layers -> pooled representation -> logits -> softmax
```

The convolutional backbone extracts visual features; softmax handles the final
mutually exclusive class decision.

#### Recurrent Networks

For sequence tasks, softmax may appear:

- at the final step for whole-sequence classification
- at every time step for sequence prediction

Example:

```text
h_t -> logits_t -> softmax -> distribution over next token
```

This is standard in:

- language modeling
- machine translation decoders
- speech-to-text decoders

## Deep Dive

### Softmax As A Map To The Probability Simplex

The output of softmax lies on the probability simplex:

`Delta^(K-1) = { p in R^K : p_i >= 0, sum_i p_i = 1 }`

So softmax is a differentiable map:

`R^K -> Delta^(K-1)`

This geometric viewpoint is useful:

- logits live in unconstrained Euclidean space
- probabilities live on a constrained simplex

The role of softmax is to move from the unconstrained score space to the
probability geometry required by probabilistic classification.

Visualization:

```text
logit space: unconstrained scores
        ->
softmax
        ->
simplex: valid distributions over classes
```

### The Partition Function

Softmax can be written as:

`p_i = exp(z_i) / Z`

where:

`Z = sum_j exp(z_j)`

is the normalization constant, often called the partition function.

This object appears all over probabilistic modeling and statistical mechanics.
It ensures the outputs define a proper probability distribution.

### Binary Classification As A Special Case

For two classes, softmax reduces to logistic behavior.

Let logits be `[z_1, z_2]`. Then:

`p_1 = exp(z_1) / (exp(z_1) + exp(z_2))`

This can be rewritten in terms of the logit difference:

`p_1 = 1 / (1 + exp(-(z_1 - z_2)))`

which is a sigmoid on the score difference.

So binary logistic regression is a special two-class case of softmax-based
multi-class modeling.

### Cross-Entropy With Softmax

Suppose the true label is one-hot:

`y = [y_1, ..., y_K]`

and predicted probabilities are:

`p = softmax(z)`

Then the cross-entropy loss is:

`L(y, p) = - sum_i y_i log p_i`

If the true class is `c`, this reduces to:

`L = -log p_c`

This has a very clean interpretation:

- if the model assigns high probability to the true class, loss is small
- if the model assigns low probability to the true class, loss is large

### Softmax Plus Cross-Entropy Simplification

One of the most important derivations in deep learning is:

if:

- `p = softmax(z)`
- `L = - sum_i y_i log p_i`

then:

`dL/dz_i = p_i - y_i`

This result is elegant and fundamental.

It means the gradient at the logits is just:

```text
predicted probability - target probability
```

That is one reason the softmax-cross-entropy combination is so standard.

### Derivative Of Softmax Itself

Softmax is not elementwise, so its Jacobian is not diagonal.

If `p_i = softmax(z)_i`, then:

`dp_i/dz_j = p_i (delta_ij - p_j)`

So:

- when `i = j`: `dp_i/dz_i = p_i (1 - p_i)`
- when `i != j`: `dp_i/dz_j = -p_i p_j`

This captures the competition structure:

- increasing one logit affects every class probability
- outputs are coupled

### Log-Sum-Exp And Stability

A numerically stable softmax/cross-entropy implementation relies on the
log-sum-exp identity.

Since:

`log softmax(z)_i = z_i - log sum_j exp(z_j)`

the hard part is computing:

`log sum_j exp(z_j)`

stably.

This is done via:

`m = max_j z_j`

then:

`log sum_j exp(z_j) = m + log sum_j exp(z_j - m)`

Because all `z_j - m <= 0`, the exponentials stay bounded.

This is the core reason deep learning libraries want logits rather than already
softmaxed probabilities at training time.

### Why Linear Last Layer Is The Technical Default

When using:

- linear final layer
- `from_logits=True`

the framework can compute:

`cross_entropy(logits, labels)`

as one fused stable operation.

If you instead manually apply softmax before the loss, you risk:

- overflow in exponentials
- underflow to zeros
- unstable logs
- less precise gradients

So the usual training design is not merely stylistic. It is a numerical and
optimization choice.

### Softmax In Language Models

Softmax plays a central role in language modeling.

At each time step, the model produces one logit per vocabulary item:

`z_t in R^V`

where `V` is the vocabulary size.

Then:

`p(x_t | context) = softmax(z_t)`

This turns the model into a conditional distribution over the next token.

Visualization:

```text
context representation
        ->
logits over vocabulary
        ->
softmax
        ->
distribution over next token
```

This is the standard decoding distribution used by:

- RNN language models
- LSTM decoders
- GRU decoders
- transformer decoders
- GPT-style models

### Softmax In Sequence Models

In recurrent and transformer sequence models, softmax often appears in two
different places.

#### 1. Output Softmax

Maps logits to a token distribution or class distribution.

#### 2. Attention Softmax

Inside attention, softmax is used over pairwise compatibility scores:

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`

This is conceptually different from output softmax:

- output softmax normalizes over candidate classes or tokens
- attention softmax normalizes over positions to form weighted averaging

So softmax appears not only at the output of modern models, but also deep inside
their internal routing mechanism.

### Calibration And Confidence

Softmax outputs are often interpreted as confidence scores, but this should be
done carefully.

High softmax probability means:

- one logit is much larger than the others

It does not automatically mean:

- the model is well calibrated
- the probability matches true empirical correctness frequency

Modern deep nets can be overconfident, so softmax probabilities are not always
well calibrated without additional methods such as:

- temperature scaling
- label smoothing
- calibration evaluation

### Temperature Softmax

A common generalization is:

`softmax(z / T)`

where `T > 0` is temperature.

Effects:

- low `T` -> sharper distribution
- high `T` -> flatter distribution

Special uses:

- calibration
- distillation
- controlling generation randomness in language models

Visualization:

```text
low temperature  -> peaked distribution
high temperature -> smoother distribution
```

### Label Smoothing

Instead of training with a perfectly one-hot target, label smoothing uses a
slightly softened target distribution.

Why:

- reduces overconfidence
- improves generalization in some settings
- discourages extreme logit separation

This modifies how softmax-based training behaves, especially in large
classification systems.

### Large-Vocabulary Problems

Softmax becomes computationally expensive when the number of classes is huge.

Examples:

- language models with very large vocabularies
- recommendation systems with enormous item spaces

The cost comes from:

- producing all logits
- exponentiating all logits
- normalizing across all classes

This motivates approximations such as:

- sampled softmax
- hierarchical softmax
- noise-contrastive methods

### Hierarchical Softmax

Hierarchical softmax replaces flat normalization over all classes with a tree.

Instead of computing probability over all `K` classes directly, the model makes
a sequence of binary or small branching decisions.

This reduces computational cost from dependence on all classes to dependence on
tree depth.

### Softmax Versus Independent Sigmoids

This distinction is critical.

#### Softmax

Use when:

- exactly one class is correct
- classes compete
- total probability mass must be shared

Examples:

- digit classification
- object category classification
- next-token prediction

#### Independent Sigmoids

Use when:

- multiple labels may be true at once
- labels do not compete for a shared total

Examples:

- image has dog and bicycle
- document has multiple tags
- patient has multiple diagnoses

### Why Softmax Is So Central

Softmax sits at the meeting point of:

- linear score models
- probabilistic modeling
- maximum likelihood training
- cross-entropy optimization

It is one of the most important bridges between raw neural outputs and
probabilistic decision-making.

### Final Synthesis

Softmax is best understood not as "just another activation," but as a
normalization map that converts unconstrained logits into a structured
probability distribution over competing classes.

Its role is central because it:

- defines a distribution on the simplex
- couples outputs into competition
- connects naturally to cross-entropy and maximum likelihood
- preserves ranking while enabling probability interpretation
- admits stable fused implementations from logits

That is why the standard modern design is:

- linear output layer for logits
- cross-entropy loss with `from_logits=True` during training
- explicit softmax only when probabilities are needed for interpretation,
  calibration, or inference.
