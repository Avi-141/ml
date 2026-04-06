# Neural Networks: Mathematical And Statistical Foundations

## Intro And Concepts

A neural network is best understood as a parameterized function

`f_theta : X -> Y`

where `theta` contains the trainable parameters and the model is chosen from a
large function class. Training is the process of selecting parameter values that
make this function useful under a task-specific objective.

This sounds simple, but it ties together several mathematical ideas:

- linear algebra for how data and parameters are represented
- calculus for how parameters are updated
- probability and statistics for how uncertainty, noise, and generalization are
  modeled
- optimization for how good parameters are found

The reason neural networks work is not that they "memorize patterns" in a vague
sense. It is that they define a flexible family of compositional functions:

`f_theta(x) = f_L(f_(L-1)(...f_1(x)))`

where each layer transforms the representation from the previous one.

### Why Compositions Matter

A single linear map:

`f(x) = Wx + b`

can only produce affine transformations. No matter how many affine layers you
stack, if there is no non-linearity between them, the whole network still
collapses to one affine transformation.

This is why activations matter:

```text
linear map + nonlinearity + linear map + nonlinearity + ...
```

The composition creates expressive function classes that can approximate complex
decision boundaries, smooth regressors, and structured mappings.

### Where The Statistics Enters

Machine learning is not just function fitting. It is statistical estimation from
finite data.

We assume there is an unknown data-generating distribution `P(X, Y)` and we only
observe a sample:

`{(x_i, y_i)}_(i=1)^n`

The training objective is typically empirical risk minimization:

`R_hat(theta) = (1/n) sum_i L(f_theta(x_i), y_i)`

This approximates the true risk:

`R(theta) = E_(x,y)~P [ L(f_theta(x), y) ]`

So the model is trained on observed examples, but the real goal is low expected
error on unseen draws from the same or related distribution.

### Errors, Losses, And Risk

The word "error" is often used informally, but in mathematical ML it helps to
separate:

- residual: prediction minus target
- loss: penalty assigned to a prediction
- risk: expected loss under a distribution

Examples:

- regression residual: `y_hat - y`
- squared loss: `(y_hat - y)^2`
- cross-entropy: `-log p_theta(y|x)`

The loss is what training minimizes. The error metric reported later may be
different.

### Why Neural Networks Use Matrices Everywhere

Neural networks process many features and many neurons at once.

A dense layer for one example is:

`z = Wx + b`

For a batch of examples:

`Z = XW^T + b`

Matrices are the natural language for this because they represent:

- many linear combinations at once
- transformations between vector spaces
- efficient parallel computation

This is not just notation. It matches the geometry of the problem:

- an input example is a vector
- a layer is a linear transformation
- the learned representation moves through a sequence of vector spaces

### The Intuition Behind Representation Learning

Each hidden layer tries to construct a new representation in which the task is
easier.

Very roughly:

```text
raw input -> low-level features -> mid-level abstractions -> task-relevant representation
```

The model is not only learning the final predictor. It is learning a coordinate
system in which prediction becomes simpler.

### Why Overparameterized Models Can Still Generalize

Modern neural networks often have more parameters than training examples, yet
they still generalize surprisingly well. This is not fully explained by classical
small-model theory alone, but several intuitions matter:

- gradient-based optimization has implicit bias
- architecture imposes structure
- data has low-dimensional regularity
- regularization and stochasticity affect which minima are found

So the useful question is not only "how many parameters are there?" but "what
kind of functions does training actually prefer?"

## Deep Dive

### Neural Networks As Structured Function Classes

A feedforward layer has the form:

`h_(l+1) = sigma(W_l h_l + b_l)`

where:

- `h_l` is the representation at layer `l`
- `W_l` is a matrix
- `b_l` is a bias
- `sigma` is a non-linearity

The network function is therefore:

`f_theta(x) = h_L`

with parameters:

`theta = {W_1, b_1, ..., W_(L-1), b_(L-1)}`

This structure matters because the network does not search over arbitrary
functions directly. It searches over functions factored into repeated linear and
nonlinear transforms.

### The Statistical Problem: Estimation Under Finite Data

Suppose the true target follows:

`Y = g(X) + epsilon`

for regression, where `epsilon` is noise.

The learner never sees `g` directly. It only sees noisy samples and must choose
`f_theta` that balances:

- approximation error: can the model class represent `g`?
- estimation error: can we estimate the right parameters from finite data?
- optimization error: can training actually find them?

Neural-network performance is therefore shaped by all three at once.

### Why Linear Algebra Is Central

Linear algebra appears because:

1. data is represented as vectors
2. layers are linear maps before activations
3. optimization uses gradients, Jacobians, and Hessians
4. curvature, conditioning, and stability depend on matrix structure

If:

`x in R^d`

and a layer has `m` neurons, then:

`W in R^(m x d)`

maps the input vector into a new `m`-dimensional representation.

Each row of `W` defines one neuron:

`z_j = w_j^T x + b_j`

So a dense layer is many learned dot products in parallel.

### Why Dot Products Matter

A dot product:

`w^T x`

measures alignment between the input and a learned feature direction. A neuron
activates strongly when the input aligns with the pattern encoded by its weight
vector.

This gives a geometric interpretation:

- weights define directions or hyperplanes
- activations measure how the input sits relative to those directions
- deeper layers compose these measurements into more abstract features

### Activations And Piecewise Geometry

For ReLU:

`sigma(z) = max(0, z)`

the network becomes piecewise linear: linear within regions, but with region
switching determined by which units are active.

This gives modern networks an important kind of expressivity:

- linear algebra inside each region
- nonlinear global behavior across regions

### Loss As A Statistical Decision Criterion

The loss function defines what kind of mistakes matter.

Examples:

#### Squared Error

`L = (y_hat - y)^2`

This corresponds to Gaussian-noise assumptions and heavily penalizes large
errors.

#### Absolute Error

`L = |y_hat - y|`

This is more robust to outliers but less smooth for optimization.

#### Cross-Entropy

`L = -log p_theta(y|x)`

This corresponds to maximum likelihood for categorical models and is the default
for classification.

So loss choice is not arbitrary. It often reflects a probabilistic model of the
data.

### Why Probability Shows Up In Classification

In classification we usually want not only a class label, but a model of class
likelihoods or posterior scores. If:

`p_theta(y|x)`

is the model's predicted probability, then minimizing negative log-likelihood is
equivalent to maximizing the likelihood of the observed labels.

This gives the standard classification training rule:

`min_theta -sum_i log p_theta(y_i | x_i)`

That is why softmax plus cross-entropy appears so naturally.

### Optimization And Generalization Are Different

A network may achieve low training loss without generalizing well.

Training addresses:

- parameter fitting on observed data

Generalization concerns:

- performance on unseen data

This gap is where regularization, inductive bias, batch normalization, data
augmentation, and optimizer behavior become important.

### Why Mini-Batches Work

Instead of computing the exact gradient over the entire dataset, training often
uses:

`g_t = (1/|B_t|) sum_(i in B_t) grad_theta L_i(theta)`

for a mini-batch `B_t`.

This gives:

- cheaper updates
- noisy but useful gradient estimates
- implicit regularization through stochasticity

So the optimizer is not following the exact population objective. It is
navigating a noisy empirical approximation.

### The Hidden Role Of Geometry

Training quality depends on the geometry of the loss landscape:

- steep directions
- flat directions
- saddle regions
- ill-conditioned valleys

That is why matrices, eigenvalues, curvature, and adaptive optimization matter.
They tell us how sensitive the objective is to parameter movement in different
directions.

### Final Synthesis

Neural networks sit at the intersection of:

- linear algebra for representation and transformation
- statistics for estimation under uncertainty
- calculus for local sensitivity
- optimization for parameter search

They work not because any one piece is magical, but because all of these pieces
fit together:

- matrices express transformations
- activations create expressive compositions
- losses encode what counts as error
- gradients tell us how to change parameters
- statistical learning determines whether those changes generalize

## Further Reading

- `Canonical reference`: Goodfellow, Bengio, and Courville, *Deep Learning*.
- `Best intuition resource`: Michael Nielsen, *Neural Networks and Deep Learning*, plus the 3Blue1Brown neural network series.
- `Best practical code resource`: `karpathy/micrograd` on GitHub.
