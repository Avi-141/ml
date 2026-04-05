# Matrices, Eigenvalues, And Eigenvectors For Machine Learning

## Intro And Concepts

Matrices appear everywhere in machine learning because they are the natural
representation of linear transformations.

If a vector `x` represents a point, feature set, or hidden state, then a matrix
`A` transforms it:

`y = Ax`

This is why matrices are not just bookkeeping devices in ML. They encode how one
representation is turned into another.

### Why Matrices Work In Neural Networks

A dense layer computes:

`z = Wx + b`

This means:

- each neuron computes a weighted sum
- the whole layer computes many weighted sums in parallel
- the matrix `W` stores all those linear coefficients at once

For a batch of inputs, matrices become even more natural:

`Z = XW^T + b`

So matrices work here because the network repeatedly applies linear maps between
vector spaces.

### What An Eigenvector Is

An eigenvector of a matrix `A` is a nonzero vector `v` such that:

`Av = lambda v`

for some scalar `lambda`, called the eigenvalue.

Interpretation:

- most vectors change both direction and magnitude under a transformation
- an eigenvector is a special direction that keeps its direction
- the eigenvalue tells how much that direction is stretched, shrunk, or flipped

This is why eigenvectors matter: they reveal the intrinsic directional structure
of a transformation.

### Geometric Intuition

Imagine applying a matrix to many arrows in the plane.

Most arrows rotate and stretch in complicated ways.
But a few special directions remain on the same line after transformation.
Those are the eigenvector directions.

Visualization:

```text
general vector     -> changes direction and length
eigenvector        -> keeps direction, only length/sign changes
eigenvalue         -> tells how much scaling happens
```

### Why Eigenvalues Matter In ML

Eigenvalues tell us about:

- curvature
- stability
- growth or decay along directions
- conditioning of optimization problems
- principal variance directions in data

This is why they appear in:

- PCA
- Hessian analysis
- covariance matrices
- dynamical systems and recurrent networks
- spectral methods

### Why Covariance Matrices Matter

Suppose the data matrix is centered. The covariance matrix is:

`Sigma = (1/n) X^T X`

This matrix summarizes how features vary and co-vary.

Its eigenvectors give the principal directions of variation.
Its eigenvalues give the amount of variance along those directions.

This is the core mathematical reason PCA works.

## Deep Dive

### Matrices As Learned Coordinate Changes

A matrix can be interpreted as:

- scaling
- rotation
- reflection
- shear
- projection

or combinations of these.

When a neural network learns a weight matrix, it is learning how to re-express
the input in a more useful coordinate system for the next layer.

This is why matrix multiplication is central to representation learning: it
creates new coordinate systems aligned with predictive structure.

### Spectral Decomposition Intuition

If a matrix is diagonalizable:

`A = Q Lambda Q^(-1)`

then:

- columns of `Q` are eigenvectors
- diagonal entries of `Lambda` are eigenvalues

This says:

```text
change coordinates into eigenvector basis
       ->
matrix acts as simple per-direction scaling
       ->
change back
```

That is why eigen-analysis is powerful: it converts a complicated transformation
into a simpler directional description.

### Why Hessian Eigenvalues Matter

For a loss `L(theta)`, the Hessian `H` contains second derivatives:

`H_ij = d^2L / dtheta_i dtheta_j`

Locally:

`L(theta + delta) approx L(theta) + grad L^T delta + 1/2 delta^T H delta`

The eigenvectors of `H` are principal curvature directions.
The eigenvalues of `H` tell curvature magnitude along those directions.

Interpretation:

- large positive eigenvalue -> steep upward curvature
- small positive eigenvalue -> flat direction
- negative eigenvalue -> direction of downward curvature, often indicating a
  saddle rather than a minimum

This is why Hessian eigenvalues help explain optimization difficulty.

### Conditioning And Optimization

If the Hessian or a quadratic form has eigenvalues with very different scales,
the problem is ill-conditioned.

Condition number:

`kappa = lambda_max / lambda_min`

for positive eigenvalues.

Large `kappa` means:

- some directions are very steep
- others are very flat
- gradient descent must use a learning rate safe for the steep directions
- progress in flat directions becomes slow

This produces zig-zagging and slow convergence.

### Why PCA Works

PCA solves:

- find directions of maximum variance

Mathematically, this becomes an eigenvalue problem for the covariance matrix.

If:

`Sigma v = lambda v`

then:

- `v` gives a principal direction
- `lambda` gives variance captured along that direction

Why this works:

The covariance matrix encodes second-order structure of the data. Its eigenbasis
is the coordinate system where that structure becomes simplest.

In that basis:

- coordinates are decorrelated
- variance is sorted by magnitude

### Singular Values Versus Eigenvalues

In machine learning, singular values are often even more useful than eigenvalues.

For a matrix `A`, the singular value decomposition is:

`A = U S V^T`

where singular values in `S` measure scaling strength along orthogonal
directions.

Why this matters:

- data matrices may not be square
- weight matrices may not be symmetric
- singular values describe amplification and contraction robustly

So when thinking about exploding or vanishing behavior in deep nets, singular
values are often more directly relevant than eigenvalues.

### Eigenvalues In Recurrent Networks

For a recurrent update:

`h_t = W h_(t-1)`

repeated multiplication by `W` governs long-term behavior.

If eigenvalues of `W` have magnitude:

- greater than 1 -> growth / explosion
- less than 1 -> decay / vanishing
- around 1 -> persistent signal

This is one reason recurrent networks historically suffered from exploding and
vanishing gradients.

The same spectral intuition explains why gating and normalization help.

### Why Symmetric Matrices Are Special

Many important ML matrices are symmetric:

- covariance matrices
- Hessians, when well-defined
- Gram matrices

Symmetric matrices have especially nice properties:

- real eigenvalues
- orthogonal eigenvectors
- stable geometric interpretation

This is why so much theory is built around them.

### Matrix Multiplication As Composition

If:

`y = A(Bx)`

then:

`y = (AB)x`

So matrix multiplication naturally captures sequential transformations.

This is exactly what a neural network does across layers:

```text
input
  ->
first learned transform
  ->
second learned transform
  ->
...
```

With non-linearities inserted between these transforms, the model becomes far
more expressive, but the linear maps remain the backbone of the computation.

### Why Diagonalization Helps Intuition

Even when a matrix is not explicitly diagonalized during training, diagonalizing
it in theory helps us understand:

- which directions matter
- where information is amplified
- where curvature is large
- how optimization behaves

This is why eigenvectors are useful: they decompose complex coupled behavior
into uncoupled directional components.

### The Statistical Meaning Of Leading Eigenvectors

In covariance settings, the top eigenvectors reveal the dominant directions in
which the data varies.

This matters because:

- variance often reflects structure
- low-variance directions may be mostly noise
- dimensionality reduction can keep informative directions and discard less
  useful ones

This is one way linear algebra becomes a statistical tool.

### Final Synthesis

Matrices work in machine learning because models repeatedly apply linear
transformations to vector representations.
Eigenvectors matter because they reveal the special directions of those
transformations.
Eigenvalues matter because they quantify scaling, curvature, variance, stability,
and conditioning along those directions.

Together, they explain:

- how layers transform data
- how curvature shapes optimization
- why PCA extracts principal structure
- why recurrent dynamics can explode or vanish
- why some optimization problems are easy and others are hard
