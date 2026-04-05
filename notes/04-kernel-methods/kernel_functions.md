# Kernel Functions

## Intro And Concepts

A kernel function is not merely a similarity score. In machine learning, a
kernel is a function `K(x, z)` that behaves like an inner product in some
possibly high-dimensional feature space:

`K(x, z) = <phi(x), phi(z)>`

for a feature map `phi`.

This single equation is the central idea behind kernel methods. It says that a
model may operate linearly in the feature space `phi(x)` while appearing
non-linear in the original input space `x`.

### Why Kernels Matter

Many learning algorithms are easiest to derive and optimize in linear form.

Examples:

- linear classifiers
- linear regressors
- margin-based methods
- principal component methods

But raw input space is often not geometrically aligned with the task.

Typical mismatch:

- classes are not linearly separable
- regression relationship is curved
- similarity in Euclidean space does not reflect semantic closeness

Kernels let us keep the mathematics of linear methods while changing the
geometry in which linearity is defined.

### The Basic Geometric Intuition

Suppose points in 2D form concentric regions:

```text
          o o o
       o         o
      o    x x    o
       o   x x   o
          o o o
```

No straight line in the original 2D plane separates inner `x` points from outer
`o` points.

Now define a new feature:

`r^2 = x_1^2 + x_2^2`

In the transformed space, points are no longer organized as a ring versus
center. They are separated by radius. A linear separator in the transformed
space can solve the problem.

Kernel methods exploit exactly this phenomenon, often without explicitly forming
the transformed features.

### The Kernel Trick

Suppose an algorithm depends only on dot products:

`x_i^T x_j`

If we explicitly transform inputs, the algorithm would instead use:

`phi(x_i)^T phi(x_j)`

If we have a kernel satisfying:

`K(x_i, x_j) = phi(x_i)^T phi(x_j)`

then we can replace every transformed dot product by a kernel evaluation.

This is the kernel trick.

It matters because `phi(x)` may be:

- very high-dimensional
- combinatorially large
- infinite-dimensional

yet `K(x, z)` may still be easy to compute.

### Why A Linear Method Becomes Non-Linear

A linear predictor in feature space has the form:

`f(x) = w^T phi(x) + b`

This is linear in `phi(x)`, but not necessarily linear in the original input
coordinates of `x`.

That is the key conceptual move:

```text
linear in transformed space
!=
linear in original space
```

So kernel methods are not "non-linear algorithms" in a vague sense. They are
linear methods expressed in a different geometry.

### The Dual View

Many kernelized methods are easier to express in dual form. A predictor often
takes the form:

`f(x) = sum_i alpha_i K(x_i, x) + b`

where:

- `x_i` are training examples
- `alpha_i` are learned coefficients

This form says the prediction at a new point is built from its similarity to
training points.

That leads to the correct intuition:

> A kernel method often predicts by comparing a new point to strategically
> weighted training examples, not by using only a direct coefficient vector on
> raw features.

### Gram Matrix / Kernel Matrix

Given training examples `x_1, ..., x_n`, define:

`G_ij = K(x_i, x_j)`

This matrix is called the Gram matrix or kernel matrix.

It is central because many kernel methods operate entirely through `G`.

```text
training points
   ->
pairwise kernel evaluations
   ->
kernel matrix G
   ->
optimization / prediction
```

### What Makes A Function A Valid Kernel

Not every similarity function is a kernel in the machine-learning sense.

A valid kernel must correspond to an inner product in some feature space. A
practical equivalent condition is:

- for every finite dataset, the kernel matrix must be symmetric and positive
  semidefinite

That means for any vector `c`:

`c^T G c >= 0`

This condition guarantees the geometry implied by the kernel is internally
consistent with an inner-product space.

### Mercer-Style Intuition

Very loosely, if a kernel is valid, then there exists some feature space in
which the kernel acts exactly like a dot product.

This is the reason the expression:

`K(x, z) = <phi(x), phi(z)>`

is not just notation. It is the mathematical foundation that lets optimization,
geometry, and generalization theory carry over from linear methods.

### The Main Kernel Families

#### Linear Kernel

`K(x, z) = x^T z`

This produces no extra non-linearity. It is useful when the data is already
close to linearly separable or when dimensionality is very high.

#### Polynomial Kernel

`K(x, z) = (x^T z + c)^d`

This implicitly introduces monomials and interactions up to degree `d`.

A degree-2 polynomial kernel behaves as if the model had access to features
like:

- `x_1^2`
- `x_2^2`
- `x_1 x_2`

without explicitly creating them.

#### RBF / Gaussian Kernel

`K(x, z) = exp(-gamma ||x - z||^2)`

This is a locality-based kernel. Two points are similar if they are close in
Euclidean distance. It is extremely flexible and corresponds to an
infinite-dimensional feature space.

#### Sigmoid Kernel

`K(x, z) = tanh(alpha x^T z + c)`

This resembles a neural activation shape, but unlike the previous kernels it is
less universally reliable in practice and is not valid for all parameter
choices.

### Visualization: How Different Kernels Think

```text
linear kernel:
similarity = alignment

polynomial kernel:
similarity = alignment + interactions + higher-order terms

RBF kernel:
similarity = locality in distance

sigmoid kernel:
similarity = squashed alignment
```

### Why Feature Scaling Matters

Distance-based and dot-product-based kernels are sensitive to scale.

If one feature has much larger magnitude than the others, then:

- dot products are dominated by it
- Euclidean distances are distorted by it
- the kernel geometry becomes misleading

So standardization is often not optional for kernels. It is part of defining a
sensible geometry.

### A Practical Mental Model

You can think of kernels as learned or chosen notions of similarity that let
simple linear machinery operate in a richer implicit space.

That is a better mental model than "kernels add features" because kernels do
more than add features: they redefine geometry, locality, and smoothness.

## Deep Dive

### Explicit Feature Maps Versus Implicit Feature Maps

Suppose the original input is `x = [x_1, x_2]`.

A degree-2 explicit map might be:

`phi(x) = [x_1^2, sqrt(2)x_1x_2, x_2^2]`

Then:

`phi(x)^T phi(z) = (x^T z)^2`

This concrete example shows what a kernel really does: it replaces explicit
feature construction with direct inner-product evaluation in the transformed
space.

For finite-dimensional maps, this is mostly a computational convenience.
For infinite-dimensional maps, it becomes conceptually essential.

### Why Kernelized Models Depend On Dot Products

Many linear models can be rewritten so that all dependence on data occurs
through dot products between examples.

This is especially true for:

- SVMs
- ridge regression in dual form
- Gaussian process covariance computation
- kernel PCA

Once an algorithm is expressed entirely through `x_i^T x_j`, kernelization is
often immediate:

`x_i^T x_j -> K(x_i, x_j)`

That substitution changes the hypothesis space without changing the top-level
optimization machinery too much.

### Kernel Ridge Regression

Consider ridge regression in primal form:

`min_w ||Xw - y||^2 + lambda ||w||^2`

In feature space this becomes:

`min_w ||Phi w - y||^2 + lambda ||w||^2`

Using the representer theorem, the solution can be written in terms of training
examples:

`f(x) = sum_i alpha_i K(x_i, x)`

and the coefficients solve:

`alpha = (G + lambda I)^(-1) y`

where `G` is the kernel matrix.

This is one of the cleanest examples of how kernels turn a linear regularized
method into a non-linear one.

### Kernel SVM Intuition

A kernel SVM finds a large-margin separator in feature space, not in raw input
space.

The decision function has the form:

`f(x) = sum_i alpha_i y_i K(x_i, x) + b`

Only certain training points have non-zero `alpha_i`; these are the support
vectors.

Geometrically:

- support vectors define the boundary
- the kernel defines the geometry in which margins are measured
- optimization chooses the maximum-margin solution in that geometry

Visualization:

```text
support vectors
      +
chosen kernel geometry
      +
margin optimization
      =
non-linear decision boundary in input space
```

### RBF Kernel As Local Basis Expansion

The RBF kernel can be understood as attaching a localized bump around each
training point.

If `x` is close to `x_i`, then `K(x_i, x)` is large.
If it is far away, then `K(x_i, x)` is small.

So a predictor of the form:

`f(x) = sum_i alpha_i exp(-gamma ||x_i - x||^2)`

is a weighted superposition of localized influence functions.

This gives the right intuition for why RBF kernels are so flexible: they can
build complex decision surfaces out of many local contributions.

### Geometry Of Gamma In The RBF Kernel

For:

`K(x, z) = exp(-gamma ||x - z||^2)`

`gamma` controls the radius of influence.

#### Small `gamma`

- similarity decays slowly
- each point influences a broad region
- decision boundaries are smoother
- bias tends to increase, variance tends to decrease

#### Large `gamma`

- similarity decays quickly
- each point influences only a narrow neighborhood
- decision boundaries can become highly irregular
- bias tends to decrease, variance tends to increase

Visualization:

```text
small gamma  -> broad hills of influence
large gamma  -> sharp narrow spikes of influence
```

### Geometry Of Degree In The Polynomial Kernel

For:

`K(x, z) = (x^T z + c)^d`

the degree `d` determines how much higher-order interaction structure is
available.

As `d` grows:

- the hypothesis class becomes more expressive
- sensitivity to interactions increases
- overfitting risk can increase

Polynomial kernels are often best when the task likely depends on global,
structured interactions rather than purely local neighborhoods.

### The Representer Theorem

A deep reason kernels are useful is the representer theorem. In many regularized
learning problems, even though the optimization is over a potentially enormous
function space, the optimal solution lies in the span of kernel evaluations on
the training data:

`f^*(x) = sum_i alpha_i K(x_i, x)`

This is why kernel methods remain computationally tractable despite implicitly
living in rich function spaces.

### RKHS Intuition

Associated with a valid kernel is a reproducing kernel Hilbert space, or RKHS.

You do not need the full functional analysis to use kernels, but conceptually it
helps to know:

- the kernel defines the function space
- the RKHS norm defines what "complexity" means in that space
- regularization penalizes complex functions according to that geometry

So choosing a kernel is not just choosing similarity. It is choosing:

- what functions are easy to express
- what smoothness means
- what complexity penalty is natural

This is one of the most important deep intuitions in kernel methods.

### Visualization: Kernel Choice Changes Function Space

```text
choose linear kernel
   -> linear functions are simple

choose polynomial kernel
   -> low-degree algebraic interactions are simple

choose RBF kernel
   -> smooth local interpolation is simple
```

The kernel is effectively a structural prior over admissible functions.

### Valid Kernels And Positive Semidefiniteness

Given any finite set of points, a valid kernel must produce a symmetric positive
semidefinite matrix.

Why that matters:

- eigenvalues are non-negative
- optimization problems remain convex where expected
- the matrix can genuinely be interpreted as a Gram matrix

If a function fails this condition, it may not correspond to any consistent
inner-product space, and the downstream algorithm may lose its theoretical
foundation.

### Why Sigmoid Is Less Standard

The sigmoid kernel is appealing because it resembles neural activations, but it
is less clean than linear, polynomial, or Gaussian kernels.

Issues:

- not all parameter settings yield a valid PSD kernel
- behavior can be sensitive to scaling
- it often performs less predictably than RBF

So although it appears in textbooks, it is much less of a default choice in
practice.

### Kernel Methods Versus Explicit Feature Engineering

There are two ways to introduce non-linearity.

#### Explicit Features

Manually build transformed features:

- polynomial terms
- interaction terms
- basis expansions

#### Kernel Methods

Keep the original input representation, but replace inner products with a kernel.

Trade-off:

- explicit features can be cheaper for very large datasets if the feature map is
  manageable
- kernels can express much richer spaces with elegant math, but often scale as
  `O(n^2)` or worse in dataset size because of the kernel matrix

### Computational Cost

Kernel methods are powerful, but they do not scale arbitrarily well.

If there are `n` training points:

- the kernel matrix is `n x n`
- memory is typically `O(n^2)`
- many solvers have superlinear time complexity in `n`

This is why classic kernel methods are strongest on small-to-medium datasets or
when approximations are used.

### Overfitting And Regularization

A flexible kernel can overfit if its effective capacity is too high.

Typical causes:

- RBF kernel with very large `gamma`
- polynomial kernel with very high degree
- SVM regularization parameter `C` too large
- insufficient noise control in the data

The right bias-variance intuition is:

```text
more flexible kernel / weaker regularization
    ->
lower bias, higher variance

less flexible kernel / stronger regularization
    ->
higher bias, lower variance
```

### Choosing A Kernel In Practice

#### Choose Linear When

- feature dimension is already large
- the task is approximately linear
- interpretability or speed matters

#### Choose Polynomial When

- you expect structured interactions
- moderate global non-linearity is likely
- local similarity is less important than algebraic interaction

#### Choose RBF When

- non-linearity is clear
- you want a strong default
- you are comfortable tuning `gamma` and regularization

#### Choose Sigmoid Rarely

- mainly for experimentation
- only with careful validation

### Tiny Worked Example

Let:

- `x = [1, 2]`
- `z = [3, 4]`

#### Linear Kernel

`K(x, z) = 1*3 + 2*4 = 11`

#### Polynomial Kernel With `c = 1`, `d = 2`

`K(x, z) = (11 + 1)^2 = 144`

#### RBF Kernel With `gamma = 0.5`

First compute:

`||x - z||^2 = (1 - 3)^2 + (2 - 4)^2 = 8`

Then:

`K(x, z) = exp(-0.5 * 8) = exp(-4)`

This is small, meaning the RBF kernel considers these points relatively far
apart.

### Final Synthesis

A kernel function is best understood as a mathematically valid inner-product
substitute that changes the geometry of learning.

It does four things at once:

- defines similarity
- defines an implicit feature space
- defines a function class
- defines the geometry in which linear methods operate

That is why kernels remain one of the most elegant ideas in machine learning:
they let us trade raw-space linearity for feature-space linearity without having
to explicitly build the feature space itself.
