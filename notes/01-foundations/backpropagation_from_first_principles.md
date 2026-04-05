# Backpropagation From First Principles

## Intro And Concepts

Backpropagation is the algorithmic procedure used to compute gradients of a loss
with respect to all parameters in a neural network efficiently.

It is not a different learning rule from gradient descent. Instead:

- gradient descent tells us how to update parameters once gradients are known
- backpropagation tells us how to compute those gradients in layered models

If a network defines:

`L(theta) = loss(f_theta(x), y)`

then training needs:

`grad_theta L`

For a deep network with millions or billions of parameters, computing those
gradients naively would be prohibitively expensive. Backpropagation exploits the
compositional structure of the network to reuse intermediate derivatives.

### The Core Idea

A neural network is a composition:

`x -> z_1 -> a_1 -> z_2 -> a_2 -> ... -> z_L -> L`

where each layer depends on previous layers.

Because the loss depends on earlier parameters only through later computations,
the chain rule gives:

`dL/dtheta = dL/dz_L * dz_L/dz_(L-1) * ... * dz_k/dtheta`

Backpropagation computes this chain efficiently by moving backward from the loss
to the earliest parameters.

### Why It Works

Backpropagation works because of two facts:

1. the network is differentiable almost everywhere
2. the chain rule factorizes global sensitivity into local sensitivities

This means each layer does not need to know the whole network. It only needs:

- the gradient flowing in from above
- its own local derivative
- the activations it saw during the forward pass

### Computational Graph View

It helps to think of the network as a directed computational graph.

```text
inputs
  ->
linear transforms
  ->
nonlinearities
  ->
logits / outputs
  ->
loss
```

Each node stores:

- a forward value
- a local derivative rule

The backward pass applies reverse-mode automatic differentiation over this graph.

### Why Reverse-Mode Is The Right Mode

If a scalar loss depends on many parameters, reverse-mode autodiff is efficient
because:

- one forward pass computes values
- one backward pass computes derivatives with respect to many parameters

This matches machine learning perfectly:

- inputs may be moderate
- parameters are huge
- loss is usually scalar

That is why backpropagation is essentially reverse-mode autodiff specialized to
neural networks.

### The Main Intuition

The backward pass answers:

> If this intermediate value had changed slightly, how much would the final loss
> have changed?

This quantity is the error signal flowing backward.

At each layer:

- upstream gradient says how much the output mattered
- local derivative says how the layer transformed its input
- multiplying them gives how much the earlier quantity mattered

## Deep Dive

### Single Neuron Derivation

Consider one neuron:

`z = w^T x + b`

`a = sigma(z)`

`L = L(a)`

We want gradients with respect to `w`, `b`, and `x`.

By the chain rule:

`dL/dz = dL/da * da/dz`

Then:

`dL/dw = dL/dz * dz/dw = dL/dz * x`

`dL/db = dL/dz`

`dL/dx = dL/dz * dz/dx = dL/dz * w`

In vector form:

`grad_w L = delta x^T`

where:

`delta = dL/dz`

This pattern repeats throughout all dense layers.

### Layerwise Backpropagation

For layer `l`:

`z_l = W_l a_(l-1) + b_l`

`a_l = sigma_l(z_l)`

Suppose we know:

`delta_l = dL/dz_l`

Then:

`dL/dW_l = delta_l a_(l-1)^T`

`dL/db_l = delta_l`

and the gradient passed to the previous activation is:

`dL/da_(l-1) = W_l^T delta_l`

Then:

`delta_(l-1) = (W_l^T delta_l) * sigma'_(l-1)(z_(l-1))`

where `*` denotes elementwise multiplication.

This is the recursive heart of backpropagation.

### Why The Forward Pass Must Be Stored

The backward pass needs intermediate values from the forward pass:

- activations
- pre-activations
- masks for dropout or ReLU regions
- normalization statistics in some layers

This is why training generally uses more memory than inference.

The backward pass is not just recomputing from scratch. It depends on cached
intermediate structure.

### Matrix Shape Intuition

Backpropagation becomes much easier once matrix shapes are explicit.

If:

- `a_(l-1)` has shape `(d_(l-1), 1)`
- `W_l` has shape `(d_l, d_(l-1))`
- `z_l` has shape `(d_l, 1)`
- `delta_l` has shape `(d_l, 1)`

then:

`dL/dW_l = delta_l a_(l-1)^T`

has shape:

`(d_l, 1)(1, d_(l-1)) = (d_l, d_(l-1))`

which matches `W_l`.

This is one reason backprop feels natural in matrix form: the shapes line up
exactly with the structure of the forward computation.

### Softmax And Cross-Entropy

One of the most important special cases is:

- logits `z`
- probabilities `p = softmax(z)`
- cross-entropy loss

Then:

`dL/dz = p - y`

This is a remarkable simplification because it avoids carrying the full softmax
Jacobian explicitly through the loss.

It is one reason softmax-cross-entropy is computationally elegant.

### Reverse-Mode Efficiency

Why is backprop efficient?

Suppose the network computes a scalar loss from many parameters. Forward-mode
autodiff would compute one directional derivative at a time. Reverse-mode
instead propagates one adjoint per intermediate quantity.

This means the cost of computing gradients with respect to all parameters is
often only a small constant multiple of the cost of the forward pass.

That is the core computational reason deep learning is feasible at scale.

### Backprop And The Chain Rule

The chain rule in one variable says:

`dL/dx = dL/du * du/dx`

In vector settings, this becomes Jacobian composition.

Backpropagation is the systematic reuse of this principle across all intermediate
computations:

```text
global derivative
    =
product / composition of local derivatives
```

This is the conceptual bridge between calculus and deep learning.

### Vanishing And Exploding Gradients

Backpropagation also explains a major training pathology.

If gradients are repeatedly multiplied by:

- numbers smaller than 1 in magnitude -> they shrink exponentially
- numbers larger than 1 in magnitude -> they grow exponentially

Then early layers may receive:

- tiny gradients: vanishing gradients
- huge gradients: exploding gradients

This is especially severe in deep networks and recurrent networks where long
chains of derivatives are multiplied together.

That is why:

- initialization matters
- normalization matters
- residual connections matter
- gating matters in RNNs

### Why Backpropagation Is Not "Error Copying"

A common shallow intuition is that backprop just "sends the error backward."
That is incomplete.

What is actually sent backward is a sensitivity:

- how much the loss would change if an intermediate quantity changed

This is not the same thing as the prediction error itself.

So a better statement is:

> Backpropagation propagates local responsibility for the final loss through the
> computational graph.

### Second-Order Perspective

Backpropagation gives first derivatives.
If we wanted curvature information, we would need second derivatives or Hessian-
vector products.

Even without computing full second-order information, backprop gives the first-
order signal that makes large-scale optimization practical. Most modern
optimization in deep learning builds on top of these gradients rather than
replacing them.

### Backprop Through Time

For recurrent networks, backpropagation is applied through an unrolled temporal
graph:

```text
h_1 -> h_2 -> h_3 -> ... -> h_T
```

The same parameters are reused at each time step, and gradients from all time
steps accumulate onto them.

This is called backpropagation through time, or BPTT, and it is the key training
mechanism behind RNNs, LSTMs, and GRUs.

That is why backprop sits naturally before sequence models in a study order.

### Final Synthesis

Backpropagation is the efficient gradient engine of neural networks.

It works because:

- networks are compositional
- derivatives obey the chain rule
- reverse-mode autodiff is efficient for scalar losses with many parameters

Mathematically, it decomposes global sensitivity into local derivatives.
Practically, it makes modern neural-network training possible.
