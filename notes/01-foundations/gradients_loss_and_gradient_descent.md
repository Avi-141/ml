# Gradients, Loss Functions, And Gradient Descent

## Intro And Concepts

The central training question in a neural network is:

> Given a loss function, how should the parameters move so that the loss goes
> down?

The answer begins with the gradient.

If the model has parameters `theta` and loss `L(theta)`, the gradient is:

`grad_theta L(theta)`

It points in the direction of steepest local increase of the loss. Therefore:

`-grad_theta L(theta)`

points in the direction of steepest local decrease.

That is the basic reason gradients work.

### Intuition: Local Linear Approximation

A differentiable function can be locally approximated as:

`L(theta + delta) approx L(theta) + grad L(theta)^T delta`

This is the first-order Taylor approximation.

If we want to decrease `L`, we want:

`grad L(theta)^T delta < 0`

The most direct choice is:

`delta = -eta grad L(theta)`

where `eta > 0` is the learning rate.

Then:

`grad L(theta)^T delta = -eta ||grad L(theta)||^2 <= 0`

So for sufficiently small steps, moving opposite the gradient lowers the loss.

This is the mathematical heart of gradient descent.

### What A Loss Function Really Does

A loss is not just an error score. It defines the geometry of training.

The loss tells the optimizer:

- what kinds of mistakes matter most
- how strongly different mistakes are penalized
- how sensitive the optimization should be to prediction changes

Examples:

#### Mean Squared Error

`L = (1/n) sum_i (y_hat_i - y_i)^2`

This makes large errors especially costly and leads to smooth gradients.

#### Cross-Entropy

`L = -(1/n) sum_i log p_theta(y_i | x_i)`

This penalizes confidently wrong predictions very strongly and is well aligned
with probabilistic classification.

### Why "Error" And "Loss" Are Not The Same

Error is often the informal notion of being wrong.
Loss is the differentiable objective used for optimization.

Example:

- classification error is 0/1
- but 0/1 loss is not convenient for gradient-based optimization

So we use surrogate losses such as cross-entropy because they are smooth,
informative, and trainable.

### Why Backpropagation Works

A neural network is a composition of functions:

`f_theta(x) = f_L(...f_2(f_1(x)))`

To compute how the loss changes with respect to early parameters, we apply the
chain rule.

If:

`L = L(z_L)`

and:

`z_l = f_l(z_(l-1))`

then:

`dL/dz_(l-1) = dL/dz_l * dz_l/dz_(l-1)`

Backpropagation is simply an efficient repeated application of the chain rule
through the computational graph.

It works because each layer contributes local derivatives, and the total effect
is the product/composition of these local sensitivities.

### Why Gradient Descent Is Reasonable Even For Nonconvex Problems

Neural-network losses are typically nonconvex, so gradient descent is not
guaranteed to find a global optimum in the classical sense.

Yet it still works well in practice because:

- local gradient information is often enough to make progress
- high-dimensional landscapes contain many acceptable minima
- stochasticity helps escape some bad regions
- the objective and architecture impose structure

So gradient descent is not "magic." It is an effective local search method in a
structured high-dimensional space.

## Deep Dive

### The Direction Of Steepest Descent

Why exactly is the negative gradient the steepest descent direction?

Consider all small directions `u` with fixed norm `||u|| = 1`. The directional
derivative is:

`D_u L(theta) = grad L(theta)^T u`

By Cauchy-Schwarz:

`grad L(theta)^T u >= -||grad L(theta)|| ||u|| = -||grad L(theta)||`

Equality is achieved when:

`u = -grad L(theta) / ||grad L(theta)||`

So among all unit directions, the negative gradient gives the steepest local
drop.

This is the precise mathematical statement behind gradient descent.

### Gradient Descent Update

The standard update is:

`theta_(t+1) = theta_t - eta grad L(theta_t)`

Interpretation:

- `grad L(theta_t)` tells us local sensitivity
- `eta` controls how far we trust that local linearization

If `eta` is too small:

- training is slow

If `eta` is too large:

- the local approximation becomes inaccurate
- updates may overshoot or diverge

### Why Learning Rate Matters

The gradient gives direction, not a safe absolute step size.

The loss surface may have:

- steep directions
- flat directions
- highly curved valleys

A step size that is safe in one region may be too large in another.

This is one reason optimization is difficult and adaptive methods are useful.

### Gradient Descent Through A Quadratic Lens

The cleanest intuition comes from a quadratic objective:

`L(theta) = 1/2 theta^T A theta - b^T theta`

where `A` is symmetric positive semidefinite.

Then:

`grad L(theta) = A theta - b`

and gradient descent becomes:

`theta_(t+1) = theta_t - eta (A theta_t - b)`

If curvature differs strongly across directions, progress is fast in some
directions and slow in others.

This is why conditioning matters.

### Why Ill-Conditioning Slows Training

Suppose one direction of parameter space is very steep and another is very flat.
Then a learning rate small enough for the steep direction may be tiny for the
flat one.

This causes zig-zagging:

```text
steep valley walls
    ->
small stable learning rate
    ->
slow progress along shallow valley floor
```

This is one reason second-order information and adaptive optimizers help.

### Loss Functions And Statistical Assumptions

Loss functions often come from likelihood models.

#### Squared Error And Gaussian Noise

If:

`Y = f_theta(X) + epsilon`

with Gaussian noise:

`epsilon ~ N(0, sigma^2)`

then maximizing likelihood is equivalent to minimizing squared error.

#### Cross-Entropy And Categorical Models

If the model outputs class probabilities:

`p_theta(y|x)`

then maximizing likelihood is equivalent to minimizing negative log-likelihood,
which becomes cross-entropy.

This is why common losses are not arbitrary penalties. They are often derived
from probabilistic assumptions.

### Why Cross-Entropy Gives Useful Gradients

For classification, the 0/1 error is not useful for gradient descent:

- it is discontinuous
- it gives no sense of how wrong a wrong answer is

Cross-entropy fixes this because it is sensitive to confidence.

If the model is confidently wrong, cross-entropy gives a very large penalty.
If it is slightly wrong, the penalty is smaller.

This creates informative gradients that tell the model not only that it is
wrong, but how wrong and in which probabilistic direction.

### Backpropagation In A Simple Layer

Consider:

`z = Wx + b`

and loss `L(z)`.

Then:

- `dL/dW = (dL/dz) x^T`
- `dL/db = dL/dz`
- `dL/dx = W^T (dL/dz)`

This shows the essential linear algebra of backprop:

- upstream gradient arrives at a layer
- local derivatives transform it
- parameter gradients are built from activations and upstream error signals

In a network, this repeats layer by layer.

### Why Mini-Batch Gradient Descent Works

The full empirical gradient is:

`g = (1/n) sum_i grad L_i(theta)`

Mini-batch gradient descent uses an estimate:

`g_B = (1/|B|) sum_(i in B) grad L_i(theta)`

This estimator is noisy but usually much cheaper.

Benefits:

- computational efficiency
- frequent updates
- stochasticity that may help explore the landscape

This yields stochastic gradient descent, SGD, and its variants.

### Momentum Intuition

Plain gradient descent reacts only to the current gradient.
Momentum accumulates past directions:

`v_(t+1) = beta v_t + grad L(theta_t)`

`theta_(t+1) = theta_t - eta v_(t+1)`

This helps because:

- consistent directions are amplified
- oscillations across steep directions are damped

Momentum can be understood as smoothing the gradient signal over time.

### Why Gradients Sometimes Fail

Common failure modes:

- vanishing gradients: derivatives become tiny through many layers
- exploding gradients: derivatives blow up through repeated multiplication
- saddle points: gradient near zero but not a minimum
- poor conditioning: updates zig-zag and slow down

Architectural and optimizer design often exists to address these issues:

- ReLU and residual connections help with vanishing gradients
- normalization stabilizes activations
- adaptive methods help with uneven scale
- gradient clipping helps with exploding gradients

### First-Order Versus Second-Order Thinking

Gradient descent uses first-order information only:

- slope

Second-order methods also use curvature:

- Hessian

If the local quadratic approximation is:

`L(theta + delta) approx L(theta) + grad L^T delta + 1/2 delta^T H delta`

then curvature tells us which directions are steep, flat, or unstable.

Full second-order methods are often too expensive for large neural networks, but
their intuition is still essential for understanding optimization.

### Why Gradient Descent Works In Practice

Gradient descent works because:

1. differentiable models provide local sensitivity information
2. the negative gradient gives a principled local descent direction
3. repeated local improvements can navigate high-dimensional objectives
4. stochasticity and architecture bias often steer optimization toward useful
   solutions

This is not a proof of perfect convergence to the best possible model. It is an
explanation for why local derivative information is enough to build powerful
learning systems.

### Final Synthesis

Gradients work because differentiability lets us locally linearize the loss.
Loss functions work because they translate prediction quality into a trainable
objective, often with probabilistic meaning.
Gradient descent works because moving opposite the gradient is the steepest local
decrease direction, and repeated local improvements are often enough to find
useful parameters in high-dimensional neural-network landscapes.

## Further Reading

- `Canonical paper`: Bottou, Curtis, and Nocedal, "Optimization Methods for Large-Scale Machine Learning" (2018).
- `Best intuition resource`: Distill, *Why Momentum Really Works*, plus the CS231n optimization notes.
- `Best practical code resource`: `karpathy/micrograd` on GitHub.
