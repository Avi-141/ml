# Why Adam Works

## Intro And Concepts

Adam is an adaptive first-order optimization method that combines two ideas:

- momentum
- per-parameter adaptive scaling

The standard update is:

`m_t = beta_1 m_(t-1) + (1 - beta_1) g_t`

`v_t = beta_2 v_(t-1) + (1 - beta_2) g_t^2`

`m_hat_t = m_t / (1 - beta_1^t)`

`v_hat_t = v_t / (1 - beta_2^t)`

`theta_(t+1) = theta_t - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)`

where:

- `g_t` is the gradient at step `t`
- `m_t` is the first-moment estimate
- `v_t` is the second-moment estimate
- `alpha` is the base learning rate

Adam works well in practice because it does not treat all parameters or all
directions equally. It smooths noisy gradients over time and scales updates by
their recent magnitude.

### Why Plain Gradient Descent Is Often Not Enough

With plain gradient descent:

`theta_(t+1) = theta_t - alpha g_t`

every parameter shares the same learning rate.

This is problematic when:

- some parameters receive large gradients
- others receive tiny gradients
- gradients are noisy across mini-batches
- curvature differs strongly across directions

Adam addresses this by using local gradient statistics.

### The First Idea: Momentum

The moving average:

`m_t = beta_1 m_(t-1) + (1 - beta_1) g_t`

acts like smoothed velocity.

Intuition:

- if gradients keep pointing in a similar direction, Adam builds momentum
- if gradients fluctuate noisily, the moving average dampens the noise

This helps accelerate progress in stable directions and reduce oscillation.

### The Second Idea: Adaptive Scaling

The second moment estimate:

`v_t = beta_2 v_(t-1) + (1 - beta_2) g_t^2`

tracks recent squared gradient magnitude coordinatewise.

Then the update divides by:

`sqrt(v_hat_t) + epsilon`

Meaning:

- parameters with consistently large gradients get smaller effective steps
- parameters with small gradients get relatively larger effective steps

This is why Adam is especially useful in settings with uneven gradient scales.

### Why Bias Correction Is Needed

At the start of training, `m_t` and `v_t` are initialized at zero, so their
moving averages are biased toward zero in early iterations.

Bias correction compensates for that:

`m_hat_t = m_t / (1 - beta_1^t)`

`v_hat_t = v_t / (1 - beta_2^t)`

Without this correction, early updates would be systematically mis-scaled.

### Intuition In One Sentence

Adam works because it moves in a momentum-smoothed gradient direction while
automatically shrinking or enlarging the step size for each parameter according
to recent gradient scale.

## Deep Dive

### Adam As Momentum Plus RMS Scaling

Adam can be seen as combining:

- momentum-style averaging of gradients
- RMSProp-style normalization by recent squared gradients

This is why it often behaves robustly out of the box:

- momentum helps with direction
- adaptive scaling helps with step size

The first moment captures directionality.
The second moment captures scale.

### Why Coordinatewise Adaptation Helps

Suppose one parameter has gradients of order `100` and another has gradients of
order `0.001`.

A single global learning rate is awkward:

- large enough for the small-gradient parameter may explode the large-gradient
  one
- safe for the large-gradient parameter may barely move the small-gradient one

Adam approximately solves this by using effective per-parameter step sizes:

`effective_step_i approx alpha / sqrt(v_hat_(t,i))`

So each coordinate gets a scale adapted to its recent gradient history.

### Why This Helps In Neural Networks

Neural networks often have:

- highly heterogeneous parameter sensitivities
- sparse gradients in some layers
- dense gradients in others
- changing gradient statistics during training

Adam is good in such settings because it quickly adapts to local scale without
requiring extensive manual tuning of per-layer learning rates.

### Adam And Noisy Mini-Batch Gradients

In mini-batch training:

`g_t = true gradient + noise`

Momentum smooths this noisy signal over time.

Second-moment tracking stabilizes update magnitudes despite stochastic
variability.

This makes Adam especially attractive early in training when gradients are noisy
and poorly scaled.

### Geometric Interpretation

Plain gradient descent uses Euclidean geometry with one global step size.

Adam changes the geometry of the update by effectively preconditioning the
gradient coordinatewise:

`update approx D_t^(-1) m_hat_t`

where `D_t` is a diagonal matrix built from `sqrt(v_hat_t) + epsilon`.

So Adam can be interpreted as doing gradient descent in a rescaled coordinate
system where steep and flat coordinates are partially normalized.

This is not full second-order optimization, but it is a cheap diagonal
approximation to curvature-related scaling.

### Why Adam Often Converges Faster Early

Adam frequently shows strong early optimization performance because:

- adaptive scaling avoids waiting for one global learning rate to fit all
  coordinates
- momentum quickly amplifies consistent directions
- initialization transients are corrected by bias correction

This often produces faster initial loss reduction than vanilla SGD.

### Why Adam Sometimes Generalizes Worse Than SGD

Although Adam often optimizes faster, it does not always generalize best.

Possible reasons:

- adaptive methods may settle into sharper minima in some settings
- SGD noise can have useful implicit regularization
- Adam may overfit small or noisy datasets more easily

This is why many training recipes use:

- Adam or AdamW for fast, stable training
- SGD with momentum in settings where final generalization is the main priority

This is not a universal rule, but it is an important practical pattern.

### Adam Versus RMSProp

RMSProp uses squared-gradient normalization but not the same bias-corrected
momentum form as Adam.

Very roughly:

- RMSProp handles scale
- Adam handles scale plus momentum

This makes Adam more complete and more widely used as a default optimizer.

### Adam Versus SGD With Momentum

SGD with momentum:

- one global learning rate
- moving average of gradients
- strong simplicity

Adam:

- moving average of gradients
- moving average of squared gradients
- coordinatewise adaptive steps

So Adam is usually easier to get working quickly, especially in modern deep
architectures with uneven gradient scales.

### The Role Of `beta_1`, `beta_2`, And `epsilon`

#### `beta_1`

Controls momentum memory.

- larger `beta_1` -> smoother but slower-to-react first-moment estimate
- smaller `beta_1` -> more reactive but noisier direction estimate

#### `beta_2`

Controls second-moment memory.

- larger `beta_2` -> more stable scaling estimate
- smaller `beta_2` -> more reactive but noisier adaptation

#### `epsilon`

Provides numerical stability and prevents division by zero.

It also affects effective step size when `v_hat_t` is very small.

### AdamW And Weight Decay

In modern practice, AdamW is often preferred over naive Adam with L2 penalty
mixed into the gradient.

Why:

- decoupled weight decay behaves more cleanly
- regularization effect is separated from adaptive gradient scaling

This matters especially in transformers and large-scale deep learning systems.

### Failure Modes Of Adam

Adam is not perfect.

Common issues:

- can converge to solutions with weaker generalization than SGD
- can be sensitive to learning rate despite adaptivity
- can behave poorly if weight decay is applied incorrectly
- may produce unstable late-stage dynamics if not scheduled properly

So "Adam works" does not mean "Adam always wins." It means Adam is a highly
effective default because it solves several practical optimization problems at
once.

### Why Adam Makes Sense Mathematically

Adam is mathematically sensible because it approximates a normalized descent
method:

1. use first moments to estimate stable direction
2. use second moments to estimate local scale
3. correct initialization bias
4. take a scaled step

It is therefore a pragmatic compromise between:

- pure first-order descent
- momentum
- diagonal preconditioning

without paying the computational cost of full curvature methods.

### When Adam Is Especially Useful

- deep networks with many layers
- transformers and language models
- sparse or uneven gradients
- fast prototyping
- settings where stable training matters more than hand-tuning SGD

### Final Synthesis

Adam works because it combines directional smoothing and adaptive scaling.
It remembers where gradients have been pointing and how large they have been,
then uses both pieces of information to compute a more stable update than plain
gradient descent.

Its success comes from matching the realities of neural-network optimization:

- noisy mini-batch gradients
- uneven parameter scales
- ill-conditioned objectives
- large parameter spaces

That is why Adam is one of the standard optimizers in modern deep learning.

## Further Reading

- `Canonical paper`: Kingma and Ba, "Adam: A Method for Stochastic Optimization" (2014), and Reddi et al., "On the Convergence of Adam and Beyond" (2018).
- `Best intuition resource`: Distill, *Why Momentum Really Works*, as a foundation for understanding Adam's momentum component.
- `Best practical code resource`: `karpathy/nanoGPT` on GitHub and the PyTorch AdamW documentation.
