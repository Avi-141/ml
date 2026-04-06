# Transfer Learning

## Intro And Concepts

Transfer learning is the study and practice of reusing information learned from
one learning problem in order to improve performance, sample efficiency, or
optimization on another learning problem.

The broad setting is:

- source domain/task: where the model first learns
- target domain/task: where the model is eventually used

The central question is not just "can weights be reused?" but:

> Under what conditions does knowledge learned on the source problem reduce the
> target problem's statistical or optimization difficulty?

### Formal Setup

Let:

- `D_s = (X_s, P_s(X))` be the source domain
- `T_s = (Y_s, P_s(Y|X))` be the source task
- `D_t = (X_t, P_t(X))` be the target domain
- `T_t = (Y_t, P_t(Y|X))` be the target task

Transfer learning studies what happens when at least one of the following
changes:

- input distribution changes: `P_s(X) != P_t(X)`
- label mapping changes: `P_s(Y|X) != P_t(Y|X)`
- label space changes: `Y_s != Y_t`
- objective changes: classification, generation, ranking, retrieval, control

The target objective is still to minimize expected target risk:

`R_t(f) = E_(x,y)~P_t [ L(f(x), y) ]`

but now `f` is initialized, constrained, or structurally informed by learning on
the source task.

### What Is Actually Transferred

In modern systems, the transferred object is usually one or more of:

- parameters `theta`
- embeddings
- learned features `h_theta(x)`
- attention patterns
- optimizer state, sometimes
- adapter modules or low-rank updates
- retrieval memory or external tool behavior

The most common operational form is parameter transfer:

`theta_t^(0) <- theta_s^*`

instead of random initialization:

`theta_t^(0) <- Init()`

This is already a strong prior over the hypothesis class.

### Why Transfer Learning Works

Transfer learning works when the source task forces the model to learn
representations that preserve structure useful for the target task.

Examples:

- vision pretraining learns edges, textures, part-whole relations, invariances
- language pretraining learns syntax, semantics, discourse, token statistics,
  long-range dependencies
- speech pretraining learns phonetic and temporal structure

This can be viewed in three complementary ways.

#### Feature Reuse

Early and middle layers often learn reusable structure rather than task-specific
surface details.

In vision this may include:

- edges
- corners
- textures
- part-whole patterns

In language this may include:

- token statistics
- syntax
- semantic relations
- discourse-level dependencies

This is why transfer often works especially well when source and target tasks
share latent structure even if they do not share identical labels.

#### Inductive Bias Transfer

Pretraining does not only provide useful weights. It also provides a useful
inductive bias:

- what kinds of features are likely to matter
- what kinds of functions are easy to express
- what kinds of solutions optimization will prefer

So transfer learning works partly because the source task teaches the model what
to pay attention to before the target task even begins.

#### Optimization Landscape Shaping

A pretrained model starts optimization in a parameter region already encoding
useful abstractions. That changes the target optimization problem from:

```text
discover useful features + discover task boundary
```

to:

```text
adapt useful features + refine task boundary
```

This is often the difference between difficult global search and relatively local
adaptation.

#### Sample Complexity Reduction

A target task with limited labels benefits because those labels no longer need to
teach the model basic structure from zero.

Instead, target labels are spent more efficiently on:

- calibration
- specialization
- task-specific separation
- output behavior

This is why transfer learning often helps most when the target dataset is small
or expensive to label.

#### Why It Works Especially Well In CNNs And Transformers

CNNs transfer well because convolutional hierarchies naturally learn reusable
visual primitives:

- early layers detect edges
- intermediate layers detect motifs and parts
- later layers compose those into object-level structure

Transformers transfer well because self-attention builds broad contextual
representations that are useful across many objectives:

- classification
- retrieval
- generation
- instruction following
- multimodal grounding

So in both CNNs and transformers, transfer works because the architecture learns
reusable intermediate structure, not just a narrow final classifier.

#### 1. Representation View

The source model learns a feature map:

`h_theta : X -> Z`

such that the target task is simpler in `Z` than in raw input space `X`.

The target head may then only need to learn:

`g_phi(h_theta(x))`

instead of discovering both the representation and the task boundary from
scratch.

#### 2. Optimization View

Pretraining places the model in a parameter region already encoding useful
structure. Fine-tuning then performs local adaptation rather than global search.

This usually yields:

- faster convergence
- lower variance across runs
- better data efficiency
- better local minima than training from scratch

#### 3. Bayesian / Prior View

A pretrained model acts like a learned prior over functions. Fine-tuning updates
that prior using target evidence. This is one reason transfer learning is
especially valuable when target data is limited.

### Positive Transfer And Negative Transfer

Transfer is not automatically beneficial.

#### Positive Transfer

Source learning reduces target error, target sample complexity, or target
optimization difficulty.

#### Negative Transfer

Source learning imposes the wrong inductive bias and hurts target performance.

This happens when:

- source and target features are misaligned
- label semantics differ in a harmful way
- domain shift is too large
- fine-tuning is too constrained or too aggressive

### Visualization: Why Transfer Helps

```text
Training from scratch
random init -> search for features + search for decision rule -> target solution

Transfer learning
pretrained features -> adapt task head / adapt full model -> target solution
```

The target problem is easier because the model no longer has to discover every
useful feature from zero.

### Visualization: Feature Reuse Across Layers

```text
input
  |
  v
[early layers]   generic patterns
[middle layers]  compositional abstractions
[late layers]    source-task specialization
  |
  v
source head
```

Typical transfer workflow:

```text
input
  |
  v
[reuse early + middle layers]
  |
  +--> freeze, probe, or partially adapt
  |
  v
new target head
```

This is why transfer often replaces the top layers first and adapts deeper
layers only if target data and compute allow it.

### Main Operational Modes

#### Feature Extraction

Freeze most of the pretrained backbone and train a small head on top.

Advantages:

- cheap
- stable
- good when target data is small

Limitations:

- target task can only use the existing representation
- domain shift may remain uncorrected

#### Fine-Tuning

Update some or all pretrained parameters on the target task.

Advantages:

- better adaptation to target distribution
- better end-task performance when enough data exists

Limitations:

- risk of overfitting
- risk of catastrophic forgetting
- more sensitive to learning rates and training recipe

#### Parameter-Efficient Transfer

Instead of updating all weights, update only small modules or low-rank changes:

- adapters
- prompt tuning
- prefix tuning
- LoRA-style updates

This is especially useful for large transformers and LLMs.

## Deep Dive

### A More Mathematical View

Suppose a model is decomposed as:

`f_(theta,phi)(x) = g_phi(h_theta(x))`

where:

- `h_theta` is the representation or backbone
- `g_phi` is the task-specific head

Training from scratch on the target task solves:

`min_(theta,phi) (1/n_t) sum_i L(g_phi(h_theta(x_i^t)), y_i^t)`

Transfer learning instead solves one of the following.

#### Linear Probe / Frozen Backbone

Keep `theta = theta_s^*` fixed and solve only:

`min_phi (1/n_t) sum_i L(g_phi(h_(theta_s^*)(x_i^t)), y_i^t)`

This directly tests whether the learned representation is linearly or simply
separable for the target task.

#### Full Fine-Tuning

Initialize with pretrained weights and optimize:

`min_(theta,phi) (1/n_t) sum_i L(g_phi(h_theta(x_i^t)), y_i^t)`

subject to initialization:

`theta^(0) = theta_s^*`

Even though the objective resembles ordinary training, the optimization path is
completely different because the starting point is informed.

#### Regularized Fine-Tuning

A useful conceptual form is:

`min_(theta,phi) L_target(theta,phi) + lambda ||theta - theta_s^*||^2`

This says: adapt to the target task, but do not move too far from pretrained
knowledge unless the target evidence justifies it.

This is one way to understand:

- small learning rates
- layer freezing
- trust-region style adaptation
- anti-forgetting regularization

### Why Pretrained Representations Generalize

A strong source task forces the model to learn invariances and compositional
structure.

In vision:

- translation invariance
- texture and shape sensitivity
- part-whole hierarchies

In language:

- lexical co-occurrence structure
- syntax
- coreference and discourse patterns
- semantic regularities

If the target task depends on those same structures, then the target classifier
or decoder is effectively solving a simpler problem.

This is often described informally as:

```text
raw input space  -> hard geometry
learned feature space -> easier geometry
```

A good pretrained representation "untangles" the target classes.

### Visualization: Untangling The Target Problem

```text
Raw space

class A:  scattered, curved, overlapping
class B:  scattered, curved, overlapping

Learned feature space h_theta(x)

class A:  compact cluster
class B:  compact cluster
```

After transfer, a linear separator in feature space may be enough even when the
raw-space boundary is highly non-linear.

### Transfer As Sample-Efficiency Improvement

If target data is scarce, a randomly initialized model has to spend examples on:

- discovering useful features
- calibrating those features
- learning the target decision rule

A pretrained model already solves much of the first two steps. Target labels are
therefore used more efficiently.

This is why transfer learning often produces the biggest gains when:

- target labels are limited
- target model is large
- source pretraining is broad and high quality

### Domain Shift Matters

Transfer quality depends strongly on the relation between source and target.

We usually distinguish:

- covariate shift: `P_s(X) != P_t(X)` but the task relation is similar
- label shift: `P_s(Y) != P_t(Y)`
- concept shift: `P_s(Y|X) != P_t(Y|X)`

Transfer is easiest under covariate shift and hardest under major concept shift.

Example:

- source: generic English web text
- target: biomedical question answering

The syntax and much of the language machinery transfers well, but domain
vocabulary and factual priors may require continued pretraining or task-specific
fine-tuning.

### When Freezing Works And When It Fails

Freezing works well when:

- the target task is close to the source representation
- target data is small
- compute is limited

Freezing fails when:

- the target domain differs substantially
- the representation is missing target-specific abstractions
- the required output behavior differs materially from the source task

This is why linear probing is informative but not sufficient. It measures
representation usefulness, not full adaptation capacity.

### Catastrophic Forgetting

Fine-tuning is not free. Updating all layers can overwrite useful source
knowledge.

Mechanistically this often appears as:

- rapid drift in higher layers
- collapse of previously useful features
- target over-specialization

Symptoms:

- excellent narrow target performance
- degraded performance on previously transferable capabilities

Mitigations:

- lower learning rate
- gradual unfreezing
- regularization toward source weights
- replay or mixed-domain training
- parameter-efficient adaptation

### Transfer In Large Models

Modern transfer learning is dominated by foundation models.

A rough picture:

```text
massive pretraining
      |
      v
general-purpose model
      |
      +--> frozen feature extractor
      +--> full fine-tuning
      +--> adapters / LoRA
      +--> instruction tuning
      +--> domain adaptation
```

The bigger the pretrained model, the more transfer becomes the default training
regime rather than a special trick.

### A Useful Geometric Intuition

Think of pretraining as shaping the loss landscape before the target task is
ever seen.

Without transfer:

- the optimizer starts in an unstructured region
- many directions correspond to bad features

With transfer:

- the optimizer starts near a basin representing useful abstractions
- fine-tuning mostly chooses how to specialize those abstractions

This does not eliminate optimization difficulty, but it changes the search
problem from "discover everything" to "adapt something already meaningful."

### Practical Failure Modes

- source-target mismatch causes negative transfer
- target data is too small for full fine-tuning
- target labels are noisy, corrupting pretrained features
- large learning rates destroy pretrained structure
- evaluation confuses memorization with successful transfer

### Final Synthesis

Transfer learning is best understood as the interaction of:

- representation reuse
- informed initialization
- inductive bias transfer
- target adaptation under distribution shift

It is not merely "reusing a model." It is a statistical and optimization
strategy for reducing the effective difficulty of the target problem by importing
structure learned elsewhere.

## Further Reading

- `Canonical paper`: Pan and Yang, "A Survey on Transfer Learning" (2010).
- `Best intuition resource`: Andrew Ng's transfer learning lectures in the DeepLearning.AI CNN sequence and the CS231n transfer learning notes.
- `Best practical code resource`: PyTorch's transfer learning tutorial and the Hugging Face Transformers fine-tuning examples.
