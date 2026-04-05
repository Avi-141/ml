# Pretraining, Supervised Pretraining, and Fine-Tuning

## Intro And Concepts

Modern deep learning systems are rarely best understood as "a model trained on a
task." They are better understood as the result of a sequence of training
phases, where each phase shapes the hypothesis space and optimization landscape
for the next.

The canonical stages are:

1. pretraining
2. supervised pretraining, when labels are available and useful at scale
3. fine-tuning for a specific downstream objective

These are not merely implementation details. They are the mechanism by which
large models acquire broad structure and then specialize efficiently.

### What Pretraining Means

Pretraining is the first major learning phase, typically performed on large and
diverse data to learn reusable representations or predictive structure.

The goal is not to solve the final downstream task directly. The goal is to
learn a parameterization that makes later tasks easier.

Formally, pretraining optimizes a source objective:

`theta^* = argmin_theta L_pretrain(theta)`

and downstream adaptation begins from `theta^*` rather than random
initialization.

### Why Pretraining Exists

Training a large model from scratch on a narrow target task is statistically and
computationally inefficient because the model must simultaneously learn:

- input structure
- feature hierarchies
- invariances
- the task boundary or generation rule

Pretraining decouples these.

```text
without pretraining:
raw data -> learn structure + learn task -> target model

with pretraining:
raw large-scale data -> learn structure
target data -> specialize learned structure
```

### Self-Supervised vs Supervised Pretraining

Pretraining is a broad term. It includes several regimes.

#### Self-Supervised Pretraining

The model learns from the internal structure of raw data without external human
labels.

Examples:

- autoregressive next-token prediction
- masked-token prediction
- masked-patch prediction
- contrastive learning

This is dominant in modern language models because unlabeled data is abundant.

#### Supervised Pretraining

The model is pretrained on a large labeled dataset before being adapted to a
smaller or more specialized task.

Classic example:

- pretrain a vision model on ImageNet
- fine-tune on retinal disease classification

The source labels are not the final labels, but they still force the network to
learn transferable structure.

### What Fine-Tuning Means

Fine-tuning is the target adaptation phase:

`theta_t^* = argmin_theta L_target(theta ; theta^(0) = theta_pretrained)`

The model is no longer learning generic structure from scratch. It is adjusting
already-learned structure toward the target distribution, target label
semantics, or target behavior.

### The Three-Stage Picture

```text
Stage 1: broad pretraining
data with scale and diversity
        |
        v
learn general representations / predictive structure
        |
        v
Stage 2: supervised pretraining or intermediate adaptation
large labeled or more task-shaped data
        |
        v
learn stronger task-aware abstractions
        |
        v
Stage 3: fine-tuning
small or domain-specific target data
        |
        v
specialized downstream model
```

Not every project uses all three stages, but the distinction is conceptually
important.

### Intuition: What Changes Across The Stages

Pretraining tends to teach:

- structure of the input distribution
- invariances
- broad semantic regularities
- reusable internal features

Supervised pretraining tends to teach:

- task-shaped features
- better discriminative boundaries
- alignment with label semantics

Fine-tuning tends to teach:

- domain-specific calibration
- label mapping for the final task
- output behavior specific to the deployment objective

### Visualization: Representation And Head

```text
input -> backbone / representation -> task head -> prediction
```

Across training phases:

```text
pretraining:
input -> learn strong backbone --------> generic predictive skill

supervised pretraining:
input -> refine backbone + task family -> stronger discriminative skill

fine-tuning:
input -> adapt backbone/head ----------> narrow target behavior
```

### Why Fine-Tuning Is Not The Same As Pretraining

Both phases may use gradient descent, but their purposes differ:

- pretraining maximizes breadth
- fine-tuning maximizes specificity

Pretraining wants broad transferability.
Fine-tuning wants low target risk.

Those goals are related but not identical, and training recipes should reflect
that difference.

## Deep Dive

### Objective Functions

Different pretraining paradigms correspond to different losses.

#### Autoregressive Language Modeling

For a token sequence `x_1, ..., x_T`, decoder-only models are pretrained with:

`L_AR(theta) = - sum_t log p_theta(x_t | x_<t)`

This objective teaches the model to compress a large amount of statistical
structure about language into its parameters.

#### Masked Language Modeling

Encoder-style pretraining uses:

`L_MLM(theta) = - sum_(t in M) log p_theta(x_t | x_notin M)`

where `M` is a set of masked positions.

This emphasizes bidirectional contextual inference rather than left-to-right
generation.

#### Supervised Pretraining

For labeled pairs `(x_i, y_i)`, supervised pretraining often uses standard
cross-entropy:

`L_sup(theta) = - (1/n) sum_i sum_c 1[y_i=c] log p_theta(c | x_i)`

What makes it "pretraining" is not the loss itself, but its role as a source
stage for later transfer.

#### Fine-Tuning Objective

Fine-tuning usually minimizes a target loss:

`L_target(theta) = (1/n_t) sum_i L(f_theta(x_i^t), y_i^t)`

but with pretrained initialization:

`theta^(0) = theta_pretrained`

This makes fine-tuning a constrained or biased form of optimization, not a fresh
learning problem.

### Why Self-Supervised Pretraining Scales So Well

Self-supervised objectives scale because they exploit raw data directly. In
language this is especially powerful because the next token or missing token
contains information about:

- syntax
- semantics
- discourse
- world regularities
- stylistic conventions

Predicting those tokens forces the model to internalize latent structure that is
later reusable for many downstream tasks.

This is the critical reason modern LLMs emerge from pretraining rather than from
task-specific supervised learning alone.

### Supervised Pretraining As Representation Shaping

Supervised pretraining often produces stronger task-family features than pure
self-supervision when the label space is broad and meaningful.

Example:

- ImageNet pretraining forces a model to separate many object classes
- this encourages part detectors, texture sensitivity, and object-level
  abstractions

Even when the downstream task differs, these features may still be more useful
than random initialization.

### Intermediate Or Continued Pretraining

There is an important middle case between broad pretraining and fine-tuning:

- domain-adaptive pretraining
- continued pretraining
- intermediate-task training

Example:

```text
general web-text pretraining
        ->
continued pretraining on legal or biomedical corpora
        ->
fine-tuning on legal QA or medical classification
```

This stage reduces domain mismatch before the narrow downstream objective is
introduced.

### Why Fine-Tuning Needs Careful Optimization

A pretrained model already occupies a useful basin in parameter space. Fine-
tuning should adapt that basin, not destroy it.

This is why fine-tuning often uses:

- smaller learning rates
- shorter schedules
- selective layer unfreezing
- weight decay and dropout
- early stopping

Conceptually, fine-tuning is often closer to local surgery than to fresh
construction.

### A Regularized View Of Fine-Tuning

A useful mathematical interpretation is:

`min_theta L_target(theta) + lambda Omega(theta, theta_pretrained)`

where `Omega` penalizes harmful deviation from the pretrained solution.

For example:

`Omega(theta, theta_pretrained) = ||theta - theta_pretrained||^2`

This captures the intuition that not all movement in parameter space is equally
desirable.

### Catastrophic Forgetting

The main optimization risk in fine-tuning is catastrophic forgetting: the model
adapts to the target objective by overwriting useful source knowledge.

Mechanistically, this can happen when:

- the target dataset is small or narrow
- gradients are high variance
- the learning rate is too large
- fine-tuning updates all layers aggressively

In language models this may show up as:

- improved narrow task accuracy
- degraded general instruction following
- reduced robustness outside the fine-tuning domain

### Layerwise Interpretation

Although the exact picture depends on architecture and scale, a common heuristic
is:

- lower layers encode more general local structure
- middle layers encode compositional abstractions
- upper layers are more task- or objective-specific

This motivates training recipes such as:

- freeze lower layers
- adapt upper layers first
- gradually unfreeze deeper layers

The heuristic is not universal, but it is operationally useful.

### Visualization: Layerwise Adaptation

```text
pretrained model

[layer 1]  general
[layer 2]  general
[layer 3]  mixed
[layer 4]  mixed
[layer 5]  source-specific
[head]     source-specific

fine-tuning options

Option A: freeze layers 1-4, retrain 5 + head
Option B: retrain all layers with small LR
Option C: adapters / LoRA in selected layers
```

### Fine-Tuning Regimes

#### Linear Probe

Freeze the backbone and train a lightweight head. This measures whether the
representation is already useful.

#### Full Fine-Tuning

Update all model weights. This usually gives the best adaptation when enough
data and careful optimization are available.

#### Parameter-Efficient Fine-Tuning

Adapt only small trainable components. In transformers this includes:

- adapters
- prompt tuning
- prefix tuning
- LoRA

These methods reduce memory and storage cost while preserving most pretrained
weights.

### Instruction Tuning And Supervised Fine-Tuning

In LLM workflows, supervised fine-tuning often means optimizing on
instruction-response pairs:

`L_SFT(theta) = - sum_t log p_theta(y_t | x, y_<t)`

where `x` is the instruction and `y` is the desired response.

This is not pretraining in the broad corpus sense. It is a specialization stage
that reshapes behavior toward helpful task execution.

### Why Pretraining + Fine-Tuning Beats Pure Supervision

If a model is trained only on a small supervised task, capacity is spent on
basic representation learning. If that same model is pretrained first, the
supervised stage can allocate more of its limited data budget to:

- calibration
- task boundary learning
- output format learning
- domain-specific adaptation

This is why the same number of downstream labels often produces much stronger
results when preceded by large-scale pretraining.

### Failure Modes

- pretraining objective learns the wrong inductive bias
- source data quality is poor
- domain mismatch remains too large
- supervised pretraining over-specializes to the source labels
- fine-tuning destroys broad capabilities
- evaluation ignores out-of-domain degradation

### Final Synthesis

Pretraining, supervised pretraining, and fine-tuning are best viewed as staged
control over what the model learns and when:

- pretraining builds broad structure
- supervised pretraining can sharpen task-family structure
- fine-tuning specializes for deployment

The strength of the modern deep learning pipeline comes from this staged
allocation of learning pressure, not from any one phase in isolation.
