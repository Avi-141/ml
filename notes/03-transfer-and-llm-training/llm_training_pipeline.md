# LLM Training Pipeline

## Intro And Concepts

A modern large language model is not produced by a single training run. It is
usually the result of several distinct stages, each solving a different problem.

A useful high-level pipeline is:

```text
data collection and filtering
        ->
tokenization
        ->
foundation-model pretraining
        ->
continued pretraining or domain adaptation
        ->
supervised fine-tuning
        ->
preference optimization / alignment
        ->
evaluation, safety hardening, and deployment
```

The important point is that these stages are not redundant. They shape different
aspects of the final model:

- pretraining gives broad language competence
- supervised fine-tuning gives task-following behavior
- alignment methods shape preference and interaction style
- evaluation and serving constraints determine whether the model is actually
  usable

### Why The Pipeline Is Staged

No single dataset and no single loss function is sufficient for all desired
behavior.

Pretraining is good at learning:

- syntax
- semantics
- world regularities
- long-range token prediction structure

But pretraining alone is not enough for:

- clean instruction following
- conversational helpfulness
- stable refusal behavior
- preference-sensitive responses

So post-training exists to reshape behavior after the base model has learned
general language structure.

### Stage 1: Data And Tokenization

Before optimization begins, the system must decide:

- what raw text, code, dialogue, and documents to include
- how to filter low-quality or unsafe data
- how to deduplicate repeated content
- how to tokenize the text into discrete units

Tokenization matters because the model predicts tokens, not characters or words
in the naive sense.

If the tokenizer is poor:

- sequence lengths grow unnecessarily
- common patterns fragment badly
- compute is wasted

So the tokenizer is part of the model design, not just a preprocessing detail.

### Stage 2: Foundation Pretraining

The base model is typically pretrained autoregressively:

`L_pretrain(theta) = - sum_t log p_theta(x_t | x_<t)`

This objective teaches the model to predict the next token over massive corpora.

The result is a foundation model with:

- broad linguistic structure
- factual regularities
- latent reasoning patterns
- code and document priors, depending on the corpus

### Stage 3: Continued Pretraining

Sometimes a general base model is further pretrained on narrower data.

Examples:

- legal text
- biomedical corpora
- enterprise documentation
- code-heavy distributions

This is often called:

- continued pretraining
- domain-adaptive pretraining

Its purpose is to reduce distribution mismatch before supervised specialization.

### Stage 4: Supervised Fine-Tuning

At this stage, the model is trained on curated input-output examples.

For instruction tuning, the loss is usually:

`L_SFT(theta) = - sum_t log p_theta(y_t | x, y_<t)`

where:

- `x` is the prompt or instruction
- `y` is the target response

This stage teaches:

- instruction following
- task formatting
- response style
- answer structure

It turns a raw next-token predictor into a more useful assistant-like model.

### Stage 5: Preference Optimization And Alignment

Even after supervised fine-tuning, multiple responses may be:

- correct
- grammatical
- but very different in usefulness or safety

So modern pipelines often include preference learning.

The rough idea is:

1. collect comparisons or preference signals
2. train a reward or preference model, or optimize directly from preferences
3. adjust the policy model toward preferred outputs

This stage aims to improve:

- helpfulness
- harmlessness
- stylistic consistency
- user alignment

### Stage 6: Evaluation And Deployment

A model is not ready just because training loss is low.

It must be tested for:

- perplexity or token prediction quality
- task performance
- robustness
- calibration
- safety behavior
- latency and serving cost

This is where practical ML engineering meets model science.

## Deep Dive

### Pretraining As Maximum Likelihood Estimation

Autoregressive pretraining minimizes:

`L(theta) = - sum_t log p_theta(x_t | x_<t)`

Equivalently, it maximizes the likelihood of the observed token sequences under
the model.

This objective is deceptively simple. Because natural language contains:

- syntax
- semantics
- discourse
- factual patterns
- latent task structure

next-token prediction forces the model to internalize a remarkably broad set of
regularities.

This is why pretraining is the foundation of the entire pipeline.

### Scaling Laws And Why Pretraining Became Central

LLM development is strongly shaped by empirical scaling behavior:

- more data
- more parameters
- more compute

often produce smoother improvements than many older training paradigms.

This made large-scale pretraining economically and scientifically worthwhile.

Once a capable base model exists, many downstream behaviors can be added more
cheaply than training a new model from scratch.

### Tokenization And Model Efficiency

The tokenizer determines the atomic prediction units.

A bad tokenizer increases:

- effective context length
- fragmentation of common patterns
- difficulty of learning reusable substructure

A good tokenizer balances:

- vocabulary size
- compression efficiency
- cross-domain reuse
- computational practicality

This affects both pretraining efficiency and post-training quality.

### Why Continued Pretraining Exists

General web-scale pretraining gives breadth, but specialized deployment domains
often need more concentrated priors.

Example:

```text
general pretraining
        ->
continued pretraining on medical text
        ->
medical question answering or summarization fine-tuning
```

This stage is especially useful when:

- terminology is highly specialized
- style differs from the general web
- factual distribution is domain-specific

### Supervised Fine-Tuning As Behavioral Reshaping

A base model trained only for next-token prediction can imitate many patterns,
but it has no dedicated training objective for:

- following explicit instructions
- being concise when needed
- answering in a preferred format
- refusing certain requests reliably

SFT adds this behavioral layer by teaching the model:

- what tasks look like
- what a good answer format looks like
- how prompt-response interaction should behave

This is a key conceptual shift from competence to usability.

### Preference Optimization

There are multiple modern approaches here, but the conceptual structure is:

#### Reward-Model Style

1. collect ranked or preferred responses
2. train a model to score responses
3. optimize the language model toward higher-scoring outputs

#### Direct Preference Optimization Style

Skip explicit reward-model deployment and optimize directly from preference pairs
using a preference objective.

The point in both cases is the same:

- language modeling tells the model what is plausible
- preference optimization tells the model what is preferred

### Why Preference Optimization Is Needed

Likelihood alone does not fully capture quality.

A response can be:

- likely under internet text
- yet unhelpful, unsafe, or poorly structured

So alignment stages exist to move from:

```text
plausible continuation
```

toward:

```text
useful and preferred assistant behavior
```

### Data Mixture Matters At Every Stage

One of the deepest practical truths in LLM training is that the data mixture is
often as important as the optimizer or architecture.

At each stage, the model inherits the properties of the data:

- pretraining data shapes broad world priors
- continued pretraining data shapes domain priors
- SFT data shapes interaction style
- preference data shapes quality judgments

So training is not just about objective functions. It is about which
distribution the model is repeatedly pushed toward.

### Optimization And Infrastructure Reality

LLM training requires more than theoretical loss definitions.

Practical issues include:

- distributed training
- mixed precision
- checkpointing
- optimizer state sharding
- gradient accumulation
- stable learning-rate schedules
- efficient data loading

This is why LLM development is as much systems engineering as model design.

### Evaluation Across The Pipeline

Different stages require different evaluation criteria.

#### Pretraining Evaluation

- perplexity
- held-out token loss
- scaling behavior

#### SFT Evaluation

- task completion quality
- instruction following
- formatting reliability

#### Preference / Alignment Evaluation

- human preference win rate
- refusal quality
- safety robustness
- policy consistency

No single metric captures the whole pipeline.

### Serving Constraints Feed Back Into Training

Real deployment constraints matter:

- latency
- throughput
- memory footprint
- context length
- inference cost

These often influence:

- model size
- tokenizer choice
- quantization strategy
- whether to use retrieval augmentation
- whether to use adapters rather than full fine-tuning

So the training pipeline is never isolated from deployment.

### Fine-Tuning, Adapters, And Specialization

Modern LLM applications often do not retrain the entire model.

Instead they may use:

- full fine-tuning
- LoRA-style low-rank adaptation
- adapters
- prompt tuning

This makes specialization cheaper and easier to maintain across many tasks or
customers.

### End-To-End View

A useful summary is:

```text
pretraining        -> learns language competence
continued training -> sharpens domain priors
SFT                -> teaches task-following behavior
preference tuning  -> teaches preference-sensitive behavior
evaluation         -> checks whether the system is usable and safe
deployment         -> constrains what is practical
```

Each stage solves a different problem. That is why modern LLM development is a
pipeline rather than a single optimization job.

### Final Synthesis

The LLM training pipeline works because it separates broad knowledge acquisition
from behavior shaping.

Pretraining gives the model general predictive competence.
Continued pretraining adapts it to the right domain.
Supervised fine-tuning teaches task and response structure.
Preference optimization teaches which valid outputs are better than others.
Evaluation and deployment determine whether the result is actually useful.

That staged decomposition is the key to understanding modern large language model
development.
