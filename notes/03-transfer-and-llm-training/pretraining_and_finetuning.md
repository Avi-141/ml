# Pretraining, Supervised Pretraining, and Fine-Tuning

## Intro And Concepts

Modern deep learning systems are rarely best understood as "a model trained on a
task." They are better understood as the result of a sequence of training
phases, where each phase shapes the hypothesis space and optimization landscape
for the next.

A common sequence is:

1. pretraining
2. supervised pretraining or task-oriented adaptation
3. fine-tuning

These stages let a model first learn broad knowledge and then gradually become
more specialized.

### What Pretraining Means

Pretraining is the process of training a model on a large dataset so that it
learns general-purpose representations before it is adapted to a specific task.

The key idea:

- first learn broad patterns
- then specialize later

Pretraining is usually done on:

- very large datasets
- broad domains
- tasks where labels may be cheap or not needed

### Intuition For Pretraining

Suppose you want a model to answer medical questions.

It helps if the model already knows:

- grammar
- vocabulary
- sentence structure
- common world knowledge
- basic reasoning patterns

Pretraining gives the model this broad foundation.

Without pretraining, the model would have to learn all of that from the smaller
medical dataset alone.

### Why Pretraining Matters

Pretraining helps because it teaches the model:

- general representations
- reusable patterns
- useful structure in the data

This often leads to:

- faster convergence
- better performance
- less labeled data needed later

### Unsupervised Or Self-Supervised Pretraining

In modern deep learning, pretraining is often self-supervised rather than fully
supervised.

That means the model learns from the raw structure of the data itself.

Examples:

- predict the next word in a sentence
- fill in missing words
- predict missing image patches
- learn whether two augmented views come from the same image

This is powerful because unlabeled data is abundant.

### Example: Language Model Pretraining

A language model may be pretrained by predicting the next token:

```text
"The cat sat on the ..." -> predict "mat"
```

By doing this across massive text corpora, the model learns:

- syntax
- semantics
- discourse structure
- factual patterns
- long-range context

This is how many modern transformers and large language models begin.

### Example: Vision Pretraining

A vision model may be pretrained by:

- classifying millions of images
- reconstructing missing image patches
- contrasting different views of the same image

This helps it learn:

- edges
- textures
- shapes
- object parts
- spatial structure

## 2. What Is Supervised Pretraining

Supervised pretraining means pretraining the model on a large labeled dataset
before adapting it to a smaller target task.

So unlike self-supervised pretraining, this stage uses explicit labels.

Example:

- train a vision model on ImageNet first
- then adapt it to a medical imaging task

Here:

- ImageNet training is supervised pretraining
- medical adaptation is later fine-tuning

### Intuition For Supervised Pretraining

The model first learns a broad but labeled skill.

For instance:

- distinguishing many object categories
- classifying general document types
- identifying speech categories

Even if the final task is different, those learned features can still be useful.

### Difference Between Pretraining And Supervised Pretraining

The word "pretraining" is broad.

It can include:

- self-supervised pretraining
- unsupervised pretraining
- supervised pretraining

So:

- **pretraining** = any broad first-stage training before the final task
- **supervised pretraining** = pretraining specifically done with labeled data

## 3. What Is Fine-Tuning

Fine-tuning is the process of taking a pretrained model and continuing training
it on a more specific target task or domain.

The model already knows something useful.
Fine-tuning teaches it how to apply that knowledge to the exact problem you
care about.

### Intuition For Fine-Tuning

A pretrained model is like someone with a broad education.
Fine-tuning is like job-specific training.

Examples:

- a general language model becomes a legal assistant
- a general image model becomes a tumor classifier
- a general speech model becomes a customer-service transcription model

### What Changes During Fine-Tuning

During fine-tuning, you might:

- replace the output head
- train only the new head
- unfreeze part of the backbone
- fine-tune the full network

The exact strategy depends on:

- dataset size
- domain similarity
- compute budget
- risk of overfitting

### Full Workflow

A full pipeline might look like this:

```text
large raw data -> pretraining -> general model
large labeled data -> supervised pretraining -> stronger task-aware model
small target dataset -> fine-tuning -> specialized model
```

Not every project uses all three stages, but this staged picture is very
helpful conceptually.

### Example 1: Computer Vision

1. supervised pretraining on ImageNet
2. replace classifier head
3. fine-tune on skin lesion images

Why it works:

- the pretrained model already knows visual features
- the target task only needs specialization

### Example 2: NLP

1. self-supervised pretraining on massive text
2. optionally further train on labeled instruction or task datasets
3. fine-tune for summarization, sentiment, QA, or domain use

This is a common pattern in modern language models.

### Example 3: Domain Adaptation

1. pretrain on general web text
2. continue pretraining on biomedical papers
3. fine-tune on medical question answering

This is useful when the target domain has special vocabulary and style.

### Parameter Transfer

The central technical idea is parameter reuse.

Instead of randomly initializing all weights, we initialize from a previously
trained model.

This gives the optimizer a much better starting point.

Random initialization:

- starts with no useful knowledge

Pretrained initialization:

- starts with useful representations already present

### Why Fine-Tuning Usually Needs Smaller Learning Rates

When fine-tuning, the model already contains useful information.

If the learning rate is too large:

- the model may quickly overwrite useful pretrained features
- performance may become unstable

So fine-tuning often uses:

- smaller learning rates
- gradual unfreezing
- regularization

### Catastrophic Forgetting

A major issue in fine-tuning is catastrophic forgetting.

This means:

- the model adapts to the new task
- but forgets useful knowledge from pretraining

This can happen if:

- the fine-tuning dataset is small
- the learning rate is too large
- the target task is narrow

### Feature Extraction vs Fine-Tuning

Sometimes you do not fine-tune the whole model.

#### Feature Extraction

- freeze the pretrained model
- use its hidden representations
- train only a lightweight task head

#### Fine-Tuning

- allow some or all pretrained weights to update
- get better adaptation when enough data is available

### Supervised Pretraining vs Fine-Tuning

These can look similar because both use labels, but their goals differ.

#### Supervised Pretraining

- done on a larger, more general labeled dataset
- goal: learn broad useful features

#### Fine-Tuning

- done on a smaller, more specific target dataset
- goal: specialize the model

### How This Relates To Transfer Learning

Transfer learning is the big idea.

Pretraining and fine-tuning are the common mechanism used to make transfer
learning happen in practice.

So:

- transfer learning = reuse learned knowledge
- pretraining = build the initial reusable knowledge
- fine-tuning = adapt it to the target task

### Real-World Examples

#### Image Models

- ResNet pretrained on ImageNet
- then fine-tuned for defect detection, medical diagnosis, or satellite imagery

#### Language Models

- transformer pretrained on web-scale text
- then fine-tuned for summarization, translation, classification, or chat

#### Speech Models

- acoustic model pretrained on large audio corpora
- then fine-tuned for a specific accent or domain

### A Short Comparison Table

| Stage | Main Goal | Data Size | Labels Needed? | Output |
|---|---|---:|---|---|
| Pretraining | Learn general representations | Very large | Often no | General model |
| Supervised Pretraining | Learn broad labeled features | Large | Yes | Stronger task-aware model |
| Fine-Tuning | Adapt to target task | Smaller | Usually yes | Specialized model |

### Summary

- Pretraining teaches a model broad reusable knowledge.
- Self-supervised pretraining learns from the structure of raw data.
- Supervised pretraining uses a large labeled dataset before the final task.
- Fine-tuning adapts the pretrained model to a specific target task.
- These stages are a major reason modern deep learning works so well at scale.

## Further Reading

- `Canonical papers`: Howard and Ruder, "Universal Language Model Fine-tuning for Text Classification" (ULMFiT, 2018); Devlin et al., "BERT" (2018); Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018).
- `Best intuition resource`: Chris McCormick's BERT fine-tuning and transformer posts.
- `Best practical code resource`: Hugging Face Transformers examples and course materials.
