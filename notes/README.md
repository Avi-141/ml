# ML Notes Roadmap

This folder is now organized into topic folders so the note library is easier to
navigate and study in a logical sequence.

Most core notes still follow the same internal structure:

1. `Intro And Concepts`
2. `Deep Dive`

The goal remains mechanism-level understanding:

- what the concept is
- why it exists
- how it works
- what the math is doing
- what geometry or intuition sits underneath it
- where it shows up in modern ML systems

## Folder Structure

```text
notes/
├── README.md
├── 01-foundations/
│   ├── README.md
│   ├── neural_networks_math_and_statistics.md
│   ├── matrices_eigenvalues_and_eigenvectors_for_ml.md
│   ├── pca_and_covariance_geometry.md
│   ├── gradients_loss_and_gradient_descent.md
│   ├── backpropagation_from_first_principles.md
│   ├── adam_optimizer.md
│   └── softmax.md
├── 02-sequence-models/
│   ├── README.md
│   ├── sequence_models_rnn_lstm_gru.md
│   ├── attention_and_self_attention.md
│   └── transformers_from_first_principles.md
├── 03-transfer-and-llm-training/
│   ├── README.md
│   ├── transfer_learning.md
│   ├── pretraining_and_finetuning.md
│   ├── where_transfer_learning_is_used.md
│   └── llm_training_pipeline.md
└── 04-kernel-methods/
    ├── README.md
    ├── kernel_functions.md
    └── kernel.txt
```

## Recommended Reading Order

If you want the cleanest end-to-end progression, read in this order:

1. `01-foundations/neural_networks_math_and_statistics.md`
2. `01-foundations/matrices_eigenvalues_and_eigenvectors_for_ml.md`
3. `01-foundations/pca_and_covariance_geometry.md`
4. `01-foundations/gradients_loss_and_gradient_descent.md`
5. `01-foundations/backpropagation_from_first_principles.md`
6. `01-foundations/adam_optimizer.md`
7. `01-foundations/softmax.md`
8. `02-sequence-models/sequence_models_rnn_lstm_gru.md`
9. `02-sequence-models/attention_and_self_attention.md`
10. `02-sequence-models/transformers_from_first_principles.md`
11. `03-transfer-and-llm-training/transfer_learning.md`
12. `03-transfer-and-llm-training/pretraining_and_finetuning.md`
13. `03-transfer-and-llm-training/where_transfer_learning_is_used.md`
14. `03-transfer-and-llm-training/llm_training_pipeline.md`
15. `04-kernel-methods/kernel_functions.md`
16. `04-kernel-methods/kernel.txt`

## Folder Map

### `01-foundations/`

Use this folder when your priority is:

- neural-network math
- linear algebra and geometry
- losses and gradients
- backpropagation
- optimization
- softmax and probabilistic outputs

Start here if you want the strongest mathematical base.

### `02-sequence-models/`

Use this folder when your priority is:

- sequence modeling
- RNNs, LSTMs, and GRUs
- attention and self-attention
- transformers

This folder explains the architectural progression from recurrence to
transformers.

### `03-transfer-and-llm-training/`

Use this folder when your priority is:

- transfer learning
- pretraining and fine-tuning
- where transfer learning appears
- modern LLM training and post-training

This folder is the bridge from core model architecture to modern foundation-
model practice.

### `04-kernel-methods/`

Use this folder when your priority is:

- kernels
- RKHS intuition
- classical non-linear ML
- SVM, SVR, KRR, and regularized kernel methods

This folder is the classical non-linear theory branch of the library.

## Suggested Study Tracks

### Track 0: Neural-Network Math And Optimization

1. `01-foundations/neural_networks_math_and_statistics.md`
2. `01-foundations/matrices_eigenvalues_and_eigenvectors_for_ml.md`
3. `01-foundations/pca_and_covariance_geometry.md`
4. `01-foundations/gradients_loss_and_gradient_descent.md`
5. `01-foundations/backpropagation_from_first_principles.md`
6. `01-foundations/adam_optimizer.md`
7. `01-foundations/softmax.md`

Best if your focus is:

- why neural networks are written with matrices
- why eigenvalues, covariance, and curvature matter
- why gradients and backpropagation work
- why Adam works
- how logits become probabilities

### Track 1: Sequence Models To Modern LLMs

1. `02-sequence-models/sequence_models_rnn_lstm_gru.md`
2. `02-sequence-models/attention_and_self_attention.md`
3. `02-sequence-models/transformers_from_first_principles.md`
4. `03-transfer-and-llm-training/transfer_learning.md`
5. `03-transfer-and-llm-training/pretraining_and_finetuning.md`
6. `03-transfer-and-llm-training/where_transfer_learning_is_used.md`
7. `03-transfer-and-llm-training/llm_training_pipeline.md`

Best if your focus is:

- sequence modeling
- attention
- transformers
- GPT and LLMs
- modern staged training pipelines

### Track 2: Classical Non-Linear Learning Theory

1. `04-kernel-methods/kernel_functions.md`
2. `04-kernel-methods/kernel.txt`

Best if your focus is:

- kernels
- implicit feature spaces
- margin methods
- non-linear regression and classification

## How The Sections Connect

At a high level:

```text
foundations
    ->
sequence models
    ->
transformers
    ->
transfer learning and LLM training

foundations
    ->
kernel geometry
    ->
classical non-linear ML
```

The common thread across the whole library is:

- how representations, geometry, and optimization make hard learning problems
  easier

## Current Library

- `01-foundations/README.md`
- `02-sequence-models/README.md`
- `03-transfer-and-llm-training/README.md`
- `04-kernel-methods/README.md`
