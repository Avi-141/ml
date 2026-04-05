# ML Notes Roadmap

This folder now contains theory notes written in a more technical style. Most of
the core documents follow the same structure:

1. `Intro And Concepts`
2. `Deep Dive`

The goal is not just definition-level understanding, but mechanism-level
understanding:

- what the concept is
- why it exists
- how it works
- what the math is doing
- what geometry or intuition sits underneath it
- where it shows up in modern ML systems

## Reading Order

If you want to study these notes in a logical sequence, this is the recommended
path:

1. `neural_networks_math_and_statistics.md`
2. `matrices_eigenvalues_and_eigenvectors_for_ml.md`
3. `gradients_loss_and_gradient_descent.md`
4. `backpropagation_from_first_principles.md`
5. `sequence_models_rnn_lstm_gru.md`
6. `attention_and_self_attention.md`
7. `transformers_from_first_principles.md`
8. `adam_optimizer.md`
9. `transfer_learning.md`
10. `pretraining_and_finetuning.md`
11. `where_transfer_learning_is_used.md`
12. `softmax.md`
13. `kernel_functions.md`
14. `kernel.txt`

Why this order:

- `neural_networks_math_and_statistics.md` gives the broad mathematical and
  statistical foundations for neural networks
- `matrices_eigenvalues_and_eigenvectors_for_ml.md` explains why matrices,
  spectra, covariance, and curvature matter in ML
- `gradients_loss_and_gradient_descent.md` builds the intuition for losses,
  gradients, backpropagation, and descent methods
- `backpropagation_from_first_principles.md` connects gradients, chain rule,
  matrix shapes, and efficient gradient computation in layered models
- `sequence_models_rnn_lstm_gru.md` explains how neural networks model ordered
  data through recurrence and gating
- `attention_and_self_attention.md` explains how dynamic relevance weighting
  addresses key sequence-model bottlenecks
- `transformers_from_first_principles.md` shows how self-attention becomes the
  main sequence computation in modern architectures
- `adam_optimizer.md` explains one of the most important practical optimizers in
  modern deep learning
- `transfer_learning.md` defines the broad idea of knowledge reuse across tasks
- `pretraining_and_finetuning.md` explains the standard modern training pipeline
- `where_transfer_learning_is_used.md` connects those ideas to transformers,
  GPT, LLMs, sequence models, and other domains
- `softmax.md` explains how neural classifiers convert logits into probabilities
  and why the standard modern training pattern uses linear logits with stable
  cross-entropy
- `kernel_functions.md` shifts to a more classical but mathematically important
  view of non-linearity through geometry and implicit feature spaces
- `kernel.txt` extends the kernel discussion into specific model families such
  as KRR, SVR, SVM, and sparse kernel-style formulations

## Note Map

### `neural_networks_math_and_statistics.md`

Focus:

- neural networks as parameterized function classes
- where linear algebra, calculus, optimization, and statistics fit together
- why matrices appear in layers
- representations, losses, and empirical risk
- approximation, estimation, and optimization viewpoints

Best read when:

- you want the broad mathematical frame for neural networks before diving into
  specific optimization details

### `matrices_eigenvalues_and_eigenvectors_for_ml.md`

Focus:

- why matrices work in ML and neural networks
- what eigenvalues and eigenvectors mean geometrically
- covariance, PCA, Hessians, and conditioning
- singular values versus eigenvalues
- recurrent dynamics and spectral intuition

Best read when:

- you want the linear algebra and geometric intuition behind neural-network
  training and statistical structure

### `gradients_loss_and_gradient_descent.md`

Focus:

- why gradients work
- why negative gradient is steepest local descent
- why loss functions are chosen the way they are
- local linearization, chain rule, and backpropagation
- stochastic gradient descent, curvature, and conditioning

Best read when:

- you want the mathematical intuition behind training itself

### `backpropagation_from_first_principles.md`

Focus:

- backpropagation as reverse-mode automatic differentiation
- chain rule through layered computation
- parameter gradients in matrix form
- why the forward pass must cache activations
- vanishing and exploding gradients
- why BPTT is just backprop on an unrolled recurrent graph

Best read when:

- you want the exact mechanism by which neural networks obtain gradients

### `sequence_models_rnn_lstm_gru.md`

Focus:

- why sequences require special architectures
- hidden-state recurrence
- RNN state updates and temporal parameter sharing
- vanishing/exploding gradients in sequence learning
- LSTM and GRU gating intuition and equations
- why recurrence led naturally to attention

Best read when:

- you want the pre-transformer story of sequence modeling

### `attention_and_self_attention.md`

Focus:

- attention as relevance-weighted retrieval
- encoder-decoder attention
- query, key, and value formalism
- self-attention and why softmax appears inside it
- multi-head attention, causal masking, and long-range dependency handling
- why attention was the bridge from recurrence to transformers

Best read when:

- you want to understand the conceptual and mathematical jump from RNNs to
  attention-based models

### `transformers_from_first_principles.md`

Focus:

- transformer blocks from first principles
- self-attention, feedforward layers, residuals, and normalization
- positional information
- encoder-only, decoder-only, and encoder-decoder structures
- why transformers scale and why they replaced recurrence in many domains

Best read when:

- you want the architectural culmination of the sequence-model track

### `adam_optimizer.md`

Focus:

- why Adam works
- momentum and second-moment adaptation
- bias correction
- coordinatewise scaling
- Adam vs SGD, RMSProp, and AdamW
- strengths, failure modes, and practical intuition

Best read when:

- you want to understand modern optimizer behavior beyond plain gradient descent

### `transfer_learning.md`

Focus:

- formal definition of transfer learning
- source task vs target task
- why transfer helps statistically and computationally
- positive transfer vs negative transfer
- feature extraction vs fine-tuning
- representation, optimization, and prior views

Best read when:

- you want the conceptual foundation for modern model reuse

### `pretraining_and_finetuning.md`

Focus:

- what pretraining is
- self-supervised vs supervised pretraining
- what fine-tuning changes
- objective-function viewpoint
- catastrophic forgetting
- parameter-efficient adaptation

Best read when:

- you want to understand how modern models are actually trained in stages

### `where_transfer_learning_is_used.md`

Focus:

- where transfer learning appears across ML
- sequence models
- transformers
- encoder-only, decoder-only, encoder-decoder models
- GPT and LLM workflows
- speech, multimodal, and scientific domains

Best read when:

- you want architectural and domain context

### `kernel_functions.md`

Focus:

- kernel as inner product in feature space
- kernel trick
- Gram matrix
- valid kernels and PSD condition
- RBF, polynomial, linear, sigmoid kernels
- RKHS intuition
- geometry, smoothness, and regularization

Best read when:

- you want the mathematical foundation behind kernel methods

### `softmax.md`

Focus:

- logits and probability distributions
- why softmax is used for multi-class classification
- why the last layer is often linear during training
- cross-entropy, gradients, and numerical stability
- standard, convolutional, recurrent, and language-model uses of softmax
- temperature, calibration, and large-vocabulary approximations

Best read when:

- you want a deep understanding of modern multi-class output layers and the
  probability geometry behind them

### `kernel.txt`

Focus:

- kernel regularization
- kernel ridge regression
- kernel SVM classification
- kernel SVR
- kernel lasso and elastic-net-style ideas
- dense vs sparse dual solutions
- practical trade-offs and scalability

Best read when:

- you want the model-family view that sits on top of kernel-function theory

## Two Suggested Study Tracks

### Track 0: Neural-Network Math And Optimization

Read in this order:

1. `neural_networks_math_and_statistics.md`
2. `matrices_eigenvalues_and_eigenvectors_for_ml.md`
3. `gradients_loss_and_gradient_descent.md`
4. `backpropagation_from_first_principles.md`
5. `adam_optimizer.md`
6. `softmax.md`

This track is best if your focus is:

- why neural networks are written with matrices
- what losses and gradients are doing mathematically
- why eigenvalues, curvature, and conditioning matter
- how backpropagation computes usable gradients through layered systems
- why gradient descent and Adam work
- how logits become probabilities

### Track 1: Modern Deep Learning Systems

Read in this order:

1. `sequence_models_rnn_lstm_gru.md`
2. `attention_and_self_attention.md`
3. `transformers_from_first_principles.md`
4. `transfer_learning.md`
5. `pretraining_and_finetuning.md`
6. `where_transfer_learning_is_used.md`
7. `softmax.md`
8. `adam_optimizer.md`

This track is best if your focus is:

- sequence modeling
- transformers
- GPT
- LLMs
- instruction tuning
- foundation models
- classification heads, logits, and next-token distributions
- practical optimization in modern deep learning

### Track 2: Classical Non-Linear Learning Theory

Read in this order:

1. `kernel_functions.md`
2. `kernel.txt`

This track is best if your focus is:

- kernels
- implicit feature spaces
- margin methods
- non-linear regression and classification

## How The Notes Connect

At a high level:

```text
transfer learning
    ->
pretraining / fine-tuning pipelines
    ->
modern transformers / LLM workflows

kernel methods
    ->
implicit feature-space geometry
    ->
non-linear classical ML models such as SVM, SVR, KRR
```

These are different parts of machine learning history and practice, but they are
connected by a shared theme:

- how learned or chosen representations make difficult problems easier

## Possible Next Additions

Natural next documents to add:

- `attention_and_self_attention.md`
- `transformers_from_first_principles.md`
- `svm_margin_duality.md`
- `rkhs_and_representer_theorem.md`
- `llm_training_pipeline.md`
- `fine_tuning_vs_in_context_learning.md`
- `pca_and_covariance_geometry.md`

## Current Notes

- `README.md`
- `neural_networks_math_and_statistics.md`
- `matrices_eigenvalues_and_eigenvectors_for_ml.md`
- `gradients_loss_and_gradient_descent.md`
- `backpropagation_from_first_principles.md`
- `sequence_models_rnn_lstm_gru.md`
- `attention_and_self_attention.md`
- `transformers_from_first_principles.md`
- `adam_optimizer.md`
- `transfer_learning.md`
- `pretraining_and_finetuning.md`
- `where_transfer_learning_is_used.md`
- `softmax.md`
- `kernel_functions.md`
- `kernel.txt`
