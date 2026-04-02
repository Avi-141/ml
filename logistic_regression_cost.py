import numpy as np

"""
Logistic Regression Cost Functions

This module implements various cost functions used in Logistic Regression and related optimization problems.

Mathematical Formulation:
The standard cost function for logistic regression is the Cross-Entropy Loss (or Log Loss).
Given a dataset with m examples, where x^(i) is the feature vector and y^(i) is the true label (0 or 1):
    f_wb(x^(i)) = g(w·x^(i) + b) = 1 / (1 + e^{-(w·x^(i) + b)})
    J(w,b) = - (1/m) * Σ [ y^(i) * log(f_wb(x^(i))) + (1 - y^(i)) * log(1 - f_wb(x^(i))) ]

Statistical Perspective:
The cross-entropy cost function is derived from Maximum Likelihood Estimation (MLE).
Assuming the target variable y follows a Bernoulli distribution given x, parameterized by p = f_wb(x):
    P(y|x; w,b) = p^y * (1-p)^(1-y)
The likelihood of the entire dataset is the dataset's joint probability.
Minimizing the negative log-likelihood of this distribution yields the exact Cross-Entropy Loss equation.

Intuition:
- If the true label y = 1 and the model predicts p ≈ 1, the cost is near 0.
- If the true label y = 1 and the model predicts p ≈ 0, the cost approaches infinity (heavy penalty).
- The log function ensures that confident but wrong predictions are penalized exponentially, driving the model to adjust its weights.
- Unlike Mean Squared Error (MSE), which is non-convex for logistic regression, Cross-Entropy Loss is convex, guaranteeing a single global minimum during gradient descent optimization.

Applications:
- Binary Classification: Spam detection (Spam/Not Spam), Disease diagnosis (Malignant/Benign), Fraud detection.
- Probability Calibration: Outputting a well-calibrated probability score (e.g., 80% chance of rain) rather than just a hard class label.
- Base for Neural Networks: Cross-entropy is the standard loss function for classification tasks in deep learning.
"""

# mean square error cost function
'''
MSE = (1/2m) * Σ(f_wb - y)^2
Errors are squared so larger errors are penalized more heavily
The division by 2m simplifies the derivative calculation during gradient descent
Its a convex function, ensuring a single global minimum, so optimization is straightforward
Forms basis for more complex regression techniques
'''
def compute_cost_mean_squared(x, y, wt, b):
    # x feature
    # y target
    # wt weights
    # b bias
    m = x.shape[0] # number of training examples
    total_cost = 0.0
    for i in range(m):
        f_wb = x[i].dot(wt) + b
        total_cost += (f_wb - y[i]) ** 2    

    return total_cost / (2 * m)


# cross entropy cost function
# log likelihood cost function
'''
Linear combination: f_wb = x[i].dot(wt) + b computes wx + b
Sigmoid activation: prediction = 1 / (1 + np.exp(-f_wb)) applies the sigmoid function to get predicted probabilities
Cross-entropy loss calculation: total_cost += -y[i] * np.log(prediction) - (1 - y[i]) * np.log(1 - prediction) computes the loss for each example
When y[i] = 1, the loss is -log(prediction); 
When y[i] = 0, the loss is -log(1 - prediction)

Its a convex max optimization function
It penalizes wrong predictions more heavily, especially when the predicted probability is far from the actual class label.
Its the maximization of likelihood estimatior for logistic regression
And provides well-calibrated probabilities for classification tasks
'''

# Maximum Likelihood Estimation = Minimizing negative Log-Likelihood
# Assumes data follows a Bernoulli distribution
# Guaranteed global minimum due to convexity
#standard binary classification
def compute_cost_cross_entropy_log_likelihood(x, y, wt, b):
    # x feature
    # y target
    # wt weights
    # b bias
    m = x.shape[0] # number of training examples
    total_cost = 0.0
    for i in range(m):
        f_wb = x[i].dot(wt) + b
        prediction = 1 / (1 + np.exp(-f_wb))
        total_cost += -y[i] * np.log(prediction) - (1 - y[i]) * np.log(1 - prediction)

    return total_cost / m


#L1 Regularization: Feature selection, sparse models
# Creates sparse models by driving some weights to exactly zero
# Feature selection: Automatically removes irrelevant features
'''
Mathematical intuition:
L1 norm creates "diamond-shaped" constraint regions
Optimization tends to hit corners where weights = 0
Higher λ = more sparsity

When to use:

High-dimensional data with many irrelevant features
Need interpretable models with fewer features
Automatic feature selection

Trade-offs:

Good : Feature selection, interpretability
Bad : Can arbitrarily select among correlated features
'''
def compute_cost_l1_lasso(x, y, wt, b, lambda_reg):
    # x feature
    # y target
    # wt weights
    # b bias
    m = x.shape[0] # number of training examples
    total_cost = 0.0
    for i in range(m):
        f_wb = x[i].dot(wt) + b
        prediction = 1 / (1 + np.exp(-f_wb))
        total_cost += -y[i] * np.log(prediction) - (1 - y[i]) * np.log(1 - prediction)
     # Add L1 regularization term
    reg_term = lambda_reg * np.sum(np.abs(wt))

    return (total_cost / m) + reg_term

#L2 Regularization: Prevent overfitting, smoother models
'''
Mathematical intuition:

L2 norm creates "circular" constraint regions
Smoothly shrinks all weights
Higher λ = more shrinkage

When to use:

Overfitting problems
Multicollinearity (correlated features)
Want to keep all features but reduce their impact

Trade-offs:

Good : Handles correlated features well, smooth regularization
Bad : Doesn't perform feature selection

'''
def compute_cost_l2_ridge(x, y, wt, b, lambda_reg):
    m = x.shape[0]
    total_cost = 0.0
    
    # Standard logistic cost
    for i in range(m):
        f_wb = x[i].dot(wt) + b
        prediction = 1 / (1 + np.exp(-f_wb))
        total_cost += -y[i] * np.log(prediction) - (1 - y[i]) * np.log(1 - prediction)
    
    # Add L2 regularization term
    reg_term = lambda_reg * np.sum(wt ** 2) / 2
    
    return total_cost / m + reg_term