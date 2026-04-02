

"""
Gradient Descent for Single Feature Linear Regression (Univariate)

This module implements the fundamental components of linear regression with a single variable,
optimized using the batch gradient descent algorithm.

Mathematical Formulation:
1. Hypothesis (Model): 
   f_wb(x) = w * x + b
   where 'w' is the weight (slope) and 'b' is the bias (y-intercept).

2. Cost Function (Mean Squared Error - MSE):
   J(w,b) = (1 / 2m) * Σ_{i=1}^{m} (f_wb(x^(i)) - y^(i))^2
   The division by 2 is a mathematical convenience that cancels out when we take the derivative.

3. Gradient Descent Update Rules:
   Repeat until convergence:
     w = w - α * (∂J / ∂w)
     b = b - α * (∂J / ∂b)
   where α (alpha) is the learning rate, and the partial derivatives are:
     ∂J / ∂w = (1 / m) * Σ_{i=1}^{m} (f_wb(x^(i)) - y^(i)) * x^(i)
     ∂J / ∂b = (1 / m) * Σ_{i=1}^{m} (f_wb(x^(i)) - y^(i))

Statistical Perspective:
Minimizing the Mean Squared Error is mathematically equivalent to Maximum Likelihood Estimation (MLE) 
under the assumption that the target variable y is generated from the linear model plus normally 
distributed noise (Gaussian error).

Intuition:
- Gradient descent is like walking down a hill blindfolded. The gradient tells you the slope 
  of the hill at your current position, and you take a step in the direction of steepest descent.
- The learning rate α determines the size of the step. If α is too small, convergence is slow. 
  If α is too large, the algorithm may overshoot the minimum and diverge.
- Since the MSE cost function for linear regression is a convex "bowl" shape, gradient descent 
  is guaranteed to find the single global minimum (assuming an appropriate learning rate).

Applications:
- Predicting a continuous outcome based on a single predictor variable.
- Examples: Predicting house prices based solely on square footage, estimating salary based on 
  years of experience, or predicting sales based on advertising spend.
"""

def compute_mse(x, y, wt, b):
    m = x.shape[0];
    cost = 0.0

    for i in range(m):
        f_wb = x[i].dot(wt) + b
        cost += (f_wb - y[i]) ** 2
    total_cost = cost / (2 * m);
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0];
    dj_dw = 0
    dj_db = 0;

    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, learning_rate, iters):
    w = w_in
    b = b_in

    for i in range(iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

    return w, b
