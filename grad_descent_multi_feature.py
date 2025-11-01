import numpy as np


def compute_mse(x, y, w, b):
    """
    Compute mean squared error cost for multiple features
    Args:
        x: (ndarray) shape (m, n) - m examples, n features
        y: (ndarray) shape (m,) - target values
        w: (ndarray) shape (n,) - weights for each feature
        b: (scalar) - bias parameter
    Returns:
        total_cost: (scalar) - cost
    """
    m = x.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        cost += (f_wb - y[i]) ** 2
    total_cost = cost / (2 * m)
    return total_cost


def compute_gradient(x, y, w, b):
    """
    Compute gradient for multiple features
    Args:
        x: (ndarray) shape (m, n) - m examples, n features
        y: (ndarray) shape (m,) - target values
        w: (ndarray) shape (n,) - weights for each feature
        b: (scalar) - bias parameter
    Returns:
        dj_dw: (ndarray) shape (n,) - gradient of cost w.r.t. w
        dj_db: (scalar) - gradient of cost w.r.t. b
    """
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, learning_rate, iters):
    """
    Perform gradient descent for multiple features
    Args:
        x: (ndarray) shape (m, n) - m examples, n features
        y: (ndarray) shape (m,) - target values
        w_in: (ndarray) shape (n,) - initial weights
        b_in: (scalar) - initial bias
        learning_rate: (scalar) - learning rate
        iters: (int) - number of iterations
    Returns:
        w: (ndarray) shape (n,) - optimized weights
        b: (scalar) - optimized bias
    """
    w = w_in.copy()
    b = b_in

    for i in range(iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        # Optional: print cost every 100 iterations
        if i % 100 == 0:
            cost = compute_mse(x, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b


if __name__ == "__main__":
    X = np.array([[2104, 5, 1], 
                  [1416, 3, 2], 
                  [852, 2, 1]])
    y = np.array([460, 232, 178])
    
    # Initialize parameters
    w_init = np.zeros(3)
    b_init = 0
    
    # Run gradient descent
    learning_rate = 1e-7
    iterations = 1000
    
    w_final, b_final = gradient_descent(X, y, w_init, b_init, learning_rate, iterations)
    
    print(f"\nFinal weights: {w_final}")
    print(f"Final bias: {b_final:.4f}")
    print(f"Final cost: {compute_mse(X, y, w_final, b_final):.4f}")
