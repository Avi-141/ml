

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
