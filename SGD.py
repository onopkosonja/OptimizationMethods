import numpy as np
import time


def sgd_direction(M, w, size):
    ind = np.random.choice(M.n, size, replace=False)
    _, mean_grad = M.first_oracle(w, ind)
    return -mean_grad


def sgd_step_size(k, nu=50, theta=0.6):
    return nu / (k + 1) ** theta


def sgd(M, e, iter_num, batch_size):
    M.total_time = time.time()
    w = M.w_0
    func_0 = M.null_oracle(w)

    M.r_k.append(func_0)
    for _ in range(iter_num):
        direction = sgd_direction(M, w, batch_size)
        step_size = sgd_step_size(M.iter_num)
        w += step_size * direction
        M.iter_num += 1

        if M.iter_num % 5 == 0:
            func_k, _ = M.first_oracle(w)
            M.r_k.append(func_k)

    M.total_time = time.time() - M.total_time
    return w

