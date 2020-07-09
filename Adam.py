import time
import numpy as np


def adam(M, e, iter_num, batch_size):
    M.total_time = time.time()
    step_size = 0.001
    w = M.w_0
    func_0 = M.null_oracle(w)
    M.r_k.append(func_0)
    m, v = 0, 0
    b1, b2 = 0.9, 0.999
    t = 0

    for _ in range(iter_num):
        t += 1
        ind = np.random.choice(M.n, batch_size, replace=False)
        grad = M.first_oracle(w, ind)[1]
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_est = m / (1 - b1 ** t)
        v_est = v / (1 - b2 ** t)
        w = w - step_size * m_est / (v_est ** (1 / 2) + e)

        if t % 5 == 0:
            func_k, _ = M.first_oracle(w)
            M.r_k.append(func_k)

    M.total_time = time.time() - M.total_time
    return w
