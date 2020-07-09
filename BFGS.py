from numpy import linalg as LA
from LineSearch import wolfe_step
import numpy as np
import copy
import time


def approximate_inverse_hessian(H_cur, x_cur, x_next, g_cur, g_next):
    s = (x_next - x_cur).reshape((x_cur.shape[0], 1))
    y = (g_next - g_cur).reshape((x_cur.shape[0], 1))
    M1 = (s.T @ y + y.T @ H_cur @ y) * (s @ s.T) / ((s.T @ y) ** 2)
    M2 = (H_cur @ y @ s.T + s @ y.T @ H_cur) / (s.T @ y)
    H_next = H_cur + M1 - M2
    return H_next


def bfgs(M, e, max_iter, *args):
    M.total_time = time.time()
    w_cur = M.w_0
    func_cur, grad_cur = M.first_oracle(w_cur)
    func_0, grad_0 = func_cur, copy.deepcopy(grad_cur)
    H_cur = np.eye(w_cur.shape[0])
    d = -H_cur @ grad_cur
    s = wolfe_step(M, w_cur, d)
    if s is None:
        s = 1
    w_next = w_cur + s * d
    func_next, grad_next = M.first_oracle(w_next)
    H_next = approximate_inverse_hessian(H_cur, w_cur, w_next, grad_cur, grad_next)

    M.r_k.append(LA.norm(grad_0) ** 2)

    while LA.norm(grad_cur) ** 2 / LA.norm(grad_0) ** 2 > e and M.iter_num < max_iter:
        H_cur, w_cur, func_cur, grad_cur = H_next, w_next, func_next, grad_next
        d = -H_cur @ grad_cur
        s = wolfe_step(M, w_cur, d)
        if s is None:
            s = 1
        w_next = w_cur + s * d
        func_next, grad_next = M.first_oracle(w_next)
        H_next = approximate_inverse_hessian(H_cur, w_cur, w_next, grad_cur, grad_next)

        M.iter_num += 1
        M.r_k.append(LA.norm(grad_cur) ** 2)
    M.total_time = time.time() - M.total_time
    return w_cur