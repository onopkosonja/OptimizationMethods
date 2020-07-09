from numpy import linalg as LA
from LineSearch import *
import time
import numpy as np


def newton(M, e, max_iter, method):
    M.total_time = time.time()
    w = M.w_0
    func_0, grad_0, hes_0 = M.second_oracle(w)
    s = method(M, w, grad_0, e)
    U, S, V = LA.svd(hes_0)
    S[S < 1e-02] = 1e-02
    S = np.diag(S)
    w = w - s * (V.T @ (LA.inv(S) @ (U.T @ grad_0)))
    func_k, grad_k, hes_k = M.second_oracle(w)

    M.r_k.append(LA.norm(grad_0) ** 2)

    while LA.norm(grad_k) ** 2 / LA.norm(grad_0) ** 2 > e and M.iter_num < max_iter:
        s = method(M, w, grad_k, e)
        U, S, V = LA.svd(hes_k)
        S[S < 1e-02] = 1e-02
        S = np.diag(S)
        w = w - s * (V.T @ (LA.inv(S) @ (U.T @ grad_k)))
        func_k, grad_k, hes_k = M.second_oracle(w)

        M.iter_num += 1
        M.r_k.append(LA.norm(grad_k) ** 2)

    M.total_time = time.time() - M.total_time
    return w

