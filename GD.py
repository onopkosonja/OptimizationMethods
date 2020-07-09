from numpy import linalg as LA
from LineSearch import *
import time


def gd(M, e, max_iter, ls_method):
    M.total_time = time.time()
    w = M.w_0

    func_0, grad_0 = M.first_oracle(w)
    s = ls_method(M, w, grad_0, e)
    w -= s * grad_0
    M.r_k.append(LA.norm(grad_0) ** 2)

    grad_k = M.first_oracle(w)[1]

    while LA.norm(grad_k) ** 2 / LA.norm(grad_0) ** 2 > e and M.iter_num < max_iter:
        s = ls_method(M, w, grad_k, e)
        w -= s * grad_k
        func_k, grad_k = M.first_oracle(w)

        M.iter_num += 1
        M.r_k.append(LA.norm(grad_k) ** 2)

    M.total_time = time.time() - M.total_time
    return w