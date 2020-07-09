from numpy import linalg as LA
from LineSearch import armijo_step
import numpy as np
import copy
import time


def hes_vect_product(M, x, v, r):
    g1 = M.first_oracle(x + r * v)[1]
    g2 = M.first_oracle(x - r * v)[1]
    return (g1 - g2) / (2 * r)


def rate_of_convergence(rate):
    def inner(x):
        if rate == 'sqrtGradNorm':
            return min(0.5, LA.norm(x) ** (1 / 2))
        elif rate == 'gradNorm':
            return min(0.5, LA.norm(x))
    return inner


def conjugate_gradient_descent(M, w, grad_k, e_k, max_iter):
    n = w.shape
    z_j = np.zeros(n)
    grad_j = hes_vect_product(M, w, z_j, e_k) + grad_k
    d_j = -grad_j

    for j in range(max_iter):
        hes_vect_pr = hes_vect_product(M, w, d_j, e_k)

        alpha_j = grad_j.T @ grad_j / (d_j.T @ hes_vect_pr)
        z_j1 = z_j + alpha_j * d_j
        grad_j1 = grad_j + alpha_j * hes_vect_pr

        if LA.norm(grad_j1) < e_k:
            return z_j1

        beta_j = grad_j1.T @ grad_j1 / (grad_j.T @ grad_j)
        d_j1 = - grad_j1 + beta_j * d_j

        grad_j = grad_j1
        z_j = z_j1
        d_j = d_j1


def hf_newton(M, eps, max_iter, tolerance_policy, tolerance_eta):
    M.total_time = time.time()
    w = M.w_0
    func_0, grad_k = M.first_oracle(w)
    grad_0 = copy.deepcopy(grad_k)
    M.r_k.append(LA.norm(grad_0) ** 2)

    while LA.norm(grad_k) ** 2 / LA.norm(grad_0) ** 2 > eps and M.iter_num < max_iter:
        conv_func = rate_of_convergence(tolerance_policy)
        if tolerance_policy == 'const':
            e_k = 0.5 * tolerance_eta
        else:
            e_k = conv_func(grad_k) * tolerance_eta
        p_k = conjugate_gradient_descent(M, w, grad_k, e_k, max_iter)
        a_k = armijo_step(M, w, p_k)
        w += a_k * p_k
        func_k, grad_k = M.first_oracle(w)

        M.iter_num += 1
        M.r_k.append(LA.norm(grad_k) ** 2)

    M.total_time = time.time() - M.total_time
    return w