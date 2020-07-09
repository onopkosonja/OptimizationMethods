from Utils import read_file, LogisticRegression
from GD import *
from Newton import *
from HFN import *
from BFGS import *
from LBFGS import *
from SGD import *
from Adam import *

import argparse
import json


def run(opt_method, path, seed, eps, max_iter, *args):
    x, y = read_file(path)
    M = LogisticRegression(x, y, seed)
    w_k = opt_method(M, eps, max_iter, *args)
    f_val = M.calc_func(w_k)
    grad = M.calc_grad(w_k)
    grad_norm = LA.norm(grad)
    res = {
        'f_opt': f_val,
        'grad_norm': grad_norm,
        'rk': M.r_k[-1],
        'oracle_calls': M.oracle_calls,
        'solution': w_k.tolist(),
        'time': M.total_time}
    return res


parser = argparse.ArgumentParser()

methods = {'GD': gd, 'Newton': newton, 'HFN': hf_newton, 'BFGS': bfgs, 'L-BFGS': bfgs,
           'SGD': sgd, 'Adam': adam}

line_search = {'ExactBrent': brent_step, 'ExactGoldenSearch': gss_step, 'Armijo': armijo_step,
           'Wolfe': wolfe_step, 'Lipshitz': lipsitz_step}


# Common arguments
parser.add_argument('--method', type=str, default='GD')
parser.add_argument('--path', type=str, default='breast-cancer.svm')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epsilon', type=float, default=1e-7)
parser.add_argument('--iter_count', type=int, default=10000)

# GD arguments
parser.add_argument('--line_search_method', type=str, default='ExactGoldenSearch')

# HFN arguments
parser.add_argument('--cg_tolerance_policy', type=str, default='const')
parser.add_argument('--cg_tolerance_eta', type=float, default=1e-2)

# L-BFGS arguments
parser.add_argument('--lbfgs_history_size', type=int, default=10)

# SGD & Adam arguments
parser.add_argument('--batch_size', type=int, default=100)


args = parser.parse_args()
method = methods[args.method]

if args.method in ('GD', 'Newton'):
    line_search_method = line_search[args.line_search_method]
    res = run(method, args.path, args.seed, args.epsilon, args.iter_count, line_search_method)

elif args.method == 'HFN':
    res = run(method, args.path, args.seed, args.epsilon, args.iter_count, args.cg_tolerance_policy,
              args.cg_tolerance_eta)

elif args.method == 'L-BFGS':
    res = run(method, args.path, args.seed, args.epsilon, args.iter_count, args.lbfgs_history_size)

elif args.method in ('SGD', 'Adam'):
    res = run(method, args.path, args.seed, args.epsilon, args.iter_count, args.batch_size)

else:
    res = run(method, args.path, args.seed, args.epsilon, args.iter_count)

res = json.dumps(res)
outfile = open('output.json', 'w')
print(res, file=outfile)
