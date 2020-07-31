import numpy as np
import pandas as pd
import dask
from dask.distributed import Client, progress
import itertools
from maxnorm.maxnorm_completion import *
from maxnorm.tenalg import *
from maxnorm.graphs import *

def generate_data(obs_mask, U, sigma):
    data = obs_mask.copy()
    clean_data = kr_get_items(U, data.coords)
    clean_data_rms = np.sqrt(np.sum(clean_data)**2 / len(clean_data))
    data.data = clean_data + np.random.randn(data.nnz) * sigma * clean_data_rms
    return data

def gen_err(Upred, Utrue):
    norm_true = kr_dot(Utrue, Utrue)
    mse_gen = kr_dot(Upred, Upred) + norm_true - 2 * kr_dot(Upred, Utrue)
    return np.sqrt(mse_gen / norm_true)

def run_simulation(n, t, r, sigma, r_fit, rep, const = 10,
                       max_iter=100, inner_max_iter=40, tol=1e-8, alg='max', verbosity=0):
    n = int(n)
    t = int(t)
    r = int(r)
    r_fit = int(r_fit)
    rep = int(rep)
    ndata =  const * r * t * n * float(np.log10(n))
    U = kr_random(n, t, r)
    max_qnorm_ub_true = max_qnorm_ub(U)
    observation_mask = obs_mask_iid(tuple([n for i in range(t)]), ndata * n**(-t))
    data = generate_data(observation_mask, U, sigma)
    clean_data_rmse = np.sqrt(loss(U, data) / data.nnz)
    delta = 2 * clean_data_rmse
    if alg == 'als':
        U_fit, cost_arr = \
          tensor_completion_alt_min(data, r_fit, init='svd', max_iter=max_iter, tol=tol,
                                        inner_max_iter=max_iter_inner)
    elif alg == 'max':
        U_fit, cost_arr = \
          tensor_completion_maxnorm(data, r_fit, delta * np.sqrt(data.nnz), epsilon=0., 
                                    #sgd=True, sgd_batch_size=int(ndata/2),
                                    #U0 = Unew1,
                                    init='svdrand', kappa=100, beta=1,
                                    verbosity=verbosity,
                                    tol=tol, max_iter=max_iter, inner_max_iter=inner_max_iter)
    elif alg == 'both':
        U_fit_als, cost_arr_als = \
          tensor_completion_alt_min(data, r_fit, init='svd', max_iter=max_iter, tol=tol,
                                        inner_max_iter=max_iter_inner)
        U_fit_max, cost_arr_max = \
          tensor_completion_maxnorm(data, r_fit, delta * np.sqrt(data.nnz), epsilon=0., 
                                    #sgd=True, sgd_batch_size=int(ndata/2),
                                    #U0 = Unew1,
                                    init='svdrand', kappa=100, beta=1,
                                    verbosity=verbosity,
                                    tol=tol, max_iter=max_iter, inner_max_iter=inner_max_iter)
    else:
        raise Exception('unexpected algorithm')

    if alg != 'both':
        loss_true = np.sqrt(loss(U, data) / data.nnz)
        loss_val = np.sqrt(loss(U_fit, data) / data.nnz)
        gen_err_val = gen_err(U_fit, U)
        max_qnorm_ub_val = max_qnorm_ub(U_fit)
        return loss_true, max_qnorm_ub_true, loss_val, max_qnorm_ub_val, gen_err_val
    else:
        loss_true = np.sqrt(loss(U, data) / data.nnz)
        loss_als = np.sqrt(loss(U_fit_als, data) / data.nnz)
        max_qnorm_ub_als = max_qnorm_ub(U_fit_als)
        gen_err_als = gen_err(U_fit_als, U)
        loss_max = np.sqrt(loss(U_fit_max, data) / data.nnz)
        max_qnorm_ub_max = max_qnorm_ub(U_fit_max)
        gen_err_max = gen_err(U_fit_max, U)
        return loss_true, max_qnorm_ub_true, loss_als, gen_err_als, loss_max, gen_err_max

if __name__ == '__main__':
    # generate parameters for a sweep
    n = [20,40,80,160]
    #n = [10]
    t = [3, 4, 5]
    r = [3]
    sigma = [0.1]
    r_fit = [1, 3, 5, 10, 15, 20]
    rep = [i for i in range(10)]
    const = [1, 5, 10, 20, 40, 80]
    # n = [10]
    # t = [3]
    # r = [3]
    # sigma = [0.1]
    # r_fit = [6]
    # rep = [0, 1, 2, 3]
    param_list = [n, t, r, sigma, r_fit, rep, const]
    params = list(itertools.product(*param_list))
    param_df = pd.DataFrame(params, columns=['n', 't', 'r', 'sigma', 'r_fit', 'rep', 'const'])

    # setup dask job
    client = Client()
    client

    lazy_results = []
    for parameters in param_df.values:
        lazy_result = dask.delayed(run_simulation)(*parameters)
        lazy_results.append(lazy_result)

    futures = dask.persist(*lazy_results)
    progress(futures)
    # call computation
    results = dask.compute(*futures)
    data = pd.DataFrame(results, columns=['loss_true', 'max_qnorm_ub_true',
                                          'loss_fit', 'max_qnorm_ub_fit', 'gen_err_fit'])
    # param_df.to_csv("params_max.csv")
    # data.to_csv("results_max.csv")
    table = param_df.join(data)
    table.to_csv("max_n.csv")
