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
    #clean_data_rms = np.sqrt(np.sum(clean_data)**2 / len(clean_data))
    clean_data_rms = 1
    data.data = clean_data + np.random.randn(data.nnz) * sigma * clean_data_rms
    return data

def gen_err(Upred, Utrue):
    norm_true = kr_dot(Utrue, Utrue)
    mse_gen = kr_dot(Upred, Upred) + norm_true - 2 * kr_dot(Upred, Utrue)
    return np.sqrt(mse_gen / norm_true)

def run_simulation(n, t, r, sigma, r_fit, rep, const = 10.,
                       max_iter=300, inner_max_iter=30, tol=1e-10, alg='max', verbosity=0,
                       kappa=100, beta=1, epsilon=1, delta=None):
    n = int(n)
    t = int(t)
    r = int(r)
    r_fit = int(r_fit)
    rep = int(rep)
    const = float(const)
    ndata = np.round(const * r * t * n * np.log10(n))
    U = kr_random(n, t, r)
    U = kr_rescale(U, np.sqrt(2), 'std')
    #observation_mask = 
    observation_mask = obs_mask_iid(tuple([n for i in range(t)]), float(ndata) * n**(-t))
    max_qnorm_ub_true = max_qnorm_ub(U)
    if verbosity > 1:
        print("Running a simulation: n = %d, t = %d, r = %d, sigma = %f, r_fit = %d\n" \
                  % (n, t, r, sigma, r_fit))
        print("max_qnorm_ub_true = %1.3e" % max_qnorm_ub_true)
        print("number of samples = %d, %1.2e%%" % (ndata, 100. * float(ndata) / n**t))
    data = generate_data(observation_mask, U, sigma)
    clean_data_rmse = np.sqrt(loss(U, data) / data.nnz)
    if delta is None:
        delta = 1 * clean_data_rmse
    if alg == 'als':
        try:
            U_fit, cost_arr = \
              tensor_completion_alt_min(data, r_fit, init='svd', max_iter=max_iter, tol=tol,
                                            inner_max_iter=max_iter_inner, epsilon=epsilon)
        except Exception:
            U_fit = None
    elif alg == 'max':
        try:
            U_fit, cost_arr = \
              tensor_completion_maxnorm(data, r_fit, delta * np.sqrt(data.nnz), epsilon=epsilon, 
                                            #sgd=True, sgd_batch_size=int(ndata/2),
                                            #U0 = Unew1,
                                            init='svdrand', kappa=kappa, beta=beta,
                                            verbosity=verbosity,
                                            tol=tol, max_iter=max_iter, inner_max_iter=inner_max_iter)
        except Exception:
            U_fit = None
    elif alg == 'both':
        try:
            U_fit_als, cost_arr_als = \
              tensor_completion_alt_min(data, r_fit, init='svd', max_iter=max_iter, tol=tol,
                                            inner_max_iter=max_iter_inner, epsilon=epsilon)
        except Exception:
            U_fit_als = None
        try:
            U_fit_max, cost_arr_max = \
              tensor_completion_maxnorm(data, r_fit, delta * np.sqrt(data.nnz), epsilon=epsilon, 
                                            #sgd=True, sgd_batch_size=int(ndata/2),
                                            #U0 = Unew1,
                                            init='svdrand', kappa=kappa, beta=beta,
                                            verbosity=verbosity,
                                            tol=tol, max_iter=max_iter, inner_max_iter=inner_max_iter)
        except Exception:
            U_fit_max = None
    else:
        raise Exception('unexpected algorithm')

    if alg != 'both':
        loss_true = np.sqrt(loss(U, data) / data.nnz)
        if U_fit is not None:
            loss_val = np.sqrt(loss(U_fit, data) / data.nnz)
            gen_err_val = gen_err(U_fit, U)
            max_qnorm_ub_val = max_qnorm_ub(U_fit)
        else:
            loss_val = np.nan
            gen_err_val = np.nan
            max_qnorm_ub_val = np.nan
        return loss_true, max_qnorm_ub_true, loss_val, max_qnorm_ub_val, gen_err_val
    else:
        loss_true = np.sqrt(loss(U, data) / data.nnz)
        if U_fit_als is not None:
            loss_als = np.sqrt(loss(U_fit_als, data) / data.nnz)
            max_qnorm_ub_als = max_qnorm_ub(U_fit_als)
            gen_err_als = gen_err(U_fit_als, U)
        else:
            loss_als = np.nan
            max_qnorm_ub_als = np.nan
            gen_err_als = np.nan
        if U_fit_max is not None:
            loss_max = np.sqrt(loss(U_fit_max, data) / data.nnz)
            max_qnorm_ub_max = max_qnorm_ub(U_fit_max)
            gen_err_max = gen_err(U_fit_max, U)
        else:
            loss_max = np.nan
            max_qnorm_ub_max = np.nan
            gen_err_max = np.nan
        return loss_true, max_qnorm_ub_true, loss_als, gen_err_als, loss_max, gen_err_max

if __name__ == '__main__':
    # generate parameters for a sweep
    n = [10, 20, 40]
    #n = [10]
    t = [3, 4]
    r = [3]
    sigma = [0.01]
    r_fit = [1, 3, 8, 16, 32, 64]
    rep = [i for i in range(6)]
    const = [5, 10, 20, 40, 100]
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
