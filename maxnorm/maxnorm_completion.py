import numpy as np
#from numba import jit
# import numpy.random as random
#import jax.numpy as np
#from jax.config import config
#from jax import jit, grad, pmap
import sparse
from scipy.optimize import root_scalar, fmin_cg
#from hottbox.core import TensorCPD
# import tensorly as tl
# from tensorly.contrib.sparse.decomposition import parafac
# from tensorly.kruskal_tensor import KruskalTensor
# from tensorly.contrib.sparse import tensor as sptensor
# from tensorly.contrib.sparse.kruskal_tensor import unfolding_dot_khatri_rao as sparse_unfolding_dot_khatri_rao
#config.update('jax_enable_x64', True)
# config.update('jax_debug_nans', True)
import copy

from .optimization import *
from .tenalg import *

def norm_2_inf(U):
    '''
    Matrix :math:`\ell^2 \to \ell^1` induced norm

    .. math:: \| U \|_{2 \to \infty} = \max_i \sqrt{\sum_{j=1}^n u_{i,j}^2 }
    '''
    return np.max(np.linalg.norm(U, axis=1))

def norm_2_1(U):
    '''
    Matrix :math:`\ell_{2,1}` norm

    .. math:: \| U \|_{2,1} = \sum_{i=1}^m \sqrt{\sum_{j=1}^n u_{i,j}^2 }
    '''
    return np.sum(np.linalg.norm(U, axis=1))

def proj_norm_2_1(U, r):
    '''
    Project matrix `U` into :math:`\ell_{2,1}` ball of radius `r`

    .. math:: \| U \|_{2,1} = \sum_{i=1}^m \sqrt{\sum_{j=1}^n u_{i,j}^2 }

    Uses the approach of Liu, Ji, Ye (UAI, 2009).
    '''
        
    def _root_fun(l, U, r):
        '''
        Root function for dual variable:
        
        .. math:: f(l) = \sum_{i=1}^m \max( \| u_i \|_2 - l, 0 ) - r
        '''
        return np.sum(np.maximum(np.linalg.norm(U, axis=1) - l, 0)) - r

    if norm_2_1(U) <= r:
        # already in ball, nothing to do
        return U
    else:
        # solve for root
        l_upper = norm_2_inf(U)
        result = root_scalar(_root_fun, args=(U, r), x0=1, bracket=[0, l_upper], rtol=1e-8)
        l = result.root
        # apply shrinkage formula
        W = np.zeros(U.shape)
        ui_norms = np.linalg.norm(U, axis=1)
        ui_scaling = (1 - l / ui_norms) * (ui_norms > l)
        W = np.multiply(U, ui_scaling[:, np.newaxis])
        # for i in range(U.shape[0]):
        #     ui_norm = np.linalg.norm(U[i, :])
        #     if ui_norm > l:
        #         W[i, :] = (1 - l / ui_norm) * U[i, :]
        # W_norm = norm_2_1(W)
        # assert W_norm <= r * (1 + 1e-8), "Returned matrix norm %g not within %g-ball, uh oh!" % (W_norm, r)
        return W

def prox_norm_2_inf(U, t):
    '''
    .. math:: \mathrm{prox}_{t \| \cdot \|_{2 \to \infty}} (U)
    '''
    return U - t * proj_norm_2_1(U / t, 1)

def max_qnorm_ub(U):
    '''
    .. math:: \prod_{i=1}^t \| U^{(i)} \|_{2 \to \infty}
    '''
    t = len(U)
    val = 1.
    for i in range(t):
        val *= norm_2_inf(U[i])
    return val

def loss(U, data):
    return (sparse_resid(U, data) ** 2).sum()

def tensor_completion_maxnorm(data, rank, delta, init='svd', U0=None,
                                  kappa=10., beta=1, epsilon=1e-2,
                                  sgd=False, sgd_batch_size=200,
                                  tol=1e-4, max_iter=10,
                                  inner_max_iter=30, inner_tol=None,
                                  inner_line_iter=30, inner_line_factr=0.5,
                                  verbosity=0, rebalance=True):

    def cost(U, data, delta, kappa, beta, epsilon):
        '''
        cost function: max-qnorm + least squares term
        '''
        resid_norm = np.sqrt(loss(U, data))
        if resid_norm <= (1 + beta / kappa) * delta:
            mu = kappa / (kappa + beta)
        else:
            mu = delta / resid_norm
        #mu = 0.
        tik = 0.
        if epsilon > 0:
            for Us in U:
                tik += 0.5 * epsilon * np.linalg.norm(Us, 'fro') ** 2
        return max_qnorm_ub(U) + 0.5 * (kappa * (1 - mu)**2 + mu**2 * beta) * resid_norm ** 2 + tik
    
    assert isinstance(data, sparse.COO), "data should be sparse.COO"
    t = data.ndim
    # initialize factor matrices
    if inner_tol is None:
        inner_tol = tol / 10
    if U0 is not None:
        U = copy.deepcopy(U0)
    else:
        if init == 'svd':
            U = [sparse_unfold_svs(data, i, rank) for i in range(t)]
        elif init == 'svdrand':
            U = [sparse_unfold_svs(data, i, rank) + \
                     0.3 / np.sqrt(data.shape[i]) * np.random.randn(data.shape[i], rank)
                     for i in range(t)]
        elif init == 'random':
            U = [np.random.randn(data.shape[i], rank) for i in range(t)]
        elif init == 'alt_min':
            U, _ = tensor_completion_alt_min(data, rank, init='svd', max_iter=10)
        else:
            raise Exception("Unrecognized init option " + init)
        # mask = data != 0
        # core, factors = parafac(data, rank, mask=mask, init='random', verbose=True, tol=1e-3)
        # scale_mat = np.diag(core.todense()**(1/t))
        # U = [factors[i].todense() @ scale_mat for i in range(rank)]
    core_values = np.ones(rank)
    #tensor = TensorCPD(U, core_values)
    cost_old = cost(U, data, delta, kappa, beta, epsilon)
    resid_norm = np.sqrt(loss(U, data) / data.nnz)
    if verbosity > 0:
        print("Initial cost: %1.3e" % cost_old)
        print("Initial qnorm_ub: %1.3e" % max_qnorm_ub(U))
        print("|| r || = %1.3e, delta = %1.3e" % (resid_norm, delta / np.sqrt(data.nnz)))
    # alternating minimization
    k = 0
    convergence_crit = np.inf
    cost_arr = np.zeros((max_iter + 1,))
    cost_arr[k] = cost_old
    while convergence_crit > tol and k < max_iter:
        try:
            if sgd:
                indices = np.random.choice(data.nnz, sgd_batch_size)
                data_k = sparse.COO(data.coords[:,indices], data.data[indices], shape=data.shape)
            else:
                data_k = data
            resid_factor = float(data.nnz) / data_k.nnz
            for i in range(t):
                # minimize out ith factor
                if verbosity > 1:
                    print("Entering inner loop for factor %d" % i)
                U_minus_i = copy.deepcopy(U)
                Ui = U_minus_i[i]
                del U_minus_i[i]
                qnorm_factr = max_qnorm_ub(U_minus_i)
                if verbosity > 1:
                    print("qnorm_factr = %1.3e" % qnorm_factr)
                # relaxed cost:
                # .. math:: \mathrm{cost}(U) = g(U) + h(U)
                # .. math:: g(U) = \frac{\kappa}{2} \left( 1 - \frac{\delta}{\max(\|r\|_2, \delta)} \right) \| r \|_2^2
                # .. math:: h(U) = \| U \|_{2,\infty} \mathrm{const}
                def g(Ui):
                    Ut = copy.deepcopy(U_minus_i)
                    Ut.insert(i, Ui)
                    resid_norm = np.sqrt(loss(Ut, data_k))
                    if resid_norm <= (1 + beta / kappa) * delta:
                        mu = kappa / (kappa + beta)
                    else:
                        mu = delta / resid_norm
                    #mu = 0.
                    tik = 0.
                    if epsilon > 0:
                        for Us in Ut:
                            tik += 0.5 * epsilon * np.linalg.norm(Us, 'fro') ** 2
                    return 0.5 * (kappa * (1 - mu)**2 + mu**2 * beta) * resid_norm ** 2 * resid_factor + tik
                def grad_g(Ui):
                    Ut = copy.deepcopy(U_minus_i)
                    Ut.insert(i, Ui)
                    resid_norm = np.sqrt(loss(Ut, data_k))
                    if resid_norm <= (1 + beta / kappa) * delta:
                        mu = kappa / (kappa + beta)
                    else:
                        mu = delta / resid_norm
                    #mu = 0.
                    return kappa * (1 - mu) * sparse_mttkrp(sparse_resid(Ut, data_k), Ut, i) * resid_factor \
                      + epsilon * Ui
                def h(Ui):
                    return qnorm_factr * norm_2_inf(Ui)
                def prox_h(Ui, s):
                    return prox_norm_2_inf(Ui, s * qnorm_factr)
                Ui = acc_prox_grad_method(Ui, g, grad_g, h, prox_h, s0 = 1.,
                                          max_iter=inner_max_iter, tol=inner_tol, gamma=inner_line_factr,
                                          max_line_iter=inner_line_iter)
                U = copy.deepcopy(U_minus_i)
                U.insert(i, Ui)
                if k < 1 and rebalance is True:
                    U = kr_balance_factors(U)
                #tensor = TensorCPD(U, core_values)
            # inner loop finished, check for convergence
            norm_ub_k = max_qnorm_ub(U)
            cost_k = cost(U, data, delta, kappa, beta, epsilon)
            cost_arr[k+1] = cost_k
            resid_norm = np.sqrt(loss(U, data) / data.nnz)
            if verbosity > 1:
                print("\n=============================\nIteration %d complete" % k)
                print("\n\nscaled || r || = %1.3e, delta = %1.3e"
                          % (resid_norm, delta / np.sqrt(data.nnz)))
                print("Max-qnorm upper bound: %1.3e" % norm_ub_k)
                print("Cost function:         %1.3e" % cost_k)
                print("\n=============================\n")
            convergence_crit = abs(cost_k - cost_old)
            cost_old = cost_k
            k += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, exiting early")
            break
    if verbosity > 0:
        print("\nfinished in %d iterations" % k)
        print("\n\nscaled || r || = %1.3e, delta = %1.3e"
                  % (resid_norm, delta / np.sqrt(data.nnz)))
        print("Max-qnorm upper bound: %1.3e" % norm_ub_k)
        print("Cost function:         %.3e" % cost_k)
    return U, cost_arr


def tensor_completion_alt_min(data, rank, init='svd', U0=None,
                                  epsilon=1e-2, tol=1e-4, max_iter=10,
                                  inner_max_iter=30, inner_tol=1e-10, verbosity=0):

    def cost(U, data, epsilon):
        '''
        cost function: least squares + Tikhonov
        '''
        tik = 0.
        if epsilon > 0:
            for Us in U:
                tik += 0.5 * epsilon * np.linalg.norm(Us, 'fro') ** 2
        return 0.5 * loss(U, data) + tik

    assert isinstance(data, sparse.COO), "data should be sparse.COO"
    t = data.ndim
    # initialize factor matrices
    if U0 is not None:
        U = copy.deepcopy(U0)
    else:
        if init == 'svd':
            U = [sparse_unfold_svs(data, i, rank) for i in range(t)]
        elif init == 'svdrand':
            U = [sparse_unfold_svs(data, i, rank) + \
                     0.3 / np.sqrt(data.shape[i]) * np.random.randn(data.shape[i], rank)
                     for i in range(t)]
        elif init == 'random':
            U = [np.random.randn(data.shape[i], rank) for i in range(t)]
        else:
            raise Exception("Unrecognized init option " + init)
        # mask = data != 0
        # core, factors = parafac(data, rank, mask=mask, init='random', verbose=True, tol=1e-3)
        # scale_mat = np.diag(core.todense()**(1/t))
        # U = [factors[i].todense() @ scale_mat for i in range(rank)]
    core_values = np.ones(rank)
    #tensor = TensorCPD(U, core_values)
    cost_old = cost(U, data, epsilon)
    resid_norm = np.sqrt(loss(U, data) / data.nnz)
    if verbosity > 0:
        print("Initial cost: %1.3e" % cost_old)
        print("|| r || = %1.3e\n" % resid_norm)
    # alternating minimization
    k = 0
    convergence_crit = np.inf
    cost_arr = np.zeros((max_iter + 1,))
    cost_arr[k] = cost_old
    while convergence_crit > tol and k < max_iter:
        try:
            for i in range(t):
                # minimize out ith factor
                if verbosity > 1:
                    print("Entering inner loop for factor %d" % i)
                U_minus_i = copy.deepcopy(U)
                Ui = U_minus_i[i].copy()
                del U_minus_i[i]
                Ui_shape = Ui.shape

                def g(Ui):
                    Ut = copy.deepcopy(U_minus_i)
                    Ut.insert(i, Ui)
                    resid_norm = np.sqrt(loss(Ut, data))
                    tik = 0.
                    if epsilon > 0:
                        for Us in Ut:
                            tik += 0.5 * epsilon * np.linalg.norm(Us, 'fro') ** 2
                    return 0.5 * resid_norm ** 2 + tik
                def grad_g(Ui):
                    Ut = copy.deepcopy(U_minus_i)
                    Ut.insert(i, Ui)
                    return sparse_mttkrp(sparse_resid(Ut, data), Ut, i) + epsilon * Ui
                def g_vec(ui):
                    return g(np.reshape(ui, Ui_shape))
                def grad_g_vec(ui):
                    return grad_g(np.reshape(ui, Ui_shape)).flatten()

                ui = fmin_cg(g_vec, Ui.flatten(), grad_g_vec, gtol=inner_tol, maxiter=inner_max_iter, disp=0)
                Ui = np.reshape(ui, Ui_shape)
                U = copy.deepcopy(U_minus_i)
                U.insert(i, Ui)
            # inner loop finished, check for convergence
            norm_ub_k = max_qnorm_ub(U)
            cost_k = cost(U, data, epsilon)
            cost_arr[k+1] = cost_k
            resid_norm = np.sqrt(loss(U, data) / data.nnz)
            if verbosity > 1:
                print("\n=============================\nIteration %d complete" % k)
                print("|| resid || = %1.3e" % resid_norm)
                print("Cost :        %1.3e" % cost_k)
                print("=============================\n")
            convergence_crit = abs(cost_k - cost_old) / abs(cost_old)
            cost_old = cost_k
            k += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, exiting early")
            break            
    if verbosity > 0:
        print("\nfinished in %d iterations" % k)
        print("convergence criterion: %.3e" % convergence_crit)
        print("|| resid || = %1.3e" % resid_norm)
        print("Cost :        %1.3e" % cost_k)
    return U, cost_arr
