import numpy as np
from scipy.sparse.linalg import svds
from numpy_groupies import aggregate # for accumarray type functionality
import sparse

def sparse_unfold(data, mode):
    '''
    Unfolding of a sparse tensor
    '''
    data = data.copy()
    # first step: swap axis with first
    if mode != 0:
        row_mode = data.coords[mode, :]
        data.coords[mode, :] = data.coords[0, :]
        data.coords[0, :] = row_mode
    # second step: reshape
    return data.reshape((data.shape[mode], -1))

#@jit(nopython=False, parallel=True)
def sparse_mttkrp(X, U, mode):
    '''
    Matricized (sparse) tensor times Khatri-Rao product of matrices

    Inspired by code from TensorToolbox and slide set by Kolda:
    https://www.osti.gov/servlets/purl/1146123
    '''
    t = X.ndim
    if len(U) != t:
        raise Exception("Factor list is wrong length")
    if mode == 0:
        R = U[1].shape[1]
    else:
        R = U[0].shape[1]
    V = np.zeros((X.shape[mode], R))
    for r in range(R):
        # Z = (U[i][:,r] for i in [x for x in range(t) if x != n])
        # V[:, r] = sparse_ttv(X, Z, n)
        data = X.data.copy()
        for i in [x for x in range(t) if x != mode]:
            data *= U[i][X.coords[i, :], r]
        V[:, r] = aggregate(X.coords[mode, :], data, func="sum", fill_value=0)
    return(V)

#@jit(nopython=False, parallel=True)
def sparse_resid(U, data):
    '''
    Compute residuals between kruskal tensor and sparse data
    '''
    resid_array = kr_get_items(U, data.coords) - data.data
    # def _kr_get_item_coords(coords):
    #     return kr_get_items(U, coords)

    # resid_array = data.copy(deep=True)
    # predictions = map(_kr_get_item_coords, [data.coords[:, i] for i in range(data.nnz)])
    # #predictions = pmap(get_item_coords)([data.coords[:, i] for i in range(data.nnz)])
    # #predictions = [kr_get_item(U, data.coords[:, i]) for i in range(data.nnz)]
    # resid_array.data = np.array(list(predictions)) - resid_array.data
    # for i in range(data.nnz):
    #     pred = kr_get_item(U, data.coords[:, i])
    #     resid_array.data[i] = pred - data.data[i]
    return sparse.COO(data.coords, resid_array, shape=data.shape)

def sparse_unfold_svs(data, mode, nsv):
    '''
    Compute singular vectors of a sparse tensor unfolding
    '''
    A = sparse_unfold(data, mode).to_scipy_sparse()
    u, s, _ = svds(A, nsv, tol=1e-8, return_singular_vectors='u')
    return u

def kr_dot(U1, U2):
    '''
    Hilbert-Schmidt inner product between two kruskal tensors represented by their factors

    Bader and Kolda, 2007. "Efficient MATLAB Computations with Sparse and Factored Tensors".

    .. math:: \langle T_1, T_2 \rangle
    '''
    r1 = U1[0].shape[1]
    r2 = U2[0].shape[1]
    t = len(U1)
    assert t == len(U2), "tensor order mismatch"
    hadamard_prod = np.ones((r1, r2))
    for i in range(t):
        hadamard_prod *= U1[i].T @ U2[i]
    return np.sum(hadamard_prod.flatten())

def kr_get_items(U, coords):
    '''
    Get entries in kruskal tensor by coordinates

    coords : (nmodes x ndata)
    '''
    r = U[0].shape[1]
    t = len(U)
    n_items = coords.shape[1]
    if n_items > 1:
        values = np.zeros((n_items,))
        for r in range(r):
            summand = np.ones((n_items,))
            for i in range(t):
                summand *= U[i][coords[i, :], r]
            values += summand
        return values
    else: 
        row = np.ones(r)
        for k in range(t):
            row *= U[k][coords[k], :]
        return np.sum(row)

def sparse_mttkrp(X, U, mode):
    '''
    Matricized (sparse) tensor times Khatri-Rao product of matrices

    Inspired by code from TensorToolbox and slide set by Kolda:
    https://www.osti.gov/servlets/purl/1146123
    '''
    t = X.ndim
    if len(U) != t:
        raise Exception("Factor list is wrong length")
    if mode == 0:
        R = U[1].shape[1]
    else:
        R = U[0].shape[1]
    V = np.zeros((X.shape[mode], R))
    for r in range(R):
        # Z = (U[i][:,r] for i in [x for x in range(t) if x != n])
        # V[:, r] = sparse_ttv(X, Z, n)
        data = X.data.copy()
        for i in [x for x in range(t) if x != mode]:
            data *= U[i][X.coords[i, :], r]
        V[:, r] = aggregate(X.coords[mode, :], data, func="sum", fill_value=0)
    return(V)
