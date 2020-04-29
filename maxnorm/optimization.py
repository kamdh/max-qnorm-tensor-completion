import numpy as np
#import jax.numpy as np
from typing import Callable

def hs_dot(A, B):
    return (A*B).flatten().sum()

def prox_grad_method(x: np.ndarray,
                         g: Callable[[np.ndarray], np.float64],
                         grad_g: Callable[[np.ndarray], np.float64],
                         h: Callable[[np.ndarray], np.float64],
                         prox: Callable[[np.ndarray, np.float64], np.float64],
                         tol: np.float64 = 1e-6,
                         max_iter: int = 100,
                         s0: np.float64 = 1,
                         max_line_iter: int = 100,
                         gamma: np.float64 = 0.8) -> np.ndarray:
    u"""Nesterov accelerated proximal gradient method
    https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
    x: initial point
    g: differentiable term in objective function
    grad_g: gradient of g
    h: non-differentiable term in objective function
    prox: proximal operator corresponding to h
    tol: relative tolerance in objective function for convergence
    max_iter: maximum number of proximal gradient steps
    s0: initial step size
    max_line_iter: maximum number of line search steps
    gamma: step size shrinkage rate for line search
    """
    # initialize step size
    s = s0
    # initial objective value
    f = g(x) + h(x)
    print(f'initial objective {f:.6e}', flush=True)
    print(f'initial smooth part {g(x):.6e}', flush=True)
    for k in range(1, max_iter + 1):
        # evaluate differtiable part of objective at current point
        g1 = g(x)
        grad_g1 = grad_g(x)
        # check for errors
        if g1 == 0:
            print("\nZero cost, breaking")
            break
        if not np.all(np.isfinite(grad_g1)):
            raise RuntimeError(f'invalid gradient at iteration {k + 1}: '
                               f'{grad_g1}')
        if np.all(grad_g1 == 0):
            print("\nZero gradient, breaking")
            break
        # store old iterate
        x_old = x
        # Armijo line search
        for line_iter in range(max_line_iter):
            # new point via prox-gradient of momentum point
            x = prox(x - s * grad_g1, s)
            # G_s(q) as in the notes linked above
            G = (1 / s) * (x_old - x)
            # test g(q - sG_s(q)) for sufficient decrease
            if g(x) <= (g1 - s * hs_dot(grad_g1, G) + (s / 2) * hs_dot(G, G)):
                # Armijo satisfied
                break
            else:
                # Armijo not satisfied
                s *= gamma  # shrink step size

        if line_iter == max_line_iter - 1:
            print('\nwarning: line search failed\n', flush=True)
            s = s0
        if not np.all(np.isfinite(x)):
            print(f'\nwarning: x contains invalid values\n', flush=True)
        # terminate if objective function is constant within tolerance
        f_old = f
        f = g(x) + h(x)
        rel_change = np.abs((f - f_old) / f_old)
        print(f'iteration {k}, objective {f:.3e}, '
              f'relative change {rel_change:.3e}', flush=True)
        # print(f'iteration {k}, objective {f:.3e}, '
        #       f'relative change {rel_change:.3e}',
        #       end='        \r', flush=True)
        if rel_change < tol:
            print(f'\nrelative change in objective function {rel_change:.2g} '
                  f'is within tolerance {tol} after {k} iterations',
                  flush=True)
            break
        if k == max_iter:
            print(f'\nmaximum iteration {max_iter} reached with relative '
                  f'change in objective function {rel_change:.2g}', flush=True)
    return x


def acc_prox_grad_method(x: np.ndarray,
                         g: Callable[[np.ndarray], np.float64],
                         grad_g: Callable[[np.ndarray], np.float64],
                         h: Callable[[np.ndarray], np.float64],
                         prox: Callable[[np.ndarray, np.float64], np.float64],
                         tol: np.float64 = 1e-6,
                         max_iter: int = 100,
                         s0: np.float64 = 1,
                         max_line_iter: int = 100,
                         gamma: np.float64 = 0.8) -> np.ndarray:
    u"""Nesterov accelerated proximal gradient method
    https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
    x: initial point
    g: differentiable term in objective function
    grad_g: gradient of g
    h: non-differentiable term in objective function
    prox: proximal operator corresponding to h
    tol: relative tolerance in objective function for convergence
    max_iter: maximum number of proximal gradient steps
    s0: initial step size
    max_line_iter: maximum number of line search steps
    gamma: step size shrinkage rate for line search
    """
    # initialize step size
    s = s0
    # initialize momentum iterate
    q = x
    # initial objective value
    f = g(x) + h(x)
    print(f'initial objective {f:.6e}', flush=True)
    for k in range(1, max_iter + 1):
        # evaluate differtiable part of objective at momentum point
        g1 = g(q)
        if g1 == 0:
            print("\nZero cost, breaking")
            break
        grad_g1 = grad_g(q)
        if not np.all(np.isfinite(grad_g1)):
            raise RuntimeError(f'invalid gradient at iteration {k + 1}: '
                               f'{grad_g1}')
        if np.all(grad_g1 == 0):
            print("\nZero gradient, breaking")
            break
        # store old iterate
        x_old = x
        # Armijo line search
        for line_iter in range(max_line_iter):
            # new point via prox-gradient of momentum point
            x = prox(q - s * grad_g1, s)
            # G_s(q) as in the notes linked above
            G = (1 / s) * (q - x)
            # test g(q - sG_s(q)) for sufficient decrease
            if g(q - s * G) <= (g1 - s * hs_dot(grad_g1, G) + (s / 2) * hs_dot(G, G)):
                # Armijo satisfied
                break
            else:
                # Armijo not satisfied
                s *= gamma  # shrink step size

        # update momentum point
        q = x + ((k - 1) / (k + 2)) * (x - x_old)

        if line_iter == max_line_iter - 1:
            print('\nwarning: line search failed\n', flush=True)
            s = s0
        if not np.all(np.isfinite(x)):
            print(f'\nwarning: x contains invalid values\n', flush=True)
        # terminate if objective function is constant within tolerance
        f_old = f
        f = g(x) + h(x)
        rel_change = np.abs((f - f_old) / f_old)
        # print(f'iteration {k}, objective {f:.3e}, '
        #       f'relative change {rel_change:.3e}', flush=True)
        print(f'iteration {k}, objective {f:.3e}, '
              f'relative change {rel_change:.3e}',
              end='        \r', flush=True)
        if rel_change < tol:
            print(f'\nrelative change in objective function {rel_change:.2g} '
                  f'is within tolerance {tol} after {k} iterations',
                  flush=True)
            break
        if k == max_iter:
            print(f'\nmaximum iteration {max_iter} reached with relative '
                  f'change in objective function {rel_change:.2g}', flush=True)
    return x
