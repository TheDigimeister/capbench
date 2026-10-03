"""Assignment LP with arbitrary per-model and global linear resources.

max sum v_ij x_ij  s.t.  sum_j x_ij <= 1,  0 <= x_ij <= allowed_ij,
   per-model rows:  sum_i a_ij x_ij <= cap_j   (one (a, cap) pair per resource)
   global rows:     sum_ij g_ij x_ij <= G
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack


def assign_lp(values, per_model=(), global_=(), allowed=None):
    n, m = values.shape
    cols = np.arange(n * m)
    model_of = cols % m
    rows, b = [], []
    for a, cap in per_model:
        a = np.broadcast_to(np.asarray(a, float), (n, m)).ravel()
        cap = np.asarray(cap, float)
        keep = np.isfinite(cap)
        A = csr_matrix((a, (model_of, cols)), shape=(m, n * m))[np.flatnonzero(keep)]
        rows.append(A)
        b.append(cap[keep])
    for g, G in global_:
        rows.append(csr_matrix(np.asarray(g, float).reshape(1, -1)))
        b.append([float(G)])
    rows.append(csr_matrix((np.ones(n * m), (cols // m, cols)), shape=(n, n * m)))
    b.append(np.ones(n))
    ub = np.ones(n * m) if allowed is None else np.asarray(allowed, float).ravel()
    res = linprog(-np.asarray(values, float).ravel(), A_ub=vstack(rows).tocsr(),
                  b_ub=np.concatenate([np.ravel(x) for x in b]),
                  bounds=np.column_stack([np.zeros(n * m), ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return -res.fun, res.x.reshape(n, m)
