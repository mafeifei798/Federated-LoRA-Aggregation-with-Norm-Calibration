"""Historical variable-projection FedRPCA functions, extracted without formula edits.

This legacy solver is not a substitute for the current PCP/TN benchmark.
Rank is an explicit fixed input; adaptive kappa/beta is not adaptive LoRA rank.
"""
from __future__ import annotations

import numpy as np
import numpy.linalg as npla

def soft_threshold(X: np.ndarray, kappa: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - kappa, 0.0)

def truncated_svd_rank_r(M: np.ndarray, rank: int):
    U, s, Vt = npla.svd(M, full_matrices=False)
    r = min(rank, len(s))
    return U[:, :r], s[:r], Vt[:r, :]

def _robust_scale_rows(X: np.ndarray):
    row_med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - row_med), axis=1, keepdims=True)
    mad = np.where(mad < 1e-8, 1e-8, mad)
    Xn = (X - row_med) / mad
    return Xn, row_med, mad

def _clean_matrix(X: np.ndarray, clip_q: float = 0.999):
    Xc = X.copy()
    d, n = Xc.shape
    for i in range(d):
        row = Xc[i]
        lo = np.quantile(row, 1.0 - clip_q) if n > 1 else row.min()
        hi = np.quantile(row, clip_q) if n > 1 else row.max()
        if np.isfinite(lo) and np.isfinite(hi):
            Xc[i] = np.clip(row, lo, hi)
    Xc[~np.isfinite(Xc)] = 0.0
    return Xc

def _svd_soft_rpca(X: np.ndarray, tau: float):
    U, s, Vt = npla.svd(X, full_matrices=False)
    s_shrink = np.maximum(s - tau, 0.0)
    L = (U * s_shrink) @ Vt
    S = X - L
    return L, S

def robust_rpca_vp(X: np.ndarray, rank: int, kappa: float | None = None,
                   max_iter: int = 200, tol: float = 1e-6, continuation: bool = True,
                   kappa_mult: float = 1.0, max_restarts: int = 1, verbose: bool = False):
    # Preprocess: cleaning + row robust scaling
    Xc = _clean_matrix(X)
    Xn, row_med, mad = _robust_scale_rows(Xc)
    d, n = Xn.shape
    # Adaptive kappa
    if kappa is None:
        base = np.median(np.abs(Xn))
        kappa = 0.8 * base if base > 0 else 1e-3
    kappa = float(kappa) * float(kappa_mult)

    best = None
    for restart in range(max_restarts):
        L = np.zeros_like(Xn)
        S = np.zeros_like(Xn)
        prev_obj = np.inf
        kappa_now = kappa * (3.0 if continuation else 1.0)
        for it in range(max_iter):
            # S-step
            S = soft_threshold(Xn - L, kappa_now)
            # L-step
            try:
                U, s, Vt = truncated_svd_rank_r(Xn - S, rank)
                L = (U * s) @ Vt
            except Exception:
                # Fallback: nuclear-norm soft-thresholding
                L, S_alt = _svd_soft_rpca(Xn, tau=kappa_now)
                # S is still obtained by soft-thresholding
            resid = Xn - L - S
            obj = 0.5 * np.sum(resid * resid) + kappa_now * np.sum(np.abs(S))
            if verbose and (it % 10 == 0):
                print(f'[RPCA-VP] r={restart} it={it} obj={obj:.6e} kn={kappa_now:.3e}')
            if abs(prev_obj - obj) / (prev_obj + 1e-12) < tol:
                break
            prev_obj = obj
            # Continuation: gradually decrease kappa to improve sparsity accuracy
            if continuation:
                kappa_now = max(kappa, 0.9 * kappa_now)
        # De-normalize
        L_denorm = L * mad + row_med
        S_denorm = S * mad
        resid0 = Xc - L_denorm - S_denorm
        obj0 = 0.5 * np.sum(resid0 * resid0) + kappa * np.sum(np.abs(S_denorm))
        if (best is None) or (obj0 < best[0]):
            best = (obj0, L_denorm, S_denorm)
    _, L_best, S_best = best
    # Final cleanup
    L_best[~np.isfinite(L_best)] = 0.0
    S_best[~np.isfinite(S_best)] = 0.0
    return L_best, S_best

def _adaptive_beta_sparse(S: np.ndarray) -> float:
    # Adaptive scaling based on sparsity
    sparsity = np.mean(np.isclose(S, 0.0))
    return float(max(0.2, min(0.8, 1.0 - 0.5 * sparsity)))

def _clip_norm(vec: np.ndarray, max_norm: float = 5.0) -> np.ndarray:
    n = npla.norm(vec) + 1e-12
    if n > max_norm:
        return vec * (max_norm / n)
    return vec

def fedrpca_aggregate(deltas: list[np.ndarray], rank: int = 2, kappa: float | None = None,
                      beta_sparse: float | None = None, clip_norm_val: float = 10.0) -> np.ndarray:
    X = np.stack(deltas, axis=1)  # (d, n)
    # Cleaning + robust scaling
    Xc = _clean_matrix(X)
    Xn, row_med, mad = _robust_scale_rows(Xc)
    # Full RPCA-VP
    Ln, Sn = robust_rpca_vp(Xn, rank=rank, kappa=kappa, max_iter=200, tol=1e-6, continuation=True)
    # De-standardize
    L = Ln * mad + row_med
    S = Sn * mad
    # Adaptive coefficient for sparse part
    if beta_sparse is None:
        beta_sparse = _adaptive_beta_sparse(S)
    # Combine
    L_mean = np.mean(L, axis=1)
    S_mean = np.mean(S, axis=1)
    agg = L_mean + float(beta_sparse) * S_mean
    # Aggregation clipping
    agg = _clip_norm(agg, max_norm=clip_norm_val)
    return agg
