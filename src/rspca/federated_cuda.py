"""Opt-in float64 CUDA PCP, independent of the frozen NumPy implementation.

This module changes the execution backend, not the PCP objective or iteration.
All QR/SVD, thresholding, iterative reductions and ADMM updates run on the
requested CUDA device. NumPy is used for reference-identical input validation,
normalization/parameter initialization and the final ``RPCAResult`` arrays.
There is no CPU decomposition fallback, approximate SVD, Gram matrix,
mixed precision, tolerance override, or automatic secondary iteration budget.

The batched interface accepts equally shaped matrices (e.g. 24 x 6144 x 50).
Each block has its own power-of-two normalization, mu/rho and first-convergence
iteration. Finished blocks are frozen while the other blocks continue.

``qr`` uses exact reduced QR followed by a full, non-truncated SVD of R;
``direct`` uses a thin SVD of the original matrix. CUDA SVD explicitly selects
cuSOLVER ``gesvd``, never the approximate ``gesvda`` driver. See
https://docs.pytorch.org/docs/2.11/generated/torch.linalg.svd.html and
https://docs.pytorch.org/docs/2.11/generated/torch.linalg.qr.html.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .federated import RPCAResult, _as_finite_float_array, _power_of_two_scale


CUDA_IMPLEMENTATION_VERSION = "pcp_float64_exact_qr_gesvd_batch_v1"
_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class _PreparedBatch:
    observed: np.ndarray
    scales: np.ndarray
    working_mu: np.ndarray
    working_rho: np.ndarray
    lam: float
    tol: float
    max_iter: int


def _prepare_batch(
    matrices: ArrayLike,
    *,
    lam: float | None,
    mu: float | None,
    tol: float,
    max_iter: int,
    rho: float | None,
) -> _PreparedBatch:
    observed = _as_finite_float_array(matrices, name="matrices", ndim=3)
    count, rows, columns = observed.shape
    if min(count, rows, columns) == 0:
        raise ValueError("matrices must contain nonempty, equally shaped matrices")
    if lam is None:
        lam = 1.0 / math.sqrt(max(rows, columns))
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a finite positive number")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be an integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    scales = np.array([_power_of_two_scale(x) for x in observed])
    working_mu = np.ones(count, dtype=np.float64)
    working_rho = np.ones(count, dtype=np.float64)
    for index, (matrix, scale) in enumerate(zip(observed, scales)):
        # Match the reference zero-matrix return before mu/rho validation.
        if not np.any(matrix):
            continue
        working = matrix / scale
        if mu is None:
            working_mu[index] = working.size / (4.0 * float(np.sum(np.abs(working))))
        else:
            if not math.isfinite(mu) or mu <= 0:
                raise ValueError("mu must be a finite positive number")
            working_mu[index] = mu * scale
        if not math.isfinite(working_mu[index]) or working_mu[index] <= 0:
            raise ValueError(
                "mu is not representable after scale-equivalent PCP normalization"
            )
        if rho is None:
            working_rho[index] = 1.0 / working_mu[index]
        else:
            if not math.isfinite(rho) or rho <= 0:
                raise ValueError("rho must be a finite positive number")
            working_rho[index] = rho / scale
        if not math.isfinite(working_rho[index]) or working_rho[index] <= 0:
            raise ValueError(
                "rho is not representable after scale-equivalent PCP normalization"
            )
        if not math.isclose(
            working_rho[index] * working_mu[index],
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("FedRPCA Algorithm 2 requires rho = 1 / mu")
    return _PreparedBatch(
        observed, scales, working_mu, working_rho, float(lam), float(tol), int(max_iter)
    )


def _fro(torch: Any, matrix: Any) -> Any:
    return torch.linalg.vector_norm(matrix, dim=(-2, -1))


def _finite_matrices(torch: Any, matrix: Any) -> Any:
    return torch.isfinite(matrix).all(dim=-1).all(dim=-1)


def _svd_factors(torch: Any, matrix: Any, method: str) -> tuple[Any, Any, Any]:
    """Exact factors on the input device; CPU is used only by private tests."""
    kwargs = {"full_matrices": False}
    if matrix.device.type == "cuda":
        kwargs["driver"] = "gesvd"
    if method == "direct":
        return torch.linalg.svd(matrix, **kwargs)
    if method != "qr":
        raise ValueError("svd_method must be 'qr' or 'direct'")
    transposed = matrix.shape[-2] < matrix.shape[-1]
    tall = matrix.mT if transposed else matrix
    q, r = torch.linalg.qr(tall, mode="reduced")
    small_u, singular_values, small_vt = torch.linalg.svd(r, **kwargs)
    tall_u = q @ small_u
    if transposed:
        return small_vt.mT, singular_values, tall_u.mT
    return tall_u, singular_values, small_vt


def _svd_sanity(torch: Any, matrix: Any, u: Any, s: Any, vt: Any) -> Any:
    """Same 1e-8 orthogonality/energy/reconstruction tests as the reference."""
    count, rows, columns = matrix.shape
    rank = min(rows, columns)
    if u.shape != (count, rows, rank) or s.shape != (count, rank) or vt.shape != (
        count, rank, columns
    ):
        raise FloatingPointError("CUDA SVD returned invalid factor shapes")
    tolerance = 1e-8
    identity = torch.eye(rank, dtype=matrix.dtype, device=matrix.device)
    matrix_norm = _fro(torch, matrix)
    singular_norm = torch.linalg.vector_norm(s, dim=-1)
    # math.isclose(a,b,rel_tol=t,abs_tol=t*b) from the NumPy reference.
    energy_limit = tolerance * torch.maximum(singular_norm.abs(), matrix_norm.abs())
    reconstruction_error = _fro(torch, (u * s.unsqueeze(-2)) @ vt - matrix)
    orthogonality_limit = tolerance * max(1.0, math.sqrt(rank))
    return (
        _finite_matrices(torch, u)
        & torch.isfinite(s).all(dim=-1)
        & _finite_matrices(torch, vt)
        & (s >= 0).all(dim=-1)
        & (u.abs().amax(dim=(-2, -1)) <= 1.0 + tolerance)
        & (vt.abs().amax(dim=(-2, -1)) <= 1.0 + tolerance)
        & (_fro(torch, u.mT @ u - identity) <= orthogonality_limit)
        & (_fro(torch, vt @ vt.mT - identity) <= orthogonality_limit)
        & torch.isfinite(matrix_norm)
        & torch.isfinite(singular_norm)
        & ((singular_norm - matrix_norm).abs() <= energy_limit)
        & torch.isfinite(reconstruction_error)
        & (reconstruction_error <= tolerance * matrix_norm)
    )


def _svt(torch: Any, matrix: Any, threshold: Any, method: str) -> tuple[Any, Any]:
    u, s, vt = _svd_factors(torch, matrix, method)
    sane = _svd_sanity(torch, matrix, u, s, vt)
    shrunk = torch.clamp(s - threshold[:, None], min=0.0)
    result = (u * shrunk.unsqueeze(-2)) @ vt
    sane = (
        sane
        & _finite_matrices(torch, result)
        & (_fro(torch, result) <= _fro(torch, matrix) * (1.0 + 1e-8))
    )
    return result, sane


def _stable_norm(torch: Any, matrix: Any, *, l1: bool) -> Any:
    magnitude = matrix.abs().amax(dim=(-2, -1))
    denominator = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    scaled = matrix / denominator[:, None, None]
    norm = scaled.abs().sum(dim=(-2, -1)) if l1 else _fro(torch, scaled)
    return magnitude * norm


def _solve_torch(
    prepared: _PreparedBatch, *, torch: Any, device: Any, svd_method: str
) -> list[RPCAResult]:
    """Shared kernel, privately CPU-testable; public callers enforce CUDA."""
    if svd_method not in ("qr", "direct"):
        raise ValueError("svd_method must be 'qr' or 'direct'")
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
    # Division happens before upload to preserve the reference's exact scaling
    # even for subnormal host inputs; no CPU matrix decomposition is performed.
    working = tensor(np.ascontiguousarray(prepared.observed / prepared.scales[:, None, None]))
    count, rows, columns = working.shape
    mu = tensor(prepared.working_mu)
    rho = tensor(prepared.working_rho)
    scales = tensor(prepared.scales)
    frobenius = _fro(torch, working)
    nonzero = working.abs().amax(dim=(-2, -1)) > 0
    if bool((nonzero & (~torch.isfinite(frobenius) | (frobenius <= 0))).any()):
        raise FloatingPointError("normalized PCP matrix has an invalid Frobenius norm")
    denominator = torch.where(nonzero, frobenius, torch.ones_like(frobenius))
    dual = torch.zeros_like(working)
    low_rank = torch.zeros_like(working)
    sparse = torch.zeros_like(working)
    residual = torch.zeros(count, dtype=torch.float64, device=device)
    iterations = torch.zeros(count, dtype=torch.int64, device=device)
    active = nonzero.clone()
    converged = ~nonzero
    dual_limit = prepared.lam + 64.0 * _EPS * max(1.0, prepared.lam)
    if bool(active.any()):
        for iteration in range(1, prepared.max_iter + 1):
            new_low, sane = _svt(
                torch, working - sparse + rho[:, None, None] * dual, rho, svd_method
            )
            sparse_input = working - new_low + rho[:, None, None] * dual
            new_sparse = torch.sign(sparse_input) * torch.clamp(
                sparse_input.abs() - (rho * prepared.lam)[:, None, None], min=0.0
            )
            primal = working - new_low - new_sparse
            new_residual = _fro(torch, primal) / denominator
            new_dual = dual + mu[:, None, None] * primal
            sane = (
                sane
                & torch.isfinite(new_residual)
                & _finite_matrices(torch, new_dual)
                & (new_dual.abs().amax(dim=(-2, -1)) <= dual_limit)
            )
            mask = active[:, None, None]
            low_rank = torch.where(mask, new_low, low_rank)
            sparse = torch.where(mask, new_sparse, sparse)
            dual = torch.where(mask, new_dual, dual)
            residual = torch.where(active, new_residual, residual)
            iterations = torch.where(active, iteration, iterations)
            newly_converged = active & (new_residual <= prepared.tol)
            converged |= newly_converged
            valid = ((~active) | sane).all()
            active = active & ~newly_converged
            # One tiny status transfer per iteration, not per block. Matrix
            # computation and factorization remain entirely on this device.
            valid_host, done_host = torch.stack((valid, ~active.any())).tolist()
            if not valid_host:
                raise FloatingPointError(
                    f"CUDA PCP SVD/proximal/finite/dual sanity failed at iteration {iteration}"
                )
            if done_host:
                break

    feasible_upper = prepared.lam * _stable_norm(torch, working, l1=True)
    objective_lower = _stable_norm(torch, low_rank, l1=False) + prepared.lam * _stable_norm(
        torch, sparse, l1=True
    )
    objective_slack = max(1e-6, 10.0 * prepared.tol)
    bad_objective = converged & (
        ~torch.isfinite(objective_lower)
        | (objective_lower > feasible_upper * (1.0 + objective_slack))
    )
    if bool(bad_objective.any()):
        raise FloatingPointError("PCP rejected false convergence from explosive component cancellation")
    u, singular_values, vt = _svd_factors(torch, low_rank, svd_method)
    if not bool(_svd_sanity(torch, low_rank, u, singular_values, vt).all()):
        raise FloatingPointError("CUDA PCP final rank SVD failed sanity checks")
    rank_tolerance = max(rows, columns) * _EPS * singular_values[:, 0]
    ranks = (singular_values > rank_tolerance[:, None]).sum(dim=-1)
    zero_tolerance = 10.0 * _EPS * working.abs().amax(dim=(-2, -1))
    sparsities = (sparse.abs() <= zero_tolerance[:, None, None]).to(torch.float64).mean(dim=(-2, -1))
    output_low = low_rank * scales[:, None, None]
    output_sparse = sparse * scales[:, None, None]
    if not bool((_finite_matrices(torch, output_low) & _finite_matrices(torch, output_sparse)).all()):
        raise FloatingPointError("PCP decomposition is not representable at the input scale")
    low_arrays = output_low.cpu().numpy()
    sparse_arrays = output_sparse.cpu().numpy()
    residuals = residual.tolist()
    iteration_counts = iterations.tolist()
    convergence = converged.tolist()
    rank_counts = ranks.tolist()
    sparsity_values = sparsities.tolist()
    return [
        RPCAResult(
            low_rank=low_arrays[i].copy(), sparse=sparse_arrays[i].copy(),
            residual=float(residuals[i]), iterations=int(iteration_counts[i]),
            converged=bool(convergence[i]), rank=int(rank_counts[i]),
            sparsity=float(sparsity_values[i]),
        )
        for i in range(count)
    ]


def principal_component_pursuit_admm_batch_cuda(
    matrices: ArrayLike,
    *,
    lam: float | None = None,
    mu: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 1_000,
    rho: float | None = None,
    device: str = "cuda",
    svd_method: str = "qr",
) -> list[RPCAResult]:
    """Solve same-shape independent PCP problems, preserving reference defaults.

    Input shape is ``(blocks, rows, columns)`` and order is preserved. Scalar
    explicit mu/rho apply to each original (unscaled) block, just as repeated
    reference calls would. The final outputs are NumPy float64 ``RPCAResult``
    objects. This call synchronizes before returning. CUDA errors propagate;
    retry policy, secondary caps, fallback and deployment are caller decisions.
    """
    prepared = _prepare_batch(matrices, lam=lam, mu=mu, tol=tol, max_iter=max_iter, rho=rho)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("CUDA PCP requires PyTorch with CUDA support") from error
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("CUDA PCP requires a CUDA device; no CPU fallback is permitted")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PCP requested but CUDA is unavailable; no CPU fallback")
    # Explicitly disable ambient autocast without changing caller global state.
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        return _solve_torch(prepared, torch=torch, device=target, svd_method=svd_method)


def principal_component_pursuit_admm_cuda(
    matrix: ArrayLike,
    *,
    lam: float | None = None,
    mu: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 1_000,
    rho: float | None = None,
    device: str = "cuda",
    svd_method: str = "qr",
) -> RPCAResult:
    """Single-matrix CUDA counterpart of ``principal_component_pursuit_admm``."""
    observed = _as_finite_float_array(matrix, name="matrix", ndim=2)
    return principal_component_pursuit_admm_batch_cuda(
        observed[None, ...], lam=lam, mu=mu, tol=tol, max_iter=max_iter, rho=rho,
        device=device, svd_method=svd_method,
    )[0]


__all__ = [
    "CUDA_IMPLEMENTATION_VERSION",
    "principal_component_pursuit_admm_cuda",
    "principal_component_pursuit_admm_batch_cuda",
]
