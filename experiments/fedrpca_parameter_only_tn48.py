"""Parameter-only TN48 ablation, not a new algorithm or accuracy result.

Reuse one original FR48 solve on the actual client endpoints. Preserve its
factor directions, replace unguarded norms by twice the raw uniform mean
delta norm, and keep the uniform classifier. No model/data/feedback API.
Production arithmetic is CUDA FP64; CPU is limited to explicit tiny tests.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np
import torch

METHOD = "fedrpca-parameter-only-tn48"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_state(state):
    require(isinstance(state, Mapping) and bool(state), "nonempty parameter state")
    for key, value in state.items():
        x = np.asarray(value)
        require(isinstance(key, str) and bool(key) and x.size > 0
                and x.dtype in (np.dtype('float32'), np.dtype('float64'))
                and bool(np.isfinite(x).all()), "finite FP32/FP64 parameters")


@torch.no_grad()
def norm_map(observed, sparse, original, *, device="cuda:0", toy=False):
    """Exact existing TN rule; no learned coefficient or task-loss selection."""
    arrays = [np.asarray(v) for v in (observed, sparse, original)]
    m, s, u = arrays
    require(type(toy) is bool and m.ndim == 3 and min(m.shape) > 0
            and s.shape == m.shape and u.shape == m.shape[:2]
            and all(x.dtype == np.float64 and np.isfinite(x).all() for x in arrays),
            "finite FP64 block inventory")
    selected = torch.device(device)
    require(selected.type == 'cuda' or (toy and selected.type == 'cpu'),
            "CUDA required for production")
    require(not toy or sum(x.size for x in arrays) <= 16384,
            "tiny explicit CPU synthetic budget")
    if not toy:
        require(m.shape == (48, 3072, 50), "native 48 factors by 50 clients")
    # Copy to avoid mutating caller-owned numpy buffers, even on CPU.
    mm, ss, uu = [torch.tensor(x, dtype=torch.float64, device=selected) for x in arrays]
    updates, details = [], []
    for i in range(len(mm)):
        maximum = float(mm[i].abs().max())
        scale = math.ldexp(1., math.frexp(maximum)[1] - 1) if maximum else 1.
        require(math.isfinite(scale) and scale > 0, "representable dyadic scale")
        mean = (mm[i] / scale).mean(dim=1)
        sparse_mean = (ss[i] / scale).mean(dim=1)
        direction = uu[i] / scale
        require(bool(torch.isfinite(mean).all() and torch.isfinite(sparse_mean).all()
                     and torch.isfinite(direction).all()), "finite normalized blocks")
        sn = float(torch.linalg.vector_norm(sparse_mean))
        un = float(torch.linalg.vector_norm(direction))
        radius = float(torch.linalg.vector_norm(2. * mean))
        require(all(math.isfinite(v) for v in (sn, un, radius)), "finite normalized norms")
        guarded = bool(sn <= np.finfo(np.float64).eps)
        if guarded:
            value = uu[i].clone()  # exact stored original, not a reparameterization
            multiplier = 1.
        elif un == 0.:
            require(radius == 0., "nonzero radius with undefined FR direction")
            value, multiplier = uu[i].clone(), 1.
        else:
            multiplier = radius / un
            value = direction * multiplier * scale
        require(math.isfinite(multiplier) and bool(torch.isfinite(value).all()),
                "finite TN output")
        updates.append(value)
        details.append(dict(block=i, canonical_guard=guarded,
                            multiplier=multiplier, sparse_mean_norm=sn,
                            original_norm=un, target_norm=un if guarded else radius))
    return torch.stack(updates).cpu().numpy(), details


def assemble(parent, local_states, fr_state, sparse, *, adapter_pairs,
             classifier_keys, device="cuda:0", toy=False):
    """Server reads only uploaded parameters and original RPCA components.

    local_states are post-local endpoints, not gradients. Their count weighting
    remains uniform, as in the original comparison. Client training happens
    outside this module. Caller must authenticate the original FR48 solve.
    """
    finite_state(parent); finite_state(fr_state)
    pairs, head = tuple(map(tuple, adapter_pairs)), tuple(classifier_keys)
    require(pairs and all(len(p) == 2 for p in pairs), "paired adapter registry")
    keys = tuple(k for pair in pairs for k in pair)
    require(len(set(keys)) == len(keys) and head and len(set(head)) == len(head)
            and not set(keys).intersection(head)
            and set(parent) == set(fr_state) == set(keys) | set(head),
            "complete disjoint adapter and classifier inventory")
    require(isinstance(local_states, (list, tuple)) and bool(local_states),
            "client parameter endpoints")
    require(toy or (len(pairs) == 24 and len(local_states) == 50),
            "native 24 pairs and 50 clients")
    for state in (fr_state, *local_states):
        finite_state(state)
        require(set(state) == set(parent)
                and all(state[k].shape == parent[k].shape
                        and state[k].dtype == parent[k].dtype for k in parent),
                "same parameter shapes and dtypes")
    for ak, bk in pairs:
        a, b = parent[ak], parent[bk]
        require(a.ndim == b.ndim == 2 and b.shape[1] == a.shape[0], "valid A/B pair")
    lengths = {parent[k].size for k in keys}
    require(len(lengths) == 1, "equal factor-block dimensions")
    observed = np.stack([np.column_stack([
        (np.asarray(x[k], dtype=np.float64) - np.asarray(parent[k], dtype=np.float64)).ravel()
        for x in local_states]) for k in keys])
    original = np.stack([(np.asarray(fr_state[k], dtype=np.float64)
                         - np.asarray(parent[k], dtype=np.float64)).ravel() for k in keys])
    for k in head:
        expected = np.mean(np.stack([np.asarray(x[k], dtype=np.float64)
                                     for x in local_states]), axis=0).astype(fr_state[k].dtype)
        require(np.array_equal(fr_state[k], expected), "original uniform classifier required")
    update, blocks = norm_map(observed, np.asarray(sparse), original, device=device, toy=toy)
    out = {k: np.asarray(v).copy() for k, v in fr_state.items()}
    for j, k in enumerate(keys):
        if not blocks[j]['canonical_guard']:
            out[k] = (np.asarray(parent[k], dtype=np.float64)
                      + update[j].reshape(parent[k].shape)).astype(parent[k].dtype)
    finite_state(out)
    require(all(np.array_equal(out[k], fr_state[k]) for k in head), "unchanged classifier")
    return out, dict(method=METHOD, blocks=blocks, n_clients=len(local_states),
                     weighting='uniform_clients', extra_server_model_forward=0,
                     extra_server_model_backward=0, extra_optimizer_steps=0,
                     extra_pcp_solves=0, feedback_rows=0,
                     extra_aggregation_compute=True, equal_flops_claim=False,
                     new_accuracy_result=False, novelty_claim=False)


def from_fr48(fr_aggregate, parent, local_states, *, adapter_pairs,
              classifier_keys, device="cuda:0", toy=False):
    """Native parameter-only hook: one pinned original FR48 callback.

    The callback must return state, snapshot, diagnostics like native context.fr.
    No GTN module, backend, dataset or feedback callback is accepted.
    """
    require(callable(fr_aggregate), "original FR48 parameter-only callback")
    base = fr_aggregate(parent, local_states)
    bank, diag = base.snapshot, base.diagnostics['cuda_aggregation']
    keys = tuple(k for p in adapter_pairs for k in p)
    require(diag['blocks'] == diag['primary_calls'] == len(keys)
            and diag['cpu_pcp_solves'] == 0, "one original CUDA decomposition")
    expected = np.stack([np.column_stack([
        (np.asarray(x[k], dtype=np.float64) - np.asarray(parent[k], dtype=np.float64)).ravel()
        for x in local_states]) for k in keys])
    require(np.array_equal(bank['observed'], expected), "RPCA bank from actual client parameters")
    require(bank['low_rank'].shape == bank['sparse'].shape == expected.shape
            and all(np.isfinite(bank[k]).all() for k in ('low_rank', 'sparse')),
            "finite original decomposition components")
    residual = np.linalg.norm(expected - bank['low_rank'] - bank['sparse'])
    require(math.isfinite(residual)
            and residual <= 2e-7 * max(np.linalg.norm(expected), np.finfo(float).tiny),
            "certified RPCA reconstruction")
    out, report = assemble(parent, local_states, base.state, bank['sparse'],
        adapter_pairs=adapter_pairs, classifier_keys=classifier_keys, device=device, toy=toy)
    report['original_pcp'] = dict(diag)
    return out, report
