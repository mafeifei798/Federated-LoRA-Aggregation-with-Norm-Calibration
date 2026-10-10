"""Strict server aggregation for the paper-aligned federated LoRA runner.

The functions in this module operate on *post-local trainable states*.  Every
LoRA target is registered explicitly as an ``(A_key, B_key)`` pair, where
``A`` has shape ``(rank, in_features)`` and ``B`` has shape
``(out_features, rank)``.  Remaining trainable classifier parameters are
listed explicitly as ``classifier_keys``.  The two registries must form an
exact partition of the state so a newly trainable parameter cannot be silently
dropped or aggregated with the wrong rule.

All paper methods use uniform client weighting.  FedAvg averages post-local
states; Task Arithmetic, TIES, and FedRPCA aggregate adapter deltas relative to
the supplied global state.  Non-adapter classifier parameters use the same
uniform post-local-state mean for every method.  FedRPCA deliberately calls the
canonical implementation with ``adaptive_beta=True``, ``sample_counts=None``,
and ``beta_max=None``: this selects the paper's unbounded per-block inverse
sparse-energy schedule rather than a sample-weighted or clipped extension.
The separately named CA-RMS extension changes only that beta estimator to the
weighted RMS/Frobenius energy ratio; it does not change PCP or client training.
The opt-in layer-joint variant applies the paper inverse-energy rule to
same-layer Q/V concatenations, separately for A and B factors. It does not
change client training.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .federated import (
    BlockDiagnostics,
    CancellationAwareBlockDiagnostics,
    coordinate_median_aggregate,
    coordinate_trimmed_mean_aggregate,
    fedavg_aggregate,
    fedrpca_aggregate,
    median_norm_clipped_fedavg_aggregate,
    sample_weighted_fedavg_aggregate,
    task_arithmetic_aggregate,
    ties_aggregate,
    ties_reference_aggregate,
)

FloatArray = NDArray[np.float64]
TrainableState = Mapping[str, ArrayLike]
AdapterPair = tuple[str, str]

METHOD_FEDAVG = "fedavg"
METHOD_TASK_ARITHMETIC = "task-arithmetic"
METHOD_TIES = "ties"
METHOD_FEDRPCA = "fedrpca"
METHOD_CA_RMS_FEDRPCA = "ca-rms-fedrpca"
METHOD_LAYER_JOINT_FEDRPCA = "layer-joint-fedrpca"
METHOD_SAMPLE_WEIGHTED_FEDAVG = "sample-weighted-fedavg"
METHOD_MEDIAN_NORM_CLIPPED_FEDAVG = "median-norm-clipped-fedavg"
METHOD_COORDINATE_TRIMMED_MEAN = "coordinate-trimmed-mean"
METHOD_COORDINATE_MEDIAN = "coordinate-median"

ALL_SERVER_METHODS = (
    METHOD_FEDAVG,
    METHOD_TASK_ARITHMETIC,
    METHOD_TIES,
    METHOD_FEDRPCA,
)

# Keep the canonical paper-method registry stable for existing formal runners.
# Extensions are opt-in through this broader server boundary.
ALL_SUPPORTED_SERVER_METHODS = ALL_SERVER_METHODS + (
    METHOD_CA_RMS_FEDRPCA,
    METHOD_LAYER_JOINT_FEDRPCA,
    METHOD_SAMPLE_WEIGHTED_FEDAVG,
    METHOD_MEDIAN_NORM_CLIPPED_FEDAVG,
    METHOD_COORDINATE_TRIMMED_MEAN,
    METHOD_COORDINATE_MEDIAN,
)

PAPER_RPCA_PRIMARY_MAX_ITER = 1_000
PAPER_RPCA_SECONDARY_MAX_ITER = 5_000
GPT2_LAYER_COUNT = 12
COORDINATE_TRIM_FRACTION = 0.1

_GPT2_QV_A_KEY = re.compile(
    r"^transformer\.h\.(?P<layer>0|[1-9][0-9]*)\.attn\.c_attn\."
    r"(?P<projection>[qv])\.lora_A\.weight$"
)


@dataclass(frozen=True)
class PaperServerResult:
    """Aggregated trainable state and JSON-serialisable diagnostics."""

    state: dict[str, FloatArray]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _LayerJointGroup:
    """One deterministic same-layer Q/V concatenation used by the extension."""

    key: str
    layer: int
    factor: str
    q_key: str
    v_key: str
    q_shape: tuple[int, int]
    v_shape: tuple[int, int]

    @property
    def q_size(self) -> int:
        return int(np.prod(self.q_shape))

    @property
    def v_size(self) -> int:
        return int(np.prod(self.v_shape))

    @property
    def size(self) -> int:
        return self.q_size + self.v_size


def aggregate_paper_state(
    method: str,
    global_state: TrainableState,
    local_states: Sequence[TrainableState],
    *,
    adapter_pairs: Sequence[AdapterPair],
    classifier_keys: Sequence[str],
    sample_counts: Sequence[int] | None = None,
    task_beta: float = 1.0,
    ties_density: float = 0.2,
    ties_variant: str = "legacy",
    rpca_tol: float = 1e-7,
    rpca_max_iter: int = 1_000,
    rpca_secondary_max_iter: int | None = None,
) -> PaperServerResult:
    """Aggregate one paper-aligned server round with strict state accounting.

    Parameters
    ----------
    method:
        One of :data:`ALL_SUPPORTED_SERVER_METHODS`.  The four entries in
        :data:`ALL_SERVER_METHODS` remain the canonical paper methods; every
        additional entry is an explicitly named opt-in extension.
    global_state, local_states:
        The pre-round global trainable state and each client's post-local
        trainable state.  Inputs are copied and never mutated.
    adapter_pairs:
        Explicit registry of every Q/V LoRA ``(A_key, B_key)`` pair.
    classifier_keys:
        Explicit registry of all trainable parameters outside those pairs.
        These keys use a uniform mean of post-local states except for the
        explicitly named sample-weighted FedAvg baseline.
    sample_counts:
        Required only for ``sample-weighted-fedavg`` and rejected for every
        other method.  Each count must be a positive integer.
    task_beta, ties_density:
        Method-specific paper-baseline hyperparameters.
    ties_variant:
        ``legacy`` preserves historical numerical behaviour.  The explicitly
        opted-in ``upstream_reference`` uses threshold-inclusive trimming and
        upstream mass-sign zero resolution, with uniform weights and scale one.
    rpca_tol, rpca_max_iter:
        Numerical controls forwarded to canonical per-factor-block FedRPCA.
    rpca_secondary_max_iter:
        Explicit protocol-v1.1 numerical-completion fallback.  When enabled,
        the primary cap must remain 1,000 and the secondary cap must be 5,000.
        A secondary solve starts from the canonical zero initialization and is
        attempted only if the primary solve does not converge.
    """

    if method not in ALL_SUPPORTED_SERVER_METHODS:
        raise ValueError(
            f"unknown paper server method {method!r}; expected one of "
            f"{ALL_SUPPORTED_SERVER_METHODS}"
        )
    if ties_variant not in {"legacy", "upstream_reference"}:
        raise ValueError("unknown TIES aggregation variant")
    if ties_variant != "legacy" and method != METHOD_TIES:
        raise ValueError("the TIES aggregation variant is valid only for ties")
    global_arrays, locals_arrays, keys = _normalise_states(global_state, local_states)
    pairs, classifier = _validate_registry(
        global_arrays,
        adapter_pairs=adapter_pairs,
        classifier_keys=classifier_keys,
    )
    adapter_keys = tuple(key for pair in pairs for key in pair)
    deltas = [
        {key: local[key] - global_arrays[key] for key in adapter_keys}
        for local in locals_arrays
    ]

    # Robust adapter-only baselines share this uniform head policy.  The
    # explicitly named classical FedAvg branch below replaces the complete
    # state, including the head, with its sample-count-weighted mean.
    output = (
        {}
        if method == METHOD_SAMPLE_WEIGHTED_FEDAVG
        else {
            key: np.asarray(
                np.mean(np.stack([local[key] for local in locals_arrays]), axis=0),
                dtype=np.float64,
            )
            for key in classifier
        }
    )
    fedrpca_blocks: dict[str, object] = {}
    method_parameters: dict[str, object] = {}
    layer_joint_groups: tuple[_LayerJointGroup, ...] = ()
    robust_aggregation: dict[str, object] | None = None
    weighting = "uniform_clients"
    classifier_aggregation = "uniform_mean_post_local_state"

    if sample_counts is not None and method != METHOD_SAMPLE_WEIGHTED_FEDAVG:
        raise ValueError(
            "sample_counts is valid only for sample-weighted-fedavg; other "
            "methods have frozen weighting policies"
        )

    if method == METHOD_FEDAVG:
        mean_state = fedavg_aggregate(
            [{key: local[key] for key in adapter_keys} for local in locals_arrays],
            sample_counts=None,
        )
        output.update(mean_state)
        adapter_aggregation = "uniform_mean_post_local_factors"
    elif method == METHOD_SAMPLE_WEIGHTED_FEDAVG:
        if sample_counts is None:
            raise ValueError(
                "sample-weighted-fedavg requires explicit positive integer "
                "sample_counts"
            )
        weighted = sample_weighted_fedavg_aggregate(
            locals_arrays, sample_counts=sample_counts
        )
        output = dict(weighted.update)
        robust_aggregation = dict(weighted.diagnostics)
        adapter_aggregation = "sample_weighted_mean_post_local_factors"
        classifier_aggregation = "sample_weighted_mean_post_local_state"
        weighting = "sample_counts"
    elif method == METHOD_MEDIAN_NORM_CLIPPED_FEDAVG:
        clipped = median_norm_clipped_fedavg_aggregate(deltas)
        output.update(
            {key: global_arrays[key] + clipped.update[key] for key in adapter_keys}
        )
        robust_aggregation = dict(clipped.diagnostics)
        adapter_aggregation = (
            "global_plus_full_adapter_median_norm_clipped_uniform_mean_delta"
        )
        method_parameters["tunable_hyperparameters"] = False
    elif method == METHOD_COORDINATE_TRIMMED_MEAN:
        trimmed = coordinate_trimmed_mean_aggregate(
            deltas, trim_fraction=COORDINATE_TRIM_FRACTION
        )
        output.update(
            {key: global_arrays[key] + trimmed.update[key] for key in adapter_keys}
        )
        robust_aggregation = dict(trimmed.diagnostics)
        adapter_aggregation = "global_plus_per_factor_coordinate_trimmed_mean_delta"
        method_parameters["trim_fraction_each_tail"] = COORDINATE_TRIM_FRACTION
    elif method == METHOD_COORDINATE_MEDIAN:
        median = coordinate_median_aggregate(deltas)
        output.update(
            {key: global_arrays[key] + median.update[key] for key in adapter_keys}
        )
        robust_aggregation = dict(median.diagnostics)
        adapter_aggregation = "global_plus_per_factor_coordinate_median_delta"
    elif method == METHOD_TASK_ARITHMETIC:
        _validate_positive_finite(task_beta, name="task_beta")
        merged_delta = task_arithmetic_aggregate(
            deltas,
            beta=float(task_beta),
            sample_counts=None,
        )
        output.update(
            {key: global_arrays[key] + merged_delta[key] for key in adapter_keys}
        )
        adapter_aggregation = "global_plus_beta_uniform_mean_adapter_delta"
        method_parameters["task_beta"] = float(task_beta)
    elif method == METHOD_TIES:
        if not math.isfinite(ties_density) or not 0.0 < ties_density <= 1.0:
            raise ValueError("ties_density must be a finite number in (0, 1]")
        merged_delta = (
            ties_reference_aggregate(deltas, density=float(ties_density))
            if ties_variant == "upstream_reference"
            else ties_aggregate(
                deltas,
                density=float(ties_density),
                sample_counts=None,
            )
        )
        output.update(
            {key: global_arrays[key] + merged_delta[key] for key in adapter_keys}
        )
        adapter_aggregation = "global_plus_uniform_ties_adapter_delta"
        method_parameters["ties_density"] = float(ties_density)
        if ties_variant == "upstream_reference":
            adapter_aggregation = "global_plus_uniform_ties_reference_adapter_delta"
            method_parameters.update(
                ties_variant=ties_variant,
                ties_density_semantics="retained_fraction_assumption",
                ties_trim="global_adapter_delta_kth_threshold_inclusive",
                ties_zero_sign="global_majority_then_negative_selector",
                ties_scale=1.0,
            )
    elif method in (
        METHOD_FEDRPCA,
        METHOD_CA_RMS_FEDRPCA,
        METHOD_LAYER_JOINT_FEDRPCA,
    ):
        _validate_positive_finite(rpca_tol, name="rpca_tol")
        if (
            isinstance(rpca_max_iter, bool)
            or not isinstance(rpca_max_iter, (int, np.integer))
            or int(rpca_max_iter) <= 0
        ):
            raise ValueError("rpca_max_iter must be a positive integer")
        if rpca_secondary_max_iter is not None:
            if isinstance(rpca_secondary_max_iter, bool) or not isinstance(
                rpca_secondary_max_iter, (int, np.integer)
            ):
                raise ValueError("rpca_secondary_max_iter must be an integer")
            if int(rpca_max_iter) != PAPER_RPCA_PRIMARY_MAX_ITER:
                raise ValueError(
                    "protocol-v1.1 requires rpca_max_iter=1000 when the "
                    "secondary fallback is enabled"
                )
            if int(rpca_secondary_max_iter) != PAPER_RPCA_SECONDARY_MAX_ITER:
                raise ValueError("protocol-v1.1 requires rpca_secondary_max_iter=5000")
        extension_kwargs: dict[str, object] = {}
        if method == METHOD_CA_RMS_FEDRPCA:
            extension_kwargs["cancellation_aware_rms"] = True
        if method == METHOD_LAYER_JOINT_FEDRPCA:
            layer_joint_groups = _build_layer_joint_groups(pairs, global_arrays)
            robust_deltas = _concatenate_layer_joint_deltas(deltas, layer_joint_groups)
            robust_keys = tuple(group.key for group in layer_joint_groups)
        else:
            robust_deltas = deltas
            robust_keys = adapter_keys
        robust = fedrpca_aggregate(
            robust_deltas,
            adapter_keys=robust_keys,
            adaptive_beta=True,
            beta_max=None,
            sample_counts=None,
            reliability_gated=False,
            rpca_tol=float(rpca_tol),
            rpca_max_iter=int(rpca_max_iter),
            rpca_secondary_max_iter=(
                None
                if rpca_secondary_max_iter is None
                else int(rpca_secondary_max_iter)
            ),
            **extension_kwargs,
        )
        if method == METHOD_LAYER_JOINT_FEDRPCA and set(robust.update) != set(
            robust_keys
        ):
            missing = sorted(set(robust_keys) - set(robust.update))
            extra = sorted(set(robust.update) - set(robust_keys))
            raise RuntimeError(
                f"{method} returned invalid update keys; missing={missing}, "
                f"extra={extra}"
            )
        if method == METHOD_LAYER_JOINT_FEDRPCA and set(robust.diagnostics) != set(
            robust_keys
        ):
            missing = sorted(set(robust_keys) - set(robust.diagnostics))
            extra = sorted(set(robust.diagnostics) - set(robust_keys))
            raise RuntimeError(
                f"{method} returned invalid diagnostic keys; missing={missing}, "
                f"extra={extra}"
            )
        fedrpca_blocks = {key: asdict(robust.diagnostics[key]) for key in robust_keys}
        policies = {str(block["beta_policy"]) for block in fedrpca_blocks.values()}
        expected_policy = (
            "paper_inverse_energy"
            if method in (METHOD_FEDRPCA, METHOD_LAYER_JOINT_FEDRPCA)
            else "cancellation_aware_rms_energy"
        )
        if policies != {expected_policy}:
            raise RuntimeError(
                f"{method} must report {expected_policy} for every adapter block, "
                f"got {sorted(policies)}"
            )
        rejected_blocks: dict[str, list[str]] = {}
        for key in robust_keys:
            diagnostic = robust.diagnostics[key]
            reasons: list[str] = []
            if not diagnostic.converged:
                reasons.append("not_converged")
            if not math.isfinite(diagnostic.residual) or diagnostic.residual > float(
                rpca_tol
            ):
                reasons.append("residual_exceeds_tolerance")
            if not math.isfinite(diagnostic.beta_base) or not math.isfinite(
                diagnostic.beta
            ):
                reasons.append("nonfinite_beta")
            if method == METHOD_CA_RMS_FEDRPCA:
                if not isinstance(diagnostic, CancellationAwareBlockDiagnostics):
                    reasons.append("missing_ca_rms_diagnostics")
                else:
                    coherence_values = (
                        diagnostic.observed_coherence,
                        diagnostic.sparse_coherence,
                    )
                    if any(
                        not math.isfinite(value) or value < 0.0 or value > 1.0
                        for value in coherence_values
                    ):
                        reasons.append("invalid_ca_rms_coherence")
                    if diagnostic.coherence_ratio is not None and (
                        not math.isfinite(diagnostic.coherence_ratio)
                        or diagnostic.coherence_ratio < 0.0
                    ):
                        reasons.append("invalid_ca_rms_coherence_ratio")
                    if diagnostic.rms_energy_ratio is not None and (
                        not math.isfinite(diagnostic.rms_energy_ratio)
                        or diagnostic.rms_energy_ratio < 0.0
                    ):
                        reasons.append("invalid_ca_rms_energy_ratio")
                    if diagnostic.sparse_exact_zero_guard and not (
                        diagnostic.sparse_weighted_zero_guard
                    ):
                        reasons.append("inconsistent_ca_rms_zero_guards")
                    if diagnostic.sparse_weighted_zero_guard:
                        if diagnostic.raw_rms_beta is not None or (
                            diagnostic.beta_base != 1.0
                        ):
                            reasons.append("invalid_ca_rms_zero_guard")
                    elif (
                        diagnostic.raw_rms_beta is None
                        or not math.isfinite(diagnostic.raw_rms_beta)
                        or diagnostic.raw_rms_beta < 0.0
                        or not math.isclose(
                            diagnostic.beta_base,
                            diagnostic.raw_rms_beta,
                            rel_tol=1e-12,
                            abs_tol=0.0,
                        )
                    ):
                        reasons.append("invalid_ca_rms_raw_beta")
            if method == METHOD_LAYER_JOINT_FEDRPCA:
                reasons.extend(
                    _layer_joint_fallback_reasons(
                        diagnostic,
                        rpca_max_iter=int(rpca_max_iter),
                        rpca_secondary_max_iter=rpca_secondary_max_iter,
                        input_is_exact_zero=all(
                            not np.any(client[key] != 0.0) for client in robust_deltas
                        ),
                    )
                )
            if reasons:
                rejected_blocks[key] = reasons
        if rejected_blocks:
            acceptance_name = (
                "layer-joint FedRPCA"
                if method == METHOD_LAYER_JOINT_FEDRPCA
                else "paper FedRPCA"
            )
            raise RuntimeError(
                f"{acceptance_name} numerical acceptance failed; refusing to apply "
                f"invalid blocks: {json.dumps(rejected_blocks, sort_keys=True)}"
            )
        if method == METHOD_LAYER_JOINT_FEDRPCA:
            split_updates = _split_layer_joint_updates(
                robust.update,
                layer_joint_groups,
                expected_keys=adapter_keys,
            )
            output.update(
                {key: global_arrays[key] + split_updates[key] for key in adapter_keys}
            )
            adapter_aggregation = (
                "global_plus_per_layer_joint_qv_factor_" "paper_adaptive_fedrpca_delta"
            )
        else:
            output.update(
                {key: global_arrays[key] + robust.update[key] for key in adapter_keys}
            )
            adapter_aggregation = (
                "global_plus_per_factor_block_paper_adaptive_fedrpca_delta"
                if method == METHOD_FEDRPCA
                else "global_plus_per_factor_block_ca_rms_fedrpca_delta"
            )
        method_parameters.update(
            {
                "adaptive_beta": True,
                "beta_max": None,
                "rpca_tol": float(rpca_tol),
                "rpca_max_iter": int(rpca_max_iter),
            }
        )
        if method == METHOD_CA_RMS_FEDRPCA:
            method_parameters["cancellation_aware_rms"] = True
        elif method == METHOD_LAYER_JOINT_FEDRPCA:
            method_parameters.update(
                {
                    "joint_grouping": "same_transformer_layer_qv_by_factor",
                    "joint_group_count": len(layer_joint_groups),
                }
            )
    if set(output) != set(keys):
        # Registry validation should make this unreachable.  Keep the server
        # boundary fail-closed if a future method branch forgets a parameter.
        missing = sorted(set(keys) - set(output))
        extra = sorted(set(output) - set(keys))
        raise RuntimeError(
            f"aggregation produced an invalid state partition; missing={missing}, "
            f"extra={extra}"
        )
    state = _copy_finite_output(output, keys)

    diagnostics: dict[str, object] = {
        "schema_version": 1,
        "method": method,
        "n_clients": len(locals_arrays),
        "weighting": weighting,
        "adapter_aggregation": adapter_aggregation,
        "classifier_aggregation": classifier_aggregation,
        "adapter_pairs": [{"a_key": a_key, "b_key": b_key} for a_key, b_key in pairs],
        "classifier_keys": list(classifier),
        "method_parameters": method_parameters,
        "fedrpca_blocks": fedrpca_blocks,
        "state_sha256": paper_state_sha256(state),
    }
    if robust_aggregation is not None:
        diagnostics["robust_aggregation"] = robust_aggregation
    if method == METHOD_LAYER_JOINT_FEDRPCA:
        diagnostics["layer_joint_groups"] = [
            _layer_joint_group_metadata(group) for group in layer_joint_groups
        ]
    return PaperServerResult(state=state, diagnostics=diagnostics)


def _build_layer_joint_groups(
    pairs: Sequence[AdapterPair], state: Mapping[str, FloatArray]
) -> tuple[_LayerJointGroup, ...]:
    """Validate and return the exact 12-layer GPT-2 Q/V grouping registry."""

    by_layer_projection: dict[tuple[int, str], AdapterPair] = {}
    for pair_index, (a_key, b_key) in enumerate(pairs):
        match = _GPT2_QV_A_KEY.fullmatch(a_key)
        if match is None:
            raise ValueError(
                "layer-joint FedRPCA requires adapter A keys named "
                "transformer.h.<layer>.attn.c_attn.<q|v>.lora_A.weight; "
                f"received adapter_pairs[{pair_index}]={a_key!r}"
            )
        expected_b = a_key[: -len("lora_A.weight")] + "lora_B.weight"
        if b_key != expected_b:
            raise ValueError(
                "layer-joint FedRPCA requires each B key to match its real GPT-2 "
                f"Q/V A key; expected {expected_b!r}, received {b_key!r}"
            )
        identity = (int(match.group("layer")), match.group("projection"))
        if identity in by_layer_projection:
            raise ValueError(
                "layer-joint FedRPCA adapter registry contains duplicate "
                f"layer/projection identity {identity}"
            )
        by_layer_projection[identity] = (a_key, b_key)

    expected_identities = {
        (layer, projection)
        for layer in range(GPT2_LAYER_COUNT)
        for projection in ("q", "v")
    }
    actual_identities = set(by_layer_projection)
    if actual_identities != expected_identities:
        missing = sorted(expected_identities - actual_identities)
        extra = sorted(actual_identities - expected_identities)
        raise ValueError(
            "layer-joint FedRPCA requires complete q/v pairs for GPT-2 layers "
            f"0..{GPT2_LAYER_COUNT - 1}; missing={missing}, extra={extra}"
        )

    groups: list[_LayerJointGroup] = []
    for layer in range(GPT2_LAYER_COUNT):
        q_pair = by_layer_projection[(layer, "q")]
        v_pair = by_layer_projection[(layer, "v")]
        for factor_index, factor in enumerate(("A", "B")):
            q_key = q_pair[factor_index]
            v_key = v_pair[factor_index]
            q_shape = state[q_key].shape
            v_shape = state[v_key].shape
            if q_shape != v_shape:
                raise ValueError(
                    "layer-joint FedRPCA requires equal Q/V shapes within each "
                    f"layer and factor; layer={layer}, factor={factor}, "
                    f"q_shape={q_shape}, v_shape={v_shape}"
                )
            if len(q_shape) != 2 or len(v_shape) != 2:
                raise ValueError("layer-joint FedRPCA factors must be matrices")
            groups.append(
                _LayerJointGroup(
                    key=(
                        f"transformer.h.{layer}.attn.c_attn.qv." f"lora_{factor}.weight"
                    ),
                    layer=layer,
                    factor=factor,
                    q_key=q_key,
                    v_key=v_key,
                    q_shape=(int(q_shape[0]), int(q_shape[1])),
                    v_shape=(int(v_shape[0]), int(v_shape[1])),
                )
            )
    if len(groups) != GPT2_LAYER_COUNT * 2:
        raise RuntimeError("layer-joint FedRPCA did not construct exactly 24 groups")
    return tuple(groups)


def _concatenate_layer_joint_deltas(
    deltas: Sequence[Mapping[str, FloatArray]],
    groups: Sequence[_LayerJointGroup],
) -> list[dict[str, FloatArray]]:
    """Concatenate Q then V for every same-layer factor and every client."""

    result: list[dict[str, FloatArray]] = []
    for client_index, client in enumerate(deltas):
        grouped: dict[str, FloatArray] = {}
        for group in groups:
            q = client[group.q_key]
            v = client[group.v_key]
            if q.shape != group.q_shape or v.shape != group.v_shape:
                raise RuntimeError(
                    "layer-joint FedRPCA delta shape changed after registry "
                    f"validation for client={client_index}, group={group.key!r}"
                )
            concatenated = np.concatenate((q.ravel(order="C"), v.ravel(order="C")))
            if (
                concatenated.shape != (group.size,)
                or not np.isfinite(concatenated).all()
            ):
                raise RuntimeError(
                    f"layer-joint FedRPCA constructed an invalid group {group.key!r}"
                )
            grouped[group.key] = np.asarray(concatenated, dtype=np.float64)
        result.append(grouped)
    return result


def _split_layer_joint_updates(
    grouped_updates: Mapping[str, ArrayLike],
    groups: Sequence[_LayerJointGroup],
    *,
    expected_keys: Sequence[str],
) -> dict[str, FloatArray]:
    """Invert the exact Q-then-V concatenation without dropping a state key."""

    split: dict[str, FloatArray] = {}
    for group in groups:
        vector = _as_finite_array(
            grouped_updates[group.key], name=f"layer_joint_update[{group.key!r}]"
        )
        if vector.shape != (group.size,):
            raise RuntimeError(
                f"layer-joint FedRPCA group {group.key!r} has shape "
                f"{vector.shape}, expected {(group.size,)}"
            )
        if group.q_key in split or group.v_key in split:
            raise RuntimeError("layer-joint FedRPCA split mapping reused a state key")
        split[group.q_key] = vector[: group.q_size].reshape(group.q_shape).copy()
        split[group.v_key] = vector[group.q_size :].reshape(group.v_shape).copy()
    if set(split) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(split))
        extra = sorted(set(split) - set(expected_keys))
        raise RuntimeError(
            "layer-joint FedRPCA split mapping is incomplete; "
            f"missing={missing}, extra={extra}"
        )
    return split


def _layer_joint_fallback_reasons(
    diagnostic: BlockDiagnostics,
    *,
    rpca_max_iter: int,
    rpca_secondary_max_iter: int | None,
    input_is_exact_zero: bool,
) -> list[str]:
    """Return strict numerical-lineage failures for one joint PCP solve."""

    reasons: list[str] = []
    if isinstance(diagnostic, CancellationAwareBlockDiagnostics):
        reasons.append("unexpected_ca_rms_diagnostics")
    if diagnostic.residual < 0.0:
        reasons.append("invalid_residual")
    if diagnostic.beta_base <= 0.0 or diagnostic.beta <= 0.0:
        reasons.append("invalid_beta")
    if diagnostic.primary_max_iter != rpca_max_iter:
        reasons.append("primary_max_iter_mismatch")
    if (
        diagnostic.primary_residual is None
        or not math.isfinite(diagnostic.primary_residual)
        or diagnostic.primary_residual < 0.0
    ):
        reasons.append("invalid_primary_residual")
    if not isinstance(diagnostic.primary_converged, (bool, np.bool_)):
        reasons.append("invalid_primary_converged")
        primary_converged = False
    else:
        primary_converged = bool(diagnostic.primary_converged)
    if not isinstance(diagnostic.fallback_triggered, (bool, np.bool_)):
        reasons.append("invalid_fallback_triggered")
        fallback_triggered = False
    else:
        fallback_triggered = bool(diagnostic.fallback_triggered)
    expected_final_max_iter = (
        int(rpca_secondary_max_iter)
        if fallback_triggered and rpca_secondary_max_iter is not None
        else rpca_max_iter
    )
    if diagnostic.final_max_iter != expected_final_max_iter:
        reasons.append("final_max_iter_mismatch")
    if isinstance(diagnostic.iterations, bool) or not isinstance(
        diagnostic.iterations, (int, np.integer)
    ):
        reasons.append("invalid_iteration_type")
    elif diagnostic.iterations < 0 or diagnostic.iterations > expected_final_max_iter:
        reasons.append("invalid_iteration_lineage")
    elif diagnostic.iterations == 0:
        canonical_zero_lineage = (
            input_is_exact_zero
            and diagnostic.residual == 0.0
            and diagnostic.rank == 0
            and diagnostic.beta_base == 1.0
            and diagnostic.beta == 1.0
            and diagnostic.primary_residual == 0.0
            and primary_converged
            and not fallback_triggered
        )
        if not canonical_zero_lineage:
            reasons.append("invalid_zero_iteration_lineage")
    if rpca_secondary_max_iter is None and fallback_triggered:
        reasons.append("fallback_without_secondary_cap")
    if fallback_triggered:
        if primary_converged:
            reasons.append("fallback_after_primary_success")
    elif not primary_converged:
        reasons.append("primary_success_lineage_missing")
    return reasons


def _layer_joint_group_metadata(group: _LayerJointGroup) -> dict[str, object]:
    return {
        "group_key": group.key,
        "layer": group.layer,
        "factor": group.factor,
        "size": group.size,
        "concatenation_order": [
            {
                "projection": "q",
                "key": group.q_key,
                "shape": list(group.q_shape),
                "start": 0,
                "stop": group.q_size,
            },
            {
                "projection": "v",
                "key": group.v_key,
                "shape": list(group.v_shape),
                "start": group.q_size,
                "stop": group.size,
            },
        ],
    }


def _normalise_states(
    global_state: TrainableState, local_states: Sequence[TrainableState]
) -> tuple[dict[str, FloatArray], list[dict[str, FloatArray]], tuple[str, ...]]:
    if not isinstance(global_state, Mapping) or not global_state:
        raise ValueError("global_state must be a non-empty mapping")
    if isinstance(local_states, (str, bytes)) or not isinstance(local_states, Sequence):
        raise TypeError("local_states must be a non-empty sequence of mappings")
    if not local_states:
        raise ValueError("local_states must contain at least one client state")
    if any(not isinstance(key, str) or not key for key in global_state):
        raise ValueError("global_state keys must be non-empty strings")
    keys = tuple(sorted(global_state))
    expected = set(keys)
    global_arrays = {
        key: _as_finite_array(global_state[key], name=f"global_state[{key!r}]")
        for key in keys
    }
    local_arrays: list[dict[str, FloatArray]] = []
    for client_index, local in enumerate(local_states):
        if not isinstance(local, Mapping):
            raise TypeError(f"local_states[{client_index}] must be a mapping")
        if set(local) != expected:
            missing = sorted(expected - set(local))
            extra = sorted(set(local) - expected)
            raise ValueError(
                f"local_states[{client_index}] keys differ from global_state; "
                f"missing={missing}, extra={extra}"
            )
        converted: dict[str, FloatArray] = {}
        for key in keys:
            array = _as_finite_array(
                local[key], name=f"local_states[{client_index}][{key!r}]"
            )
            if array.shape != global_arrays[key].shape:
                raise ValueError(
                    f"local_states[{client_index}][{key!r}] shape {array.shape} "
                    f"does not match global shape {global_arrays[key].shape}"
                )
            converted[key] = array
        local_arrays.append(converted)
    return global_arrays, local_arrays, keys


def _validate_registry(
    state: Mapping[str, FloatArray],
    *,
    adapter_pairs: Sequence[AdapterPair],
    classifier_keys: Sequence[str],
) -> tuple[tuple[AdapterPair, ...], tuple[str, ...]]:
    if isinstance(adapter_pairs, (str, bytes)) or not isinstance(
        adapter_pairs, Sequence
    ):
        raise TypeError("adapter_pairs must be a non-empty sequence of (A_key, B_key)")
    if not adapter_pairs:
        raise ValueError("adapter_pairs must not be empty")
    pairs: list[AdapterPair] = []
    adapter_keys: list[str] = []
    for pair_index, pair in enumerate(adapter_pairs):
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise TypeError(
                f"adapter_pairs[{pair_index}] must be a two-element (A_key, B_key)"
            )
        a_key, b_key = pair
        if not isinstance(a_key, str) or not a_key:
            raise ValueError(f"adapter_pairs[{pair_index}] A_key must be non-empty")
        if not isinstance(b_key, str) or not b_key:
            raise ValueError(f"adapter_pairs[{pair_index}] B_key must be non-empty")
        if a_key == b_key:
            raise ValueError(f"adapter_pairs[{pair_index}] reuses one key as A and B")
        pairs.append((a_key, b_key))
        adapter_keys.extend((a_key, b_key))
    if len(set(adapter_keys)) != len(adapter_keys):
        raise ValueError("adapter_pairs contains duplicate or reused parameter keys")

    if isinstance(classifier_keys, (str, bytes)) or not isinstance(
        classifier_keys, Sequence
    ):
        raise TypeError("classifier_keys must be a non-empty sequence of keys")
    if not classifier_keys:
        raise ValueError("classifier_keys must not be empty")
    classifier = tuple(classifier_keys)
    if any(not isinstance(key, str) or not key for key in classifier):
        raise ValueError("classifier_keys must contain only non-empty strings")
    if len(set(classifier)) != len(classifier):
        raise ValueError("classifier_keys contains duplicate keys")
    overlap = sorted(set(adapter_keys) & set(classifier))
    if overlap:
        raise ValueError(f"adapter and classifier registries overlap: {overlap}")

    registered = set(adapter_keys) | set(classifier)
    unknown = sorted(registered - set(state))
    if unknown:
        raise KeyError(f"registered keys are absent from state: {unknown}")
    unregistered = sorted(set(state) - registered)
    if unregistered:
        raise ValueError(f"trainable state contains unregistered keys: {unregistered}")

    for pair_index, (a_key, b_key) in enumerate(pairs):
        a = state[a_key]
        b = state[b_key]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"adapter_pairs[{pair_index}] A and B values must be matrices"
            )
        if min(a.shape) == 0 or min(b.shape) == 0:
            raise ValueError(f"adapter_pairs[{pair_index}] factors must be non-empty")
        if b.shape[1] != a.shape[0]:
            raise ValueError(
                f"adapter_pairs[{pair_index}] has incompatible factor shapes: "
                f"A={a.shape}, B={b.shape}"
            )
    return tuple(pairs), classifier


def _as_finite_array(value: ArrayLike, *, name: str) -> FloatArray:
    # Accept CPU or CUDA torch tensors without importing torch in the package
    # core.  The returned server state is deliberately framework-independent.
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    array = np.asarray(candidate)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(array, dtype=np.float64)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _copy_finite_output(
    state: Mapping[str, ArrayLike], keys: Sequence[str]
) -> dict[str, FloatArray]:
    result: dict[str, FloatArray] = {}
    for key in keys:
        array = np.asarray(state[key], dtype=np.float64)
        if not np.isfinite(array).all():
            raise FloatingPointError(
                f"aggregated parameter {key!r} contains non-finite values"
            )
        result[key] = array.copy()
    return result


def paper_state_sha256(state: TrainableState) -> str:
    """Hash a paper-server state using its stable byte-framing contract."""

    digest = hashlib.sha256()
    for key in sorted(state):
        encoded_key = key.encode("utf-8")
        array = np.ascontiguousarray(state[key], dtype=np.float64)
        digest.update(len(encoded_key).to_bytes(8, "big"))
        digest.update(encoded_key)
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


# Compatibility for existing audited runners that imported the original
# private helper before the hash contract became public.
_state_sha256 = paper_state_sha256


def _validate_positive_finite(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


__all__ = [
    "ALL_SERVER_METHODS",
    "ALL_SUPPORTED_SERVER_METHODS",
    "AdapterPair",
    "COORDINATE_TRIM_FRACTION",
    "METHOD_CA_RMS_FEDRPCA",
    "METHOD_COORDINATE_MEDIAN",
    "METHOD_COORDINATE_TRIMMED_MEAN",
    "METHOD_FEDAVG",
    "METHOD_FEDRPCA",
    "METHOD_LAYER_JOINT_FEDRPCA",
    "METHOD_MEDIAN_NORM_CLIPPED_FEDAVG",
    "METHOD_SAMPLE_WEIGHTED_FEDAVG",
    "METHOD_TASK_ARITHMETIC",
    "METHOD_TIES",
    "PaperServerResult",
    "TrainableState",
    "aggregate_paper_state",
    "paper_state_sha256",
]
