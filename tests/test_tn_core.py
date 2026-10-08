"""Tiny synthetic unit tests only; no models, training or benchmark evaluation."""
import numpy as np
import pytest

from experiments import fedrpca_parameter_only_tn48 as tn


def test_direction_and_radius_with_exact_sparse_guard():
    rng = np.random.default_rng(7)
    observed = rng.normal(size=(3, 4, 5))
    sparse = .2 * observed
    sparse[0] = 0.
    original = observed.mean(axis=2) + sparse.mean(axis=2)
    out, details = tn.norm_map(observed, sparse, original, device='cpu', toy=True)
    np.testing.assert_array_equal(out[0], original[0])
    assert details[0]['canonical_guard']
    for i in (1, 2):
        assert np.linalg.norm(out[i]) == pytest.approx(2. * np.linalg.norm(observed[i].mean(axis=1)))
        np.testing.assert_allclose(out[i] / np.linalg.norm(out[i]),
                                   original[i] / np.linalg.norm(original[i]))


def test_production_does_not_silently_fall_back_to_cpu():
    x = np.ones((1, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        tn.norm_map(x, .2*x, x.mean(axis=2), device='cpu', toy=False)


def test_uniform_classifier_is_unchanged():
    parent = {'A': np.zeros((2, 2)), 'B': np.zeros((2, 2)), 'head': np.zeros(2)}
    local = [{k: np.full_like(v, .1*i) for k, v in parent.items()} for i in (1, 2, 3)]
    fr = {k: np.full_like(v, .3) for k, v in parent.items()}
    fr['head'] = np.mean(np.stack([x['head'] for x in local]), axis=0)
    observed = np.stack([np.column_stack([x[k].ravel() for x in local]) for k in ('A', 'B')])
    out, report = tn.assemble(parent, local, fr, .3*observed,
        adapter_pairs=(('A', 'B'),), classifier_keys=('head',), device='cpu', toy=True)
    np.testing.assert_array_equal(out['head'], fr['head'])
    assert report['feedback_rows'] == report['extra_pcp_solves'] == 0
    assert report['extra_server_model_forward'] == report['extra_server_model_backward'] == 0
