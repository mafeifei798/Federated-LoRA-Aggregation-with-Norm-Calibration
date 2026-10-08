"""Historical VP solver checks, separate from the saved PCP/TN benchmark."""
import numpy as np

from experiments import legacy_fedrpca_vp as legacy


def test_truncated_rank_and_shapes():
    matrix = np.arange(24., dtype=np.float64).reshape(6, 4)
    u, s, vt = legacy.truncated_svd_rank_r(matrix, 2)
    assert u.shape == (6, 2) and s.shape == (2,) and vt.shape == (2, 4)
    assert np.linalg.matrix_rank((u*s) @ vt) <= 2


def test_small_solver_is_finite_and_does_not_modify_input():
    matrix = np.random.default_rng(5).normal(size=(6, 4))
    before = matrix.copy()
    low, sparse = legacy.robust_rpca_vp(matrix, rank=2, max_iter=3)
    assert low.shape == sparse.shape == matrix.shape
    assert np.isfinite(low).all() and np.isfinite(sparse).all()
    np.testing.assert_array_equal(matrix, before)
