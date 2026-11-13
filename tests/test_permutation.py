import numpy as np

from qhyper.group_response import coherent_group_response, short_time_coefficient
from qhyper.hamiltonian import build_hamiltonian


def test_hamiltonian_and_group_response_are_permutation_equivariant():
    rng = np.random.default_rng(837)
    n = 9
    a = rng.uniform(size=(n, n))
    a = (a + a.T) / 2
    a[a < 0.55] = 0.0
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    original_group = (0, 2, 3, 7)
    times = [0.0, 0.25, 0.5, 2.0]
    permutation = rng.permutation(n)
    inverse = np.argsort(permutation)
    permuted_a = a[np.ix_(permutation, permutation)]
    permuted_group = tuple(sorted(int(inverse[node]) for node in original_group))

    h = build_hamiltonian(a)
    permuted_h = build_hamiltonian(permuted_a)
    np.testing.assert_allclose(
        permuted_h.toarray(), h.toarray()[np.ix_(permutation, permutation)]
    )
    original = coherent_group_response(h, original_group, times)
    changed = coherent_group_response(permuted_h, permuted_group, times)
    for key in ("R_coh", "R_inc", "C", "features"):
        np.testing.assert_allclose(original[key], changed[key], atol=1e-12)
    np.testing.assert_allclose(
        short_time_coefficient(h, original_group),
        short_time_coefficient(permuted_h, permuted_group),
        atol=1e-12,
    )
