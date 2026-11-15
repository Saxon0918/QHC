"""Numerical witness for the explicitly normalized deferred C3 formula."""

from itertools import permutations

import numpy as np
from scipy.linalg import expm


_PERMUTATIONS = tuple(permutations(range(3)))


def _parity(permutation):
    inversions = sum(permutation[i] > permutation[j]
                     for i in range(3) for j in range(i + 1, 3))
    return -1 if inversions % 2 else 1


def _c3(hamiltonian, time):
    unitary = expm(-1j * time * hamiltonian)
    amplitudes = [np.prod([unitary[p[i], i] for i in range(3)])
                  for p in _PERMUTATIONS]
    boson = abs(sum(amplitudes)) ** 2
    fermion = abs(sum(_parity(p) * a
                      for p, a in zip(_PERMUTATIONS, amplitudes))) ** 2
    distinguishable = sum(abs(a) ** 2 for a in amplitudes)
    return float((boson + fermion) / 2 - distinguishable)


def test_open_wedge_has_positive_c3_without_triangle():
    h = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert h[0, 2] == 0
    ratio = _c3(h, 0.02) / 0.02**4
    assert abs(ratio - 4.0) < 0.02


def test_triangle_coefficient_has_three_wedges():
    h = np.ones((3, 3), dtype=float) - np.eye(3)
    ratio = _c3(h, 0.02) / 0.02**4
    assert abs(ratio - 12.0) < 0.05
