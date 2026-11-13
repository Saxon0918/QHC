#!/usr/bin/env python3
"""Run the P0 mathematical acceptance checks without requiring pytest."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.linalg import expm

from qhyper.hamiltonian import build_hamiltonian
from qhyper.group_response import coherent_group_response, short_time_coefficient
from qhyper.synthetic import triangle_ambiguity_example


def verify() -> dict:
    a = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
    ], dtype=float)
    h = build_hamiltonian(a)
    h_dense = h.toarray() if hasattr(h, "toarray") else np.asarray(h)
    nodes = (0, 1, 2)
    t = 1.0
    u = expm(-1j * t * h_dense)
    result = coherent_group_response(h, nodes, [t], exact_incoherent=True)
    rc, ri, c = (float(result[key][0]) for key in ("R_coh", "R_inc", "C"))

    assert np.allclose(u.conj().T @ u, np.eye(len(a)), atol=1e-10)
    p = np.abs(u) ** 2
    assert np.allclose(p.sum(axis=0), 1, atol=1e-10)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-10)
    assert 0 <= rc <= 1 and 0 <= ri <= 1 and -1 <= c <= 1

    # Equations (1) and (2): interference and exhaustive sign average.
    cross = (2 / len(nodes)) * sum(
        np.real(np.sum(u[list(nodes), i] * np.conjugate(u[list(nodes), j])))
        for i, j in itertools.combinations(nodes, 2)
    )
    assert np.isclose(c, cross, atol=1e-10)
    phase_mean = 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=len(nodes)):
        s = np.zeros(len(a), dtype=complex)
        s[list(nodes)] = np.asarray(signs) / np.sqrt(len(nodes))
        phase_mean += np.sum(np.abs((u @ s)[list(nodes)]) ** 2) / 2 ** len(nodes)
    assert np.isclose(ri, phase_mean, atol=1e-10)

    # Equation (3): the short-time coefficient, not a claim of quantum gain.
    small_t = 1e-3
    small_c = float(coherent_group_response(
        h, nodes, [small_t], exact_incoherent=True)["C"][0])
    coefficient = float(short_time_coefficient(h, nodes))
    assert abs(small_c / small_t ** 2 - coefficient) < 1e-5

    assert abs(float(coherent_group_response(
        h, nodes, [0.0], exact_incoherent=True)["C"][0])) < 1e-10
    assert abs(float(coherent_group_response(
        h, tuple(range(len(a))), [t], exact_incoherent=True)["C"][0])) < 1e-10

    # The statistic is permutation invariant when H, candidate and A all move.
    perm = np.array([3, 1, 4, 0, 2])
    inverse = np.argsort(perm)
    moved_nodes = tuple(sorted(int(inverse[i]) for i in nodes))
    moved_h = build_hamiltonian(a[np.ix_(perm, perm)])
    moved = coherent_group_response(moved_h, moved_nodes, [t], exact_incoherent=True)
    assert np.allclose([rc, ri, c], [moved[k][0] for k in ("R_coh", "R_inc", "C")], atol=1e-10)

    # Equation (4) is a conservative fixed-candidate perturbation bound.
    delta = np.zeros_like(h_dense)
    delta[0, 3] = delta[3, 0] = 0.01
    changed = coherent_group_response(h_dense + delta, nodes, [t], exact_incoherent=True)
    assert abs(float(changed["C"][0]) - c) <= 4 * t * np.linalg.norm(delta, 2) + 1e-10

    first, second = triangle_ambiguity_example()
    assert np.array_equal(first.adjacency_csr.toarray(), second.adjacency_csr.toarray())
    assert first.hyperedges_truth != second.hyperedges_truth
    first_h = build_hamiltonian(first.adjacency_csr)
    second_h = build_hamiltonian(second.adjacency_csr)
    first_c = coherent_group_response(first_h, (0, 1), [t], exact_incoherent=True)["C"]
    second_c = coherent_group_response(second_h, (0, 1), [t], exact_incoherent=True)["C"]
    assert np.allclose(first_c, second_c)

    return {
        "passed": True,
        "checks": ["unitarity", "bistochasticity", "probability_bounds",
                   "interference_identity", "rademacher_identity",
                   "short_time_coefficient", "degenerate_cases",
                   "permutation", "perturbation_bound", "projection_ambiguity"],
        "example": {"R_coh": rc, "R_inc": ri, "C": c,
                    "short_time_coefficient": coefficient},
        "ambiguity": "不能仅从 A 区分两个真实超图",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
