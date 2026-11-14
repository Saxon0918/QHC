"""Conditional identification from repeated *binary* graph projections.

This module is a separate research protocol, not an input to the eight-window
P1 pilot.  The identities require independent Poisson hyperedge event counts
with stable rates across comparable, equal-duration windows.  They identify
event *rates* from a population of projected graphs, not the latent events in
one particular test window.  The simulation exposes latent counts only to
synthetic tests; estimators accept observed pair projections alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, expm1, isfinite, log
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .graph_io import Edge, canonicalize_edge


Dyad = tuple[int, int]


def _dyad(nodes: Iterable[int]) -> Dyad:
    pair = canonicalize_edge(nodes)
    if len(pair) != 2:
        raise ValueError("a dyad must contain exactly two distinct nodes")
    return pair


def _dyads(pairs: Iterable[Iterable[int]]) -> tuple[Dyad, ...]:
    result = tuple(_dyad(pair) for pair in pairs)
    if not result or len(set(result)) != len(result):
        raise ValueError("provide at least one distinct dyad")
    return result


def _rates(rates: Mapping[Iterable[int], float], n_nodes: int | None = None
           ) -> dict[Edge, float]:
    if not isinstance(rates, Mapping):
        raise TypeError("rates must map hyperedges to nonnegative intensities")
    normalized: dict[Edge, float] = {}
    for raw_edge, raw_rate in rates.items():
        edge = canonicalize_edge(raw_edge, n_nodes=n_nodes)
        if edge in normalized:
            raise ValueError("duplicate hyperedge after canonicalization")
        rate = float(raw_rate)
        if not isfinite(rate) or rate < 0:
            raise ValueError("event intensities must be finite and nonnegative")
        normalized[edge] = rate
    return normalized


@dataclass(frozen=True)
class PoissonWindowSample:
    """Synthetic witness; latent counts must never be passed to test inference."""

    hyperedges: tuple[Edge, ...]
    latent_counts: np.ndarray  # (windows, hyperedges)
    pair_counts: np.ndarray  # (windows, nodes, nodes)
    binary_adjacency: np.ndarray  # (windows, nodes, nodes)
    duration: float


def simulate_poisson_windows(
    rates: Mapping[Iterable[int], float], *, n_nodes: int, duration: float,
    n_windows: int, seed: int = 0,
) -> PoissonWindowSample:
    """Draw independent latent group counts, then project each window to pairs.

    This is a deliberately falsifiable synthetic model.  ``pair_counts`` are
    provided for a *separate* count-observation protocol; the binary estimator
    below uses only ``binary_adjacency``.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes < 2:
        raise ValueError("n_nodes must be an integer at least two")
    if not isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    if isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 1:
        raise ValueError("n_windows must be a positive integer")
    normalized = _rates(rates, n_nodes=n_nodes)
    edges = tuple(sorted(normalized))
    intensities = np.array([normalized[edge] for edge in edges], dtype=float)
    rng = np.random.default_rng(seed)
    latent = (rng.poisson(duration * intensities, size=(n_windows, len(edges)))
              if edges else np.zeros((n_windows, 0), dtype=np.int64))
    pair_counts = np.zeros((n_windows, n_nodes, n_nodes), dtype=np.int64)
    for column, edge in enumerate(edges):
        for i, j in combinations(edge, 2):
            pair_counts[:, i, j] += latent[:, column]
            pair_counts[:, j, i] += latent[:, column]
    binary = (pair_counts > 0).astype(np.int8)
    return PoissonWindowSample(edges, latent, pair_counts, binary, float(duration))


def joint_absence_rate(rates: Mapping[Iterable[int], float],
                       dyads: Iterable[Iterable[int]]) -> float:
    """Return F(J) = -log P(all dyads in J are absent) / window duration.

    Independent Poisson group events imply that F(J) is the sum of rates for
    all groups containing *at least one* dyad of J.  The value is independent
    of the common window duration.
    """
    normalized = _rates(rates)
    pairs = _dyads(dyads)
    return float(sum(rate for edge, rate in normalized.items()
                     if any(i in edge and j in edge for i, j in pairs)))


def joint_absence_probability(rates: Mapping[Iterable[int], float],
                              dyads: Iterable[Iterable[int]],
                              duration: float) -> float:
    """Analytic probability of simultaneous dyad absence under the model."""
    if not isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    return exp(-duration * joint_absence_rate(rates, dyads))


def superset_intensity_from_absence(
    cover_dyads: Iterable[Iterable[int]],
    absence_rate: Callable[[tuple[Dyad, ...]], float],
) -> float:
    """Inclusion-exclusion recover S(V)=sum_{e containing V} lambda_e.

    ``V`` is the union of endpoints of the distinct dyads supplied.  The
    callback evaluates F(J) for every nonempty subset J of those dyads.
    Different dyad covers of the same V must agree under the assumed model.
    """
    pairs = _dyads(cover_dyads)
    total = 0.0
    for size in range(1, len(pairs) + 1):
        sign = 1.0 if size % 2 else -1.0
        for subset in combinations(pairs, size):
            value = float(absence_rate(subset))
            if not isfinite(value) or value < 0:
                raise ValueError("joint absence rates must be finite and nonnegative")
            total += sign * value
    return float(total)


def theoretical_superset_intensity(rates: Mapping[Iterable[int], float],
                                   vertices: Iterable[int]) -> float:
    """Direct latent-rate expression for checking the observable identity."""
    group = canonicalize_edge(vertices)
    normalized = _rates(rates)
    return float(sum(rate for edge, rate in normalized.items()
                     if set(group).issubset(edge)))


def mobius_invert_superset_intensities(
    superset_rates: Mapping[Iterable[int], float],
    candidate_universe: Iterable[Iterable[int]], *, tolerance: float = 1e-9,
) -> dict[Edge, float]:
    """Recover rates on a declared finite support by downward Möbius inversion.

    Preconditions are substantive: S(V) must be supplied for *every* group in
    ``candidate_universe``, and all groups with nonzero latent rates must be in
    that universe.  Missing larger groups otherwise alias into smaller rates.
    This function rejects negative recovered rates beyond numerical tolerance.
    """
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    candidates = [canonicalize_edge(edge) for edge in candidate_universe]
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate_universe contains duplicate groups")
    supplied: dict[Edge, float] = {}
    for raw_edge, raw_value in superset_rates.items():
        edge = canonicalize_edge(raw_edge)
        if edge in supplied:
            raise ValueError("duplicate superset rate after canonicalization")
        value = float(raw_value)
        if not isfinite(value) or value < 0:
            raise ValueError("superset intensities must be finite and nonnegative")
        supplied[edge] = value
    if set(supplied) != set(candidates):
        raise ValueError("supply one superset intensity per candidate group")
    ordered = sorted(candidates, key=lambda edge: (-len(edge), edge))
    recovered: dict[Edge, float] = {}
    for edge in ordered:
        members = set(edge)
        value = supplied[edge] - sum(
            rate for larger, rate in recovered.items()
            if members.issubset(larger)
        )
        if value < -tolerance:
            raise ValueError("negative inverted rate: assumptions or estimates are inconsistent")
        recovered[edge] = max(0.0, float(value))
    return recovered


def estimate_joint_absence_rate(
    binary_windows: np.ndarray, dyads: Iterable[Iterable[int]], *,
    duration: float, min_windows: int = 100, min_absences: int = 5,
) -> float:
    """Empirical F(J) from comparable binary windows, with adequacy guards.

    The defaults are *minimum screening safeguards*, not a power calculation.
    In particular, the existing eight P1 windows must not be treated as an
    adequate sample for this estimator.  A zero joint-absence count cannot be
    smoothed silently into a finite identification claim.
    """
    if not isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    if (isinstance(min_windows, bool) or not isinstance(min_windows, int) or
            min_windows < 1 or isinstance(min_absences, bool) or
            not isinstance(min_absences, int) or min_absences < 1):
        raise ValueError("minimum sample and absence counts must be positive integers")
    graphs = np.asarray(binary_windows)
    if graphs.ndim != 3 or graphs.shape[1] != graphs.shape[2]:
        raise ValueError("binary_windows must have shape (windows,nodes,nodes)")
    n_windows, n_nodes, _ = graphs.shape
    if n_windows < min_windows:
        raise ValueError("insufficient repeated windows for joint-absence estimation")
    if (not np.all(np.isin(graphs, (0, 1))) or
            not np.array_equal(graphs, np.swapaxes(graphs, 1, 2)) or
            np.any(np.diagonal(graphs, axis1=1, axis2=2))):
        raise ValueError("each observed projection must be binary, symmetric, and loop-free")
    pairs = _dyads(dyads)
    if any(j >= n_nodes for _, j in pairs):
        raise ValueError("dyad vertex outside observed graph")
    absent = np.ones(n_windows, dtype=bool)
    for i, j in pairs:
        absent &= graphs[:, i, j] == 0
    count = int(np.count_nonzero(absent))
    if count < min_absences:
        raise ValueError("too few joint-absence observations for a finite estimate")
    return float(-log(count / n_windows) / duration)


def gamma_poisson_predictive_presence(
    training_counts: Sequence[int], training_exposures: Sequence[float], *,
    prior_shape: float, prior_rate: float, future_exposure: float,
) -> float:
    """Training-only shrinkage prior P(N_future >= 1) for one group.

    If lambda ~ Gamma(shape=a, rate=b) and train counts have independent
    Poisson(exposure_t * lambda) likelihoods, the posterior is Gamma(a',b').
    Integrating lambda yields pi=1-(b'/(b'+future_exposure))**a'.  Test-window
    adjacency or hidden test hyperedges are deliberately absent from the API.
    """
    counts = np.asarray(tuple(training_counts))
    exposures = np.asarray(tuple(training_exposures), dtype=float)
    if (counts.ndim != 1 or exposures.shape != counts.shape or
            not np.all(np.isfinite(exposures)) or np.any(exposures < 0) or
            not np.all(np.equal(counts, np.floor(counts))) or
            np.any(counts < 0)):
        raise ValueError("training counts and nonnegative exposures must align")
    if any(not isfinite(value) or value <= 0
           for value in (prior_shape, prior_rate, future_exposure)):
        raise ValueError("Gamma hyperparameters and future exposure must be positive")
    posterior_shape = prior_shape + float(np.sum(counts))
    posterior_rate = prior_rate + float(np.sum(exposures))
    exponent = posterior_shape * log(posterior_rate / (posterior_rate + future_exposure))
    return float(-expm1(exponent))
