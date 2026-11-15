"""Locked B1 experiment: label-blind latent hypergraph construction.

The same graph, features, times, landmarks, support budget, candidate pool,
decoder, and node split are used for quantum transport and a short-time
matched classical Markov control.  This runner never reads true hyperedges.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse
import yaml

from .construction_candidates import propose_groups
from .construction_cluster import cluster_transport_hypergraph
from .construction_compression import compress_kernel
from .construction_evaluation import evaluate_fusion, evaluate_operator, fused_operator
from .construction_kernels import (matched_markov, quantum_transport,
                                   spread_matched_markov)
from .construction_sparse import (selected_transport_columns,
                                  sparse_transport_hypergraph)
from .construction_synthetic import make_synthetic_graph, stratified_node_split
from .experiment_logger import RunProfiler, environment_record, sha256_file, write_json
from .hamiltonian import build_hamiltonian, checked_adjacency


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_grid(values: Any, name: str, *, allow_zero: bool = False) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    result: list[float] = []
    for value in values:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
            raise ValueError(f"{name} must contain finite {'nonnegative' if allow_zero else 'positive'} numbers")
        result.append(float(value))
    return result


def load_construction_config(path: Path) -> dict[str, Any]:
    """Validate a compact, precommitted B1 protocol before any graph is drawn."""
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("construction config must be a mapping")
    project = cfg.get("project")
    if (not isinstance(project, dict) or project.get("mode") not in {"synthetic", "npz"}
            or not isinstance(project.get("protocol"), str)
            or not project["protocol"]):
        raise ValueError("project needs mode=synthetic|npz and protocol name")
    _positive_int(project.get("seed"), "project.seed", minimum=0)
    mode = project["mode"]
    if mode == "synthetic":
        synthetic = cfg.get("synthetic")
        if not isinstance(synthetic, dict):
            raise ValueError("synthetic settings are required")
        regimes = synthetic.get("regimes")
        if (not isinstance(regimes, list) or not regimes
                or len(regimes) != len(set(regimes))
                or any(regime not in {"group", "pairwise", "heterophily"}
                       for regime in regimes)):
            raise ValueError("synthetic.regimes must be distinct supported regimes")
        seeds = synthetic.get("graph_seeds")
        if (not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
                or any(isinstance(seed, bool) or not isinstance(seed, int)
                       or seed < 0 for seed in seeds)):
            raise ValueError("synthetic.graph_seeds must be distinct nonnegative integers")
        _positive_int(synthetic.get("n_nodes"), "synthetic.n_nodes", minimum=12)
    else:
        real = cfg.get("npz")
        if (not isinstance(real, dict) or not isinstance(real.get("files"), list)
                or not real["files"] or any(not isinstance(item, str) or not item
                                          for item in real["files"])):
            raise ValueError("npz.files must list one or more local NPZ paths")
    kernel = cfg.get("kernel")
    if not isinstance(kernel, dict):
        raise ValueError("kernel settings are required")
    time_policy = kernel.get("time_policy", "fixed")
    if time_policy == "fixed":
        times = _positive_grid(kernel.get("times"), "kernel.times")
        weights = _positive_grid(kernel.get("weights"), "kernel.weights")
        if len(times) != len(weights) or len(set(times)) != len(times):
            raise ValueError("kernel times must be distinct and align with weights")
    elif time_policy == "gcn_trace_match":
        candidates = _positive_grid(kernel.get("time_candidates"), "kernel.time_candidates")
        if len(set(candidates)) != len(candidates):
            raise ValueError("kernel.time_candidates must be distinct")
        if mode != "synthetic":
            raise ValueError("gcn_trace_match is a dense synthetic diagnostic only")
    else:
        raise ValueError("kernel.time_policy must be fixed or gcn_trace_match")
    if kernel.get("kind") != "norm_laplacian":
        raise ValueError("B1 currently fixes kernel.kind=norm_laplacian")
    lam = kernel.get("lam")
    if (isinstance(lam, bool) or not isinstance(lam, (int, float))
            or not np.isfinite(lam) or not 0 <= lam <= 1):
        raise ValueError("kernel.lam must lie in [0,1]")
    _positive_int(kernel.get("dense_limit"), "kernel.dense_limit", minimum=2)
    if kernel["dense_limit"] > 512:
        raise ValueError("dense_limit > 512 requires a separate memory audit")
    sparse_cfg = cfg.get("sparse")
    if not isinstance(sparse_cfg, dict):
        raise ValueError("sparse settings are required")
    _positive_int(sparse_cfg.get("landmarks"), "sparse.landmarks", minimum=2)
    _positive_int(sparse_cfg.get("top_k"), "sparse.top_k", minimum=2)
    _positive_int(sparse_cfg.get("batch_size"), "sparse.batch_size")
    hard = cfg.get("hard")
    if not isinstance(hard, dict) or not isinstance(hard.get("enabled"), bool):
        raise ValueError("hard.enabled must be boolean")
    if hard["enabled"]:
        _positive_int(hard.get("max_candidates"), "hard.max_candidates")
        _positive_int(hard.get("budget"), "hard.budget")
        _positive_int(hard.get("min_size"), "hard.min_size", minimum=2)
        _positive_int(hard.get("max_size"), "hard.max_size", minimum=2)
        if hard["max_size"] < hard["min_size"]:
            raise ValueError("hard max_size must be >= min_size")
        _positive_int(hard.get("neighborhood_cap"), "hard.neighborhood_cap")
        _positive_grid(hard.get("weight_grid"), "hard.weight_grid")
        if not isinstance(hard.get("use_features", False), bool):
            raise ValueError("hard.use_features must be boolean")
        if hard.get("use_features", False):
            _positive_int(hard.get("feature_k"), "hard.feature_k")
    cluster = cfg.get("cluster", {"enabled": False})
    if not isinstance(cluster, dict) or not isinstance(cluster.get("enabled"), bool):
        raise ValueError("cluster.enabled must be boolean")
    if cluster["enabled"]:
        _positive_int(cluster.get("n_groups"), "cluster.n_groups")
    evaluation = cfg.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation settings are required")
    _positive_grid(evaluation.get("C_grid"), "evaluation.C_grid")
    fusion = cfg.get("fusion", {"enabled": False})
    if not isinstance(fusion, dict) or not isinstance(fusion.get("enabled"), bool):
        raise ValueError("fusion.enabled must be boolean")
    if fusion["enabled"]:
        alphas = _positive_grid(fusion.get("alphas"), "fusion.alphas", allow_zero=True)
        if (len(set(alphas)) != len(alphas) or any(alpha > 1.0 for alpha in alphas)
                or 0.0 not in alphas or 1.0 not in alphas or len(alphas) < 3):
            raise ValueError("fusion.alphas must be distinct values in [0, 1] with both endpoints and an interior value")
    return cfg


def _read_csr_npz(archive: Any, name: str) -> sparse.csr_matrix:
    if name in archive:
        return sparse.csr_matrix(archive[name])
    required = [f"{name}_{suffix}" for suffix in ("data", "indices", "indptr", "shape")]
    if not all(key in archive for key in required):
        raise ValueError(f"NPZ needs {name} or its data/indices/indptr/shape CSR fields")
    data, indices, indptr, shape = (archive[key] for key in required)
    return sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))


def _load_npz_graph(path: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray,
                                         dict[str, np.ndarray], dict[str, object]]:
    """Load an explicit graph/split archive without pickle or implicit downloads."""
    with np.load(path, allow_pickle=False) as archive:
        adjacency = checked_adjacency(_read_csr_npz(archive, "adjacency"))
        features = _read_csr_npz(archive, "features").toarray() if "features" not in archive else np.asarray(archive["features"])
        if "labels" not in archive:
            raise ValueError("NPZ needs labels")
        labels = np.asarray(archive["labels"])
        keys = {"train": "train_idx", "validation": "validation_idx", "test": "test_idx"}
        if not all(key in archive for key in keys.values()):
            raise ValueError("NPZ needs train_idx, validation_idx, test_idx")
        split = {role: np.asarray(archive[key]) for role, key in keys.items()}
    n = adjacency.shape[0]
    if features.ndim != 2 or features.shape[0] != n or labels.shape != (n,):
        raise ValueError("NPZ graph, features, labels have inconsistent shapes")
    if np.any(adjacency.data != 1.0):
        raise ValueError("B1 binary-graph protocol requires all observed edges to equal one")
    _validate_split_indices(split, n)
    return adjacency, features, labels, split, {
        "source": str(path), "source_sha256": sha256_file(path), "n_nodes": n,
        "n_features": int(features.shape[1]), "n_edges": int(adjacency.nnz // 2),
    }


def _validate_split_indices(split: dict[str, np.ndarray], n: int) -> None:
    """Reject malformed NPZ indices before a construction or classifier runs."""
    all_indices: list[np.ndarray] = []
    for role in ("train", "validation", "test"):
        values = np.asarray(split[role])
        if (values.ndim != 1 or values.size == 0
                or not np.issubdtype(values.dtype, np.integer)
                or np.issubdtype(values.dtype, np.bool_)):
            raise ValueError(f"{role} indices must be a nonempty integer vector")
        if np.any(values < 0) or np.any(values >= n) or np.unique(values).size != values.size:
            raise ValueError(f"{role} indices are out of range or repeated")
        all_indices.append(values)
    joined = np.concatenate(all_indices)
    if np.unique(joined).size != joined.size:
        raise ValueError("train, validation, test indices must be disjoint")


def _gcn_operator(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    n = adjacency.shape[0]
    augmented = adjacency + sparse.eye(n, format="csr")
    degree = np.asarray(augmented.sum(axis=1)).ravel()
    scaling = sparse.diags(1.0 / np.sqrt(degree), format="csr")
    return (scaling @ augmented @ scaling).tocsr()


def _combined_sparse(parts: list[Any], weights: np.ndarray
                     ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, float]:
    """Merge multi-time soft incidence, aggregating singleton residuals."""
    n = parts[0].incidence.shape[0]
    groups = [part.incidence[:, :len(part.landmark_ids)] for part in parts]
    residual = sum((float(w) * part.residual_degrees for w, part in zip(weights, parts)),
                   np.zeros(n, dtype=np.float64))
    incidence = sparse.hstack(groups + [sparse.diags(residual, format="csr")],
                              format="csr")
    edge_weights = np.concatenate([
        np.full(len(part.landmark_ids), float(w), dtype=np.float64)
        for w, part in zip(weights, parts)
    ] + [np.ones(n, dtype=np.float64)])
    theta = sum((float(w) * part.normalized_theta for w, part in zip(weights, parts)),
                sparse.csr_matrix((n, n), dtype=np.float64)).tocsr()
    trace_bound = float(sum(float(w) * part.trace_bound
                            for w, part in zip(weights, parts)))
    degrees = np.asarray(incidence @ edge_weights).ravel()
    if not np.allclose(degrees, 1.0, atol=1e-7, rtol=0.0):
        raise ArithmeticError("combined soft incidence lost unit node degrees")
    edge_degree = np.asarray(incidence.sum(axis=0)).ravel()
    inverse = np.divide(edge_weights, edge_degree,
                        out=np.zeros_like(edge_weights), where=edge_degree > 0)
    exact = (incidence @ sparse.diags(inverse, format="csr") @ incidence.T).tocsr()
    if not np.allclose((exact - theta).data, 0.0, atol=1e-8, rtol=0.0):
        raise ArithmeticError("combined soft incidence does not induce its stated operator")
    return theta, incidence, edge_weights, trace_bound


def _combined_cluster(parts: list[Any], weights: np.ndarray
                      ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, float]:
    """Concatenate partition hyperedges from each time with convex weights."""
    n = parts[0].incidence.shape[0]
    incidence = sparse.hstack([part.incidence for part in parts], format="csr")
    edge_weights = np.concatenate([
        np.full(part.incidence.shape[1], float(w), dtype=np.float64)
        for w, part in zip(weights, parts)
    ])
    theta = sum((float(w) * part.normalized_theta for w, part in zip(weights, parts)),
                sparse.csr_matrix((n, n), dtype=np.float64)).tocsr()
    trace_error = float(sum(float(w) * part.trace_error
                            for w, part in zip(weights, parts)))
    degrees = np.asarray(incidence @ edge_weights).ravel()
    if not np.allclose(degrees, 1.0, atol=1e-8, rtol=0.0):
        raise ArithmeticError("clustered incidence lost unit node degrees")
    edge_degree = np.asarray(incidence.sum(axis=0)).ravel()
    exact = (incidence @ sparse.diags(edge_weights / edge_degree, format="csr")
             @ incidence.T).tocsr()
    if not np.allclose((exact - theta).data, 0.0, atol=1e-8, rtol=0.0):
        raise ArithmeticError("clustered incidence does not induce its stated operator")
    return theta, incidence, edge_weights, trace_error


def _graph_inputs(config_path: Path, cfg: dict[str, Any]):
    if cfg["project"]["mode"] == "synthetic":
        setup = cfg["synthetic"]
        for regime in setup["regimes"]:
            for seed in setup["graph_seeds"]:
                adjacency, features, labels, metadata = make_synthetic_graph(
                    regime, seed, n=setup["n_nodes"])
                split = stratified_node_split(labels, seed=seed + cfg["project"]["seed"])
                metadata = {key: value for key, value in metadata.items()
                            if key != "latent_groups_diagnostic_only"}
                yield f"{regime}-{seed}", adjacency, features, labels, split, metadata
    else:
        for index, item in enumerate(cfg["npz"]["files"]):
            path = (config_path.parent / item).resolve()
            adjacency, features, labels, split, metadata = _load_npz_graph(path)
            yield f"npz-{index}-{path.stem}", adjacency, features, labels, split, metadata


def _paired_summary(records: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    by_graph: dict[tuple[str, int | str], dict[str, dict[str, Any]]] = {}
    for row in records:
        key = (str(row["regime"]), row["graph_id"])
        by_graph.setdefault(key, {})[row["method"]] = row
    summary: dict[str, Any] = {}
    for regime in sorted({key[0] for key in by_graph}):
        graph_rows = [arms for (r, _), arms in by_graph.items() if r == regime]
        methods = sorted({method for arms in graph_rows for method in arms})
        unit = ("independent_synthetic_draw" if mode == "synthetic"
                else "archive_independence_unverified")
        regime_summary: dict[str, Any] = {"n_graph_records": len(graph_rows),
                                           "unit_type": unit,
                                           "methods": {}, "paired_quantum_minus_markov": {},
                                           "paired_quantum_minus_spread_markov": {}}
        for method in methods:
            values = [arms[method]["evaluation"]["test_macro_f1"]
                      for arms in graph_rows if method in arms]
            regime_summary["methods"][method] = {
                "n": len(values), "test_macro_f1_mean": float(np.mean(values)),
                "test_macro_f1_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        for suffix in ("direct", "sparse", "cluster", "hard"):
            for control, destination in (("markov", "paired_quantum_minus_markov"),
                                         ("spread_markov", "paired_quantum_minus_spread_markov")):
                left, right = f"quantum_{suffix}", f"{control}_{suffix}"
                deltas = [arms[left]["evaluation"]["test_macro_f1"]
                          - arms[right]["evaluation"]["test_macro_f1"]
                          for arms in graph_rows if left in arms and right in arms]
                if deltas:
                    regime_summary[destination][suffix] = {
                        "n": len(deltas), "mean": float(np.mean(deltas)),
                        "values": [float(delta) for delta in deltas],
                    }
        fused_quantum = "graph_quantum_sparse_fusion"
        for control, destination in (
            ("graph_gcn", "paired_fusion_quantum_minus_graph"),
            ("graph_identity_fusion", "paired_fusion_quantum_minus_identity"),
            ("graph_markov_sparse_fusion", "paired_fusion_quantum_minus_markov"),
        ):
            matched = [arms for arms in graph_rows
                       if fused_quantum in arms and control in arms]
            if matched:
                deltas = [float(arms[fused_quantum]["evaluation"]["test_macro_f1"]
                                - arms[control]["evaluation"]["test_macro_f1"])
                          for arms in matched]
                regime_summary[destination] = {
                    "n": len(deltas), "mean": float(np.mean(deltas)),
                    "values": deltas,
                    "alpha_selection": "independent_validation_selection_from_shared_grid",
                }
        summary[regime] = regime_summary
    return summary


def run_construction(config_path: Path, output: Path) -> dict[str, Any]:
    """Run the prespecified B1 comparison and write audit-ready outputs."""
    run_start = perf_counter()
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    cfg = load_construction_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    records: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    profile = RunProfiler()
    for graph_id, A, X, y, split, metadata in _graph_inputs(config_path, cfg):
        graph_start = perf_counter()
        n = A.shape[0]
        kind = str(metadata.get("regime", "npz"))
        graph_key = str(graph_id)
        stage = f"{graph_key}"
        graph_dir = output / "graphs" / graph_key
        graph_dir.mkdir(parents=True, exist_ok=False)
        with profile.stage(f"{stage}/hamiltonian"):
            H = build_hamiltonian(A, X if cfg["kernel"]["lam"] > 0 else None,
                                  kind="norm_laplacian", lam=cfg["kernel"]["lam"])
        rng = np.random.default_rng(np.random.SeedSequence(
            [cfg["project"]["seed"], int(metadata.get("seed", len(graphs)))]))
        landmark_count = min(cfg["sparse"]["landmarks"], n)
        landmark_ids = np.sort(rng.choice(n, size=landmark_count, replace=False))
        dense = n <= cfg["kernel"]["dense_limit"]
        if cfg.get("cluster", {}).get("enabled", False) and (not dense or n > 128):
            raise ValueError("cluster construction requires a dense graph with at most 128 nodes")
        time_policy = cfg["kernel"].get("time_policy", "fixed")
        time_record: dict[str, Any] = {"policy": time_policy}
        calibrated_quantum = None
        kernel_seconds: dict[str, float] = {}
        if time_policy == "fixed":
            times = cfg["kernel"]["times"]
            weights = np.asarray(cfg["kernel"]["weights"], dtype=np.float64)
            weights /= weights.sum()
        else:
            if not dense:
                raise ValueError("gcn_trace_match requires n <= kernel.dense_limit")
            calibration_start = perf_counter()
            with profile.stage(f"{stage}/trace_calibration"):
                candidates = cfg["kernel"]["time_candidates"]
                proposed = quantum_transport(H, candidates,
                                             max_nodes=cfg["kernel"]["dense_limit"])
                proposed_k = [np.asarray(P).T @ np.asarray(P)
                              for P in proposed["transport"]]
                proposed_traces = [float(np.trace(K) / n) for K in proposed_k]
                target_trace = float(_gcn_operator(A).diagonal().mean())
                selected_index = int(np.argmin(np.abs(
                    np.asarray(proposed_traces) - target_trace)))
                times = [float(candidates[selected_index])]
                weights = np.array([1.0], dtype=np.float64)
                calibrated_quantum = {
                    "transport": [proposed["transport"][selected_index]],
                    "K": proposed_k[selected_index],
                    "times": np.asarray(times), "weights": weights.copy(),
                }
                time_record.update({
                    "target_gcn_mean_diagonal": target_trace,
                    "time_candidates": candidates,
                    "quantum_mean_diagonal_by_time": proposed_traces,
                    "selected_time": times[0],
                    "achieved_quantum_mean_diagonal": proposed_traces[selected_index],
                    "absolute_trace_gap": abs(proposed_traces[selected_index] - target_trace),
                    "relative_trace_gap": (abs(proposed_traces[selected_index] - target_trace)
                                           / max(target_trace, 1e-12)),
                })
            kernel_seconds["quantum"] = perf_counter() - calibration_start
        full: dict[str, dict[str, object]] = {}
        if dense:
            if calibrated_quantum is None:
                precompute_start = perf_counter()
                with profile.stage(f"{stage}/dense_quantum"):
                    full["quantum"] = quantum_transport(
                        H, times, weights, max_nodes=cfg["kernel"]["dense_limit"])
                kernel_seconds["quantum"] = perf_counter() - precompute_start
            else:
                full["quantum"] = calibrated_quantum
            precompute_start = perf_counter()
            with profile.stage(f"{stage}/dense_markov"):
                full["markov"] = matched_markov(H, times, weights,
                                                 max_nodes=cfg["kernel"]["dense_limit"])
            kernel_seconds["markov"] = perf_counter() - precompute_start
            precompute_start = perf_counter()
            with profile.stage(f"{stage}/dense_spread_markov"):
                full["spread_markov"] = spread_matched_markov(
                    H, times, weights, max_nodes=cfg["kernel"]["dense_limit"])
            kernel_seconds["spread_markov"] = perf_counter() - precompute_start
        operators: dict[str, sparse.spmatrix | np.ndarray] = {
            "features_only": sparse.eye(n, format="csr"),
            "graph_gcn": _gcn_operator(A),
        }
        diagnostics: dict[str, dict[str, Any]] = {
            "features_only": {}, "graph_gcn": {},
        }
        constructions: dict[str, Any] = {}
        candidates = ()
        if dense and cfg["hard"]["enabled"]:
            with profile.stage(f"{stage}/hard_candidates_shared"):
                candidates = propose_groups(
                    A, min_size=cfg["hard"]["min_size"],
                    max_size=cfg["hard"]["max_size"],
                    max_candidates=cfg["hard"]["max_candidates"],
                    neighborhood_cap=cfg["hard"]["neighborhood_cap"],
                    features=X if cfg["hard"].get("use_features", False) else None,
                    feature_k=cfg["hard"].get("feature_k", 10))
            constructions["hard_candidate_count"] = len(candidates)
            constructions["hard_candidate_uses_features"] = bool(
                cfg["hard"].get("use_features", False))
        family_costs: dict[str, dict[str, float]] = {}
        families = [("quantum", "quantum"), ("markov", "markov")]
        if dense:
            families.append(("spread_markov", "spread_markov"))
        for family, label in families:
            family_start = perf_counter()
            if dense:
                target = np.asarray(full[family]["K"])
                operators[f"{label}_direct"] = target
                diagnostics[f"{label}_direct"] = {
                    "operator_type": "complete_transport_gram",
                    "mean_diagonal": float(np.trace(target) / n),
                    "matrix_nnz_at_1e-12": int(np.count_nonzero(np.abs(target) > 1e-12)),
                }
                columns = [np.asarray(P)[:, landmark_ids]
                           for P in full[family]["transport"]]
            else:
                with profile.stage(f"{stage}/{label}_selected_columns"):
                    columns = selected_transport_columns(
                        H, times, landmark_ids, family=("quantum" if family == "quantum"
                                                       else "matched_markov"),
                        batch_size=cfg["sparse"]["batch_size"])
                target = None
            with profile.stage(f"{stage}/{label}_soft_sparse"):
                parts = [sparse_transport_hypergraph(
                    matrix, top_k=min(cfg["sparse"]["top_k"], n),
                    landmark_ids=landmark_ids) for matrix in columns]
                theta, incidence, edge_weights, trace_bound = _combined_sparse(
                    parts, weights)
            operators[f"{label}_sparse"] = theta
            approximation = (float(np.linalg.norm(theta.toarray() - target, ord="fro"))
                             if target is not None else None)
            exact_error = theta.toarray() - target if target is not None else None
            diagnostics[f"{label}_sparse"] = {
                "operator_type": "certified_sparse_soft_hypergraph",
                "trace_upper_bound": trace_bound,
                "frobenius_error_to_full": approximation,
                "spectral_error_to_full": (float(np.linalg.norm(exact_error, ord=2))
                                           if exact_error is not None else None),
                "trace_error_exact": (float(np.trace(exact_error))
                                      if exact_error is not None else None),
                "mean_diagonal": float(theta.diagonal().mean()),
                "n_group_edges": len(times) * landmark_count,
                "n_singleton_edges": n,
                "incidence_nnz": int(incidence.nnz),
                "induced_graph_nnz": int(theta.nnz),
                "retained_mass_mean": float(np.mean([p.retained_mass.mean() for p in parts])),
            }
            sparse.save_npz(graph_dir / f"{label}_incidence.npz", incidence)
            np.save(graph_dir / f"{label}_edge_weights.npy", edge_weights)
            sparse.save_npz(graph_dir / f"{label}_theta.npz", theta)
            constructions[f"{label}_sparse_supports"] = [
                [list(group) for group in p.supports] for p in parts]
            if dense and cfg.get("cluster", {}).get("enabled", False):
                with profile.stage(f"{stage}/{label}_cluster"):
                    merged_parts = [cluster_transport_hypergraph(
                        np.asarray(P), n_clusters=min(cfg["cluster"]["n_groups"], n))
                        for P in full[family]["transport"]]
                    clustered, cluster_incidence, cluster_weights, cluster_trace = (
                        _combined_cluster(merged_parts, weights))
                operators[f"{label}_cluster"] = clustered
                cluster_error = target - clustered.toarray()
                diagnostics[f"{label}_cluster"] = {
                    "operator_type": "covariance_certified_endpoint_clustering",
                    "trace_error_exact": cluster_trace,
                    "spectral_error_to_full": float(np.linalg.norm(cluster_error, ord=2)),
                    "frobenius_error_to_full": float(np.linalg.norm(
                        cluster_error, ord="fro")),
                    "mean_diagonal": float(clustered.diagonal().mean()),
                    "n_group_edges": int(cluster_incidence.shape[1]),
                    "incidence_nnz": int(cluster_incidence.nnz),
                    "induced_graph_nnz": int(clustered.nnz),
                }
                sparse.save_npz(graph_dir / f"{label}_cluster_incidence.npz",
                                cluster_incidence)
                np.save(graph_dir / f"{label}_cluster_weights.npy", cluster_weights)
                sparse.save_npz(graph_dir / f"{label}_cluster_theta.npz", clustered)
                constructions[f"{label}_cluster_partitions"] = [
                    [list(group) for group in part.clusters] for part in merged_parts]
            if dense and cfg["hard"]["enabled"]:
                with profile.stage(f"{stage}/{label}_hard"):
                    hard = compress_kernel(
                        target, candidates, budget=cfg["hard"]["budget"],
                        weight_grid=cfg["hard"]["weight_grid"])
                operators[f"{label}_hard"] = hard.theta
                diagnostics[f"{label}_hard"] = {
                    "operator_type": "greedy_hard_hypergraph",
                    "frobenius_error_to_full": hard.error_fro,
                    "mean_diagonal": float(np.trace(hard.theta) / n),
                    "n_group_edges": len(hard.selected_groups),
                    "n_singleton_edges": n,
                    "incidence_nnz": int(hard.incidence.nnz),
                    "status": hard.status,
                }
                sparse.save_npz(graph_dir / f"{label}_hard_incidence.npz", hard.incidence)
                np.save(graph_dir / f"{label}_hard_weights.npy", hard.weights)
                constructions[f"{label}_hard_selected"] = [
                    {"group": list(group), "weight": float(weight)}
                    for group, weight in zip(hard.selected_groups, hard.weights[n:])]
            post_kernel_seconds = perf_counter() - family_start
            family_costs[label] = {
                "kernel_precompute_seconds": kernel_seconds.get(label, 0.0),
                "post_kernel_construction_diagnostics_io_seconds": post_kernel_seconds,
                "total_construction_excluding_shared_seconds": (
                    kernel_seconds.get(label, 0.0) + post_kernel_seconds),
            }
        write_json(graph_dir / "construction.json", constructions)
        for method, operator in operators.items():
            start = perf_counter()
            with profile.stage(f"{stage}/evaluate/{method}"):
                scores = evaluate_operator(operator, X, y, split,
                                           C_grid=cfg["evaluation"]["C_grid"])
            records.append({
                "regime": kind, "graph_id": graph_key, "method": method,
                "evaluation": scores, "operator": diagnostics[method],
                "evaluation_seconds": perf_counter() - start,
            })
        if cfg.get("fusion", {}).get("enabled", False):
            graph_operator = operators["graph_gcn"]
            fusion_sources = {
                "graph_identity_fusion": operators["features_only"],
                "graph_quantum_sparse_fusion": operators["quantum_sparse"],
                "graph_markov_sparse_fusion": operators["markov_sparse"],
            }
            for method, additional in fusion_sources.items():
                start = perf_counter()
                with profile.stage(f"{stage}/evaluate/{method}"):
                    scores = evaluate_fusion(
                        graph_operator, additional, X, y, split,
                        alphas=cfg["fusion"]["alphas"],
                        C_grid=cfg["evaluation"]["C_grid"])
                    selected_alpha = float(scores["selected_alpha"])
                    selected_operator = fused_operator(
                        graph_operator, additional, selected_alpha)
                    sparse.save_npz(graph_dir / f"{method}_theta.npz",
                                    sparse.csr_matrix(selected_operator))
                records.append({
                    "regime": kind, "graph_id": graph_key, "method": method,
                    "evaluation": scores,
                    "operator": {
                        "operator_type": "convex_graph_and_additional_propagation",
                        "graph_source": "original_binary_graph_with_self_loops",
                        "additional_source": ("identity" if method == "graph_identity_fusion"
                                              else method.removeprefix("graph_").removesuffix("_fusion")),
                        "selected_alpha": selected_alpha,
                        "mean_diagonal": float(selected_operator.diagonal().mean()),
                        "induced_graph_nnz": int(sparse.csr_matrix(selected_operator).nnz),
                    },
                    "evaluation_seconds": perf_counter() - start,
                })
        kernel_comparison = {}
        if dense:
            quantum_k = np.asarray(full["quantum"]["K"])
            markov_k = np.asarray(full["markov"]["K"])
            kernel_comparison = {
                "quantum_minus_markov_fro": float(np.linalg.norm(
                    quantum_k - markov_k, ord="fro")),
                "quantum_minus_markov_max_abs": float(np.max(np.abs(
                    quantum_k - markov_k))),
                "quantum_minus_spread_markov_fro": float(np.linalg.norm(
                    quantum_k - np.asarray(full["spread_markov"]["K"]), ord="fro")),
                "spread_markov_generator_times": [
                    (float(value) if np.isfinite(value) else "infinity")
                    for value in np.asarray(full["spread_markov"]["markov_times"])],
            }
        graphs.append({
            "graph_id": graph_key, "regime": kind, "metadata": metadata,
            "landmark_ids": landmark_ids.tolist(),
            "train_idx": split["train"].tolist(),
            "validation_idx": split["validation"].tolist(),
            "test_idx": split["test"].tolist(),
            "dense_reference_computed": dense,
            "time_selection": time_record,
            "kernel_comparison": kernel_comparison,
            "family_construction_costs": family_costs,
            "graph_processing_seconds_excluding_data_creation": perf_counter() - graph_start,
        })
    report = {
        "protocol": cfg["project"]["protocol"],
        "mode": cfg["project"]["mode"],
        "status": "exploratory_b1_no_confirmatory_claim",
        "config_sha256": sha256_file(config_path),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parent.parent)): sha256_file(path)
            for path in sorted(list(Path(__file__).resolve().parent.glob("construction_*.py"))
                               + [Path(__file__).resolve().parent / "hamiltonian.py",
                                  Path(__file__).resolve().parent / "experiment_logger.py",
                                  Path(__file__).resolve().parent.parent / "main.py"])
        },
        "environment": environment_record(),
        "graphs": graphs,
        "records": records,
        "summary": _paired_summary(records, mode=cfg["project"]["mode"]),
        "profile": profile.record(),
        "total_processing_seconds_excluding_final_report_write": perf_counter() - run_start,
    }
    write_json(output / "report.json", report)
    return report
