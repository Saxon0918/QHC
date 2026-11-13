# Quantum-inspired hypergraph reconstruction (QHC)

This repository implements the **first, falsifiable P0/P1 stage** of the
[research specification](docs/RESEARCH_PLAN.md). Its question is whether a
group-conditioned coherent quantum-walk response improves discrimination of
recorded hyperedges from false clique candidates **after** strong classical
structure and dynamics controls are supplied. It runs on a classical computer.
No quantum speedup, irreducible multi-body detection, or recovery advantage is
assumed.

The plan has two separate targets. A reconstructs recorded hyperedges from a
projected ordinary graph, using separately labelled training hypergraphs.
B constructs a latent incidence matrix for downstream node tasks without
ground-truth hyperedges. This first delivery tests A's mechanism with
independent synthetic graphs. It does **not** implement B, soft incidence,
end-to-end candidate renewal, or claim that Gate 1/2 has passed.

## Run the first acceptance checks

Use Python 3.9+ with NumPy, SciPy, scikit-learn, NetworkX, PyYAML, psutil and
pytest. On the development host, `/opt/anaconda3/envs/pytorch/bin/python`
already has these dependencies; the system `python3` does not have SciPy.

```bash
python -m pip install -e '.[test]'
python scripts/00_verify_theory.py
python -m pytest -q tests
python scripts/05_train_candidate_scorer.py --config configs/p1_smoke.yaml
```

`main.py` provides equivalent `verify` and `p1` subcommands for PyCharm.
The smoke configuration is an **engineering check**, with few held-out graphs;
its numerical output is not a Gate 1 result. The larger
`configs/synthetic_p0.yaml` is a reproducible starting grid, not a tuned
scientific conclusion. Every run writes a new `results/RUN_ID` directory by
default and refuses to overwrite a nonempty explicit output directory.

## What is implemented

| Component | Main API | Check |
| --- | --- | --- |
| Observation and truth | `qhyper.graph_io.GraphRecord`, `project_hypergraph` | Binary and multiplicity projection, normalized member sets, independent event metadata |
| Synthetic experiments | `generate_synthetic`, `triangle_ambiguity_example` | S1 separable/decoy groups, S2 overlap, S3 same projection with different truths |
| Fixed candidates | `propose_candidates`, `candidate_coverage` | Budgeted clique/subclique proposals, per-size recall ceiling |
| Quantum response | `build_hamiltonian`, `extract_group_quantum_features` | Hermitian topology Hamiltonian, sparse batched action, `R_coh`, `R_inc`, `C`, random-phase reference |
| Classical controls | `classical_candidate_matrix`, `dynamic_family_matrix` | Size/degree/boundary/external-neighbor and exact short-time term; heat, wave, spectral walk and quantum-pair features |
| Scoring and metrics | `fit_candidate_scorer`, `evaluate_candidate_predictions` | Same train-only scaler and logistic model family, validation-only selection, exact-set and candidate metrics |

The primary Hamiltonian is `H = (I-D^{-1/2} A D^{-1/2})/2`, with zero
Laplacian row/diagonal for isolated nodes. The group feature uses a uniform
same-phase candidate state and an internal projector. `C = R_coh - R_inc`
can be negative. Its short-time coefficient is a classical external-coupling
inner product; the exact coefficient is already included in C0. A random-phase
mean is the Q4 dephased reference; its sampling variance is diagnostic only.

For the noiseless binary projection, true hyperedges must be cliques, but
cliques need not be true hyperedges. Candidates are generated from `A` alone;
the same ordered pool is used by every feature variant. `GraphRecord.observed()`
removes truth before feature extraction. Test truth is joined after caches are
written, for metrics only. Features, scalers, class weights, regularization
selection, and thresholds never fit on test labels. There is a cross-graph
train/validation/test split. C0–C4 and Q1–Q4 variants share the same scorer
family and validation search grid.

`configs/p1_smoke.yaml` starts at size 3, so all true size-2 event edges in
S1 are outside candidate scope. The output reports this in candidate coverage
and exact-recall ceilings. A later school-contact protocol must explicitly
include size 2. The Cornell `nverts`/`simplices` local-file loader is available,
but no real-data experiment or official published-baseline reproduction is
claimed here. No raw dataset is downloaded automatically.

## Run records

Each P1 run saves the resolved configuration, dependency/platform record,
synthetic data manifest and hashes, S3 ambiguity witness, candidate coverage
and Recall@Budget, a label-free feature cache, label-free diagnostics, held-out
scores/labels for evaluation, paired **graph-level** bootstrap comparisons,
and wall-clock/RSS records for candidate generation, every feature family and
scorer training. RSS peaks are sampled every 20 ms and can miss shorter spikes.

The primary planned comparisons are Q3 against C0, C2 and C3. Reported AP is
accompanied by positive base rate and per-cardinality results. A positive
smoke-run difference alone does not establish a useful mechanism. Gate 1 also
requires at least two real-data protocols, a preregistered meaningful effect,
matched false-clique analyses and acceptable cost; Gate 2 requires exact
hyperedge recovery improvement under the same candidates and selection rule.

`docs/P0_P1_PROTOCOL.md` records the current decisions and limits. The code
is deliberately modular so later A1 real-data protocols can be added without
using test truth during candidate or feature construction.
