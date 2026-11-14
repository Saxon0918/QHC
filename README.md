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
ground-truth hyperedges. The current P1 code tests A's mechanism on independent
synthetic graphs and on separately timed Cornell event datasets. It does
**not** implement B, soft incidence, end-to-end candidate renewal, or claim
that Gate 1/2 has passed.

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
include size 2. The real-data protocol has its own size-2-inclusive
configuration and `p1-real` command. No official published-baseline
reproduction is claimed. No raw dataset is downloaded automatically.

## Temporal real-data P1 pilot

Download `email-Enron.tar.gz` from the [Cornell Enron release](https://www.cs.cornell.edu/~arb/data/email-Enron/)
and `contact-high-school.tar.gz` from the [Cornell high-school temporal release](https://www.cs.cornell.edu/~arb/data/contact-high-school/).
The official pages point to Google Drive archives. Archive contents and raw
timestamp values must be checked on the machine that can access them. Unpack
the exact archives without adding them to Git:

```bash
python scripts/06_unpack_cornell_events.py --archive /path/to/email-Enron.tar.gz --output data/raw/email-Enron --source-id cornell/email-Enron/temporal-at-most-25 --source-url 'https://www.cs.cornell.edu/~arb/data/email-Enron/' --archive-url 'https://drive.google.com/open?id=1tTVZkdpgRW47WWmsrdUCukHz0x2M6N77'
python scripts/06_unpack_cornell_events.py --archive /path/to/contact-high-school.tar.gz --output data/raw/contact-high-school --source-id cornell/contact-high-school/temporal-20-second --source-url 'https://www.cs.cornell.edu/~arb/data/contact-high-school/' --archive-url 'https://drive.google.com/open?id=1VA2P62awVYgluOIh1W4NZQQgkQCBk-Eu'
```

Then choose a single-seed pilot or the full prespecified seed sweep:

```bash
python main.py p1-real --config configs/p1_real_pilot.yaml
python main.py p1-real-sweep --config configs/p1_real_pilot.yaml
```

The two commands are alternatives; the sweep includes seed 42 from the pilot.

The runner fixes eight chronological windows per source from unique timestamp
quantiles, assigning four to training, one to validation, and three to test.
It writes the actual boundaries and absolute input paths to
`config_resolved.yaml` before scoring. The experiment uses each test window's
binary projection to reconstruct member sets **in that window**; it does not
forecast future events. Identical event IDs never cross splits, and one time
stamp is never split. All variants use the same candidates, training labels,
validation search, and held-out scores. Only member sets, not event frequency,
are the binary target. The minimum meaningful AP change is fixed at 0.02 in
the configuration.
The sweep reports each source's AP difference for seeds 42–46 and its spread.
These seeds test algorithm stability on the same data; they are not independent
networks or valid uncertainty clusters.

The high-school reference groups are maximal cliques of pairwise contacts in
20-second intervals, so they are operational labels rather than independent
evidence of irreducible multi-person interactions. Both datasets have repeated
node identities across time. The report logs repeated member sets and source
node overlap across roles. Each source gets its own scorer and report under
`results/RUN_ID/sources/source-0` and `source-1`; the root report compares the
two source-level AP differences with equal weight. It does not treat windows
or candidates as independent replicates. Inspect the root
`real_input_audit.json` and `candidate_metrics.json`, then each source's
`candidate_coverage.json`, `candidate_metrics.json`, `fairness_checks.json`,
and `runtime_memory.json`. Two source clusters cannot establish Gate 1. The
full feature run is CPU capable but can be computationally expensive; it is
suitable for the server.

## Run records

Each P1 run saves the resolved configuration, dependency/platform record,
data manifest and hashes, candidate coverage
and Recall@Budget, a label-free feature cache, label-free diagnostics, held-out
scores/labels for evaluation, paired cluster-level bootstrap comparisons,
and wall-clock/RSS records for candidate generation, every feature family and
scorer training. RSS peaks are sampled every 20 ms and can miss shorter spikes.
Synthetic runs also save the S3 ambiguity witness; real runs save the source
archive and time-window audit when the archive manifest is available.

The primary planned comparisons are Q3 against C0, C2 and C3. Reported AP is
accompanied by positive base rate and per-cardinality results. A positive
smoke-run difference alone does not establish a useful mechanism. Gate 1 also
requires at least two real-data protocols, a preregistered meaningful effect,
matched false-clique analyses and acceptable cost; Gate 2 requires exact
hyperedge recovery improvement under the same candidates and selection rule.

`docs/P0_P1_PROTOCOL.md` records the current decisions and limits. The shared
P1 evaluator keeps test truth out of candidate and feature construction for
both synthetic and temporal real-data runs.

## Theory-led P1 revision

The independent [revision protocol](docs/P1_REVISION_PROTOCOL.md) specifies
three separable tests: candidate support under a fixed budget, old Q3 versus
group-subspace spectral features on one frozen candidate pool, and exact-set
selection on that same pool. New `Qvol`/`Qcorr` features are compared with
direct boundary-Gram moments and matched heat/wave Gram statistics. The old
P1 variant IDs and results remain unchanged. A separate temporal module tests
the conditional repeated-window Poisson identifiability equations on
synthetic data; the existing eight real windows are too few to use it for
empirical rate recovery.

```bash
python main.py p1-revision --config configs/p1_revision_smoke.yaml
python main.py p1-revision --config configs/p1_revision.yaml
```

The first command is a small synthetic engineering check. The second requires
the Cornell raw event files and archive manifests at the paths in
`configs/p1_real_pilot.yaml`; run it on the server where those files exist.
Its new-feature results on the previously inspected Enron and school windows
are **exploratory**, not a passed scientific gate. Candidate AP is compared
only within the same frozen pool. Across search budgets, the current runner
reports candidate Recall@Budget, oracle F1 ceilings, and independent search
run times. Full exact-set F1/Jaccard is evaluated only on the frozen
progressive pool at `candidate.analysis_budget`; it is not a cross-budget
comparison. Confirmatory claims require an untouched time period or
independent source.
