# Quantum-inspired hypergraph construction and reconstruction (QHC)

This repository implements the P0/P1 recovery experiments and a separate B1
latent-construction prototype under the
[research specification](docs/RESEARCH_PLAN.md). P0/P1 asks whether a
group-conditioned coherent quantum-walk response improves discrimination of
recorded hyperedges from false clique candidates **after** strong classical
structure and dynamics controls are supplied. It runs on a classical computer.
No quantum speedup, irreducible multi-body detection, or recovery advantage is
assumed.

The plan has two separate targets. A reconstructs recorded hyperedges from a
projected ordinary graph, using separately labelled training hypergraphs.
B constructs a latent incidence matrix for downstream node tasks without
ground-truth hyperedges. P1 tests A's mechanism on independent synthetic
graphs and separately timed Cornell event datasets. B1 constructs soft and
hard latent groups from the allowed graph and tests them on fixed downstream
node tasks. Neither stage claims that a scientific gate has passed.

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

## Theory-led latent construction (B1)

The [B1 mathematical protocol](docs/LATENT_CONSTRUCTION_THEORY.md) derives a
soft incidence matrix from shared quantum-walk measurement destinations. It
also derives a classical continuous-time Markov control that agrees through
the first nontrivial short-time term. Small graphs also receive a second
classical control matched to the quantum kernel's total mixing trace. A
mass-conserving sparse approximation
keeps a fixed set of destinations and their top-k node weights, with singleton
residuals; it remains a valid normalized hypergraph operator with a spectral
error upper bound. A separate local-neighborhood candidate pool permits
nonclique hard groups. Both quantum and classical kernels use exactly the same
pool, landmark IDs, budgets, times, features, splits and downstream decoder.
Hard candidates default to topology only; `hard.use_features: true` adds
mutual-kNN positive-cosine feature links to the local proposal graph, and
must be reported as a separate input-information ablation.

```bash
python main.py b1-construction --config configs/b1_construction_smoke.yaml
python main.py b1-construction --config configs/b1_construction_trace_pilot.yaml
python main.py b1-construction --config configs/b1_construction_cluster_pilot.yaml
```

The first CPU run has three independently drawn graphs in each of group-generated,
pairwise-only and heterophilous regimes. It reports test Macro-F1 and LogLoss,
operator error, storage counts, timings and sampled RSS in
`results/RUN_ID/report.json`. The three graphs per regime are an engineering
and negative-control check, not enough to claim generalization. Direct
quantum/Markov weighted graph kernels and ordinary graph propagation are
included. Linear HGNN propagation on the full soft incidence is exactly the
same as direct propagation with its induced weighted graph; the code tests
that identity. A classification change does not prove recorded event recovery
or irreducible multi-body interaction.
The second config uses five new graphs per regime and chooses the *nearest*
time from a
predeclared grid solely by matching the original graph's average diagonal
propagation strength. Its selection uses no node labels and records the cost
of computing every candidate time. It is still an exploratory synthetic
pilot, not a real-data confirmation. The third config uses five further new
graphs per regime and compares a four-group Ward merge of measurement
destinations with top-k sparsification at a similar incidence-entry budget.
The merge has an exact covariance error identity but requires the full
transport matrix, so it is a small-graph diagnostic.
The observed numbers and their limits are written in the
[B1 pilot results](docs/B1_PILOT_RESULTS.md).

For a real node-classification pilot, use
`configs/b1_construction_npz.yaml` on the server. It expects each `.npz` to
contain `features` (N×D), `labels` (N), and disjoint `train_idx`,
`validation_idx`, `test_idx` arrays. Supply `adjacency` as a dense N×N array
or four CSR arrays named `adjacency_data`, `adjacency_indices`,
`adjacency_indptr`, `adjacency_shape`. A CSR feature matrix can likewise use
`features_data/indices/indptr/shape`. The B1 loader never downloads a dataset
or invents a split. Preserve the source dataset's standard masks where
available, record its license and preprocessing, and keep test labels sealed
until the final metrics. The NPZ config computes only selected transport
columns above 128 nodes, avoiding a full dense quantum propagator; the
exact full-kernel and hard-compression arms are then explicitly omitted.

If a PyTorch-compatible `torch-geometric` installation is available on the
server, the optional exporter below downloads the public-split Planetoid
Cora graph and writes the exact NPZ schema plus a checksum manifest. It does
not create hyperedge labels. Keep the official source and package version in
the manifest; CiteSeer and PubMed are also supported by `--dataset`, with the
config's NPZ path changed accordingly.

```bash
python scripts/prepare_b1_planetoid.py --dataset Cora --output data/processed/cora_b1.npz
python main.py b1-construction --config configs/b1_construction_npz.yaml
```

The follow-up graph-fusion pilot uses the same Cora archive, quantum/Markov
construction budget, node split, features and decoder. It adds three arms:
original-graph propagation mixed with the quantum sparse hypergraph, the
matched Markov sparse hypergraph, or the identity operator. The identity arm
checks whether any gain merely comes from retaining more of each node's own
features. For each arm, the fusion weight and logistic-regression `C` are
chosen jointly by validation LogLoss from the predeclared grids; test metrics
are computed only for the chosen pair. The report records the validation curve,
chosen weight and saved propagation operator for each arm. This is an
exploratory follow-up because the earlier Cora test result has already been
inspected.

```bash
python main.py b1-construction --config configs/b1_construction_npz_fusion.yaml
```

To inspect every predeclared fusion weight rather than only the
validation-selected deployment model, run the fixed-alpha exploratory protocol:

```bash
python main.py b1-construction --config configs/b1_construction_npz_fusion_fixed.yaml
```

For each of the three fusion families, `evaluation.fixed_alpha_results` in
`report.json` contains validation and test Accuracy, Macro-F1 and LogLoss for
every configured alpha. Each fixed alpha independently selects only `C` on
validation; the original `selected_alpha` result remains separate. Do not
choose a preferred alpha by looking at these test rows: Cora's test outcomes
have already informed this follow-up, so the grid table is exploratory.

The B1 output directory stores the resolved config, source hashes, split
indices, constructed incidence and operator matrices, per-graph scores and
resource records. Family construction times exclude shared input/Hamiltonian
preparation and downstream classifier fitting; the report also gives a
whole-run processing time and sampled stage RSS. Very brief allocation peaks
can be missed. `results/` is Git-ignored because large experiment outputs
should be selected for versioning deliberately. The B1 code does not modify
any P1 recovery protocol or prior result.

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
