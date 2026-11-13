# P0/P1 protocol and implementation decisions

This file interprets `RESEARCH_PLAN.md` for the first delivery. It records
choices before treating any result as a formal scientific comparison.

## Current scope

1. P0 verifies equations (1)–(4), unitarity, double stochasticity, permutation
   equivariance, degeneracies and the `K3` projection ambiguity. Dense `expm`
   is a small-graph oracle; normal feature extraction uses batched sparse
   `expm_multiply` and bounded temporary state columns.
2. P1 generates independent S1/S2 graphs at declared seeds. Every method gets
   the same observed binary `A`, fixed clique candidates, labels and split.
   S3 is a separate negative-identifiability check. There are no node features
   in this primary protocol.
3. C0 includes the exact second-order term, external common-neighbor and
   boundary statistics. Heat, wave, spectral walks and pairwise quantum
   controls use the same candidates. Q2 separates response components. Q3
   includes the specified `[R_coh,R_inc,C]` time grid; its independent-column
   variant diagnoses the effect of the linearly redundant `C` column under
   regularized logistic scoring. Q4 uses the random-phase mean without its
   sampling variance as a predictor.
4. Each variant uses `StandardScaler + LogisticRegression`, with the same
   prespecified validation grid for regularization. AP is primary; AUROC,
   Brier, log loss and per-size AP are secondary. Each test graph gets its own
   Precision@Budget. Paired AP differences resample whole test graphs rather
   than individual correlated candidates.
   The validation threshold maximizes **candidate-level** F1. Exact-set
   reconstruction reported by the P1 script is a diagnostic under that
   threshold, not a validation-optimized Gate 2 result.
5. The smoke run is for code validation only. Its small number of graph
   clusters makes confidence intervals unstable. Gate 1 remains unevaluated
   until two real-data protocols, fixed minimum meaningful effect and adequate
   replication are added. Do not tune the method on smoke test results and
   later present those same graphs as a clean holdout.

## Data boundaries

The synthetic generator preserves full event lists, creates the binary clique
projection and optionally computes the multiplicity projection under a
separate switch. The principal metric deduplicates equal member sets. The
candidate generator receives an `observed()` graph without truth. The
feature cache contains candidate identities and numeric features, never
labels. Ground truth is read only after cache creation for allowed train
labels, validation decisions and held-out evaluation.

The default size range is 3–5. S1 also contains genuine size-2 pair events
that deliberately form false triangular clique candidates. Consequently,
candidate recall and exact-set recall cannot reach 1 under this scope. The
report gives both actual coverage and the ceiling implied by excluded sizes.
For school contact data, a distinct preregistered size-2-inclusive protocol
is necessary. Cornell temporal data must be loaded from exact versioned event
files; the SNAP ordinary graph with the same `email-Enron` label is not an
acceptable substitute.

## Limits on interpretation

Projection is many-to-one. The group statistic is a sum of pairwise complex
amplitude cross terms conditioned on a candidate set, not proof of an
irreducible physical multi-particle interaction. Classically simulating a
unitary does not give quantum computational speedup. The short-time signal
may be explained by weighted common external neighbors. Favor a negative or
conditional result over an unsupported "quantum advantage" claim.

This repository currently has no real-data reconstruction result and no
downstream structure-learning result. The approximate exact-set diagnostic in
the P1 report is not a Gate 2 result because the first protocol is synthetic
and the full reconstruction model/baselines have not been evaluated.
