# Data handling

Raw external data are not downloaded by the code. Place verified local files
under `data/raw/` and use `scripts/02_prepare_real_data.py` with a versioned
source ID and exact source URL. Its output separates `observed/` from `truth/`
and includes hashes and an unsplit manifest. A time-aware or independent-graph
train/test protocol must be fixed before those truth records are used for a
formal reconstruction experiment.

Synthetic graphs can be materialized with `scripts/01_generate_synthetic.py`.
The default P1 runner regenerates them from declared seeds and configuration,
then writes a run-level manifest. Generated data and raw data are ignored by
Git; retain manifests and any released data under their applicable license.
