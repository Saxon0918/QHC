# Data handling

Raw external data are not downloaded by the code. The two temporal P1 sources
are the Cornell [email-Enron](https://www.cs.cornell.edu/~arb/data/email-Enron/)
archive restricted to events of at most 25 members and the separate
[contact-high-school](https://www.cs.cornell.edu/~arb/data/contact-high-school/)
temporal archive. The release pages link Google Drive tarballs; they do not
provide three verified text-file URLs. The pages do not state a redistribution
license. Keep archives outside Git and consult the release pages for citation
and access terms.

After downloading the two named tarballs, run `scripts/06_unpack_cornell_events.py`
once per archive. It accepts archive members ending in `nverts.txt`,
`simplices.txt`, and `times.txt`, checks that exactly one regular file of each
kind exists, and copies their contents to canonical names under `data/raw/`.
It records the archive and extracted-file SHA-256 hashes and observed extraction
time. Supply `--downloaded-at-utc` if the actual download time is known; a
missing value remains explicitly unknown. The real P1 runner checks these
hashes before using the files.

The older `scripts/02_prepare_real_data.py` prepares one unsplit local Cornell
dataset with separated `observed/` and `truth/` files. The temporal P1 runner
instead reads raw event files, fixes disjoint complete-event time windows, and
constructs each window's projection independently. It does not treat repeated
member sets as separate positives in its main binary target.

Synthetic graphs can be materialized with `scripts/01_generate_synthetic.py`.
The default P1 runner regenerates them from declared seeds and configuration,
then writes a run-level manifest. Generated data and raw data are ignored by
Git; retain manifests and any released data under their applicable license.
