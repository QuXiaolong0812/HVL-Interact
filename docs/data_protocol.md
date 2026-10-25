# NTU data and split contract

The [official NTU RGB+D 120 paper](https://arxiv.org/html/1905.04757#S3.SS2) defines the
subject/setup partitions. The [dataset repository](https://github.com/shahroudy/NTURGB-D)
defines sample IDs, joint records and the incomplete-skeleton exclusion list.

## Indexed sequences

`prepare` accepts an extracted skeleton tree and the official incomplete-skeleton list. It filters
the 26 interaction IDs, preserves persistent body tracking IDs, rejects truncated records and keeps
the two tracks with the largest valid-joint counts. Motion variance and body ID break ties.
Both fully tracked and inferred joints are accepted. A tracked `(0,0,0)` is valid; untracked joints
use a separate boolean mask. A sequence with fewer than two nonempty tracks or no usable shared
root/body scale is recorded as rejected. No body-identity stitching is inferred across different IDs.

Each NPZ contains `coordinates[2,T,25,3]` and `valid[2,T,25]`. `index.json` contains their relative
paths and checksums, original file checksums, class IDs, body IDs and quality statistics.
The index also records exclusion reasons. Reading is pickle-free. Dataset paths must remain
within the processed-data root. Training and evaluation verify every indexed archive checksum.

## Partitions

- X-Sub: the 53 training subjects from Section III-B1 of the official NTU120 paper.
- X-Set: even setup IDs train; odd setup IDs test.
- Validation: deterministic group holdout within the official training side. A group comprises
  setup, performer and replication; all cameras remain together. Groups are hash-ordered with
  the requested seed. The held-out count is `ceil(fraction × group_count)`, bounded to leave
  at least one training group.
- Supervised: all 26 classes occur in training, validation and testing.
- Zero-shot: 20 seen classes occur in training and validation. Six explicitly declared unseen
  classes occur in testing. Seen official-test samples and unseen official-training samples are
  outside this strict zero-shot experiment.

Every eligible sample must be present exactly once in its permitted partition. Empty partitions,
missing class coverage, duplicates, capture-group overlap and official-protocol violations are
errors. Sparse datasets may fail class coverage under grouped validation; inspect the data or
choose a documented validation seed/fraction using training-side information only.

The split stores the dataset-index checksum. Its own checksum is recorded in every experiment.
Changing a split or prompt bank invalidates resume and evaluation against an existing checkpoint.
The six unseen IDs and protocol are never selected automatically.

## Temporal and geometric conventions

Empty boundary frames are removed. At each resampled timestamp, a joint is valid only if both
linear-interpolation endpoints are valid. Exact observed timestamps retain their own validity.
Both participants share the frame-wise root midpoint and a sequence-wide body scale.
An available single root defines the origin when the other is missing. Frames without roots use
linear interpolation of shared origins, with endpoint extension at the sequence boundaries.

The renderer fits all valid joints from the full normalized sequence to a single aspect-preserving
viewport. It then samples frames. The viewport is never refitted separately for each rendered frame.
Spatial and temporal training transforms are applied before both coordinate encoding and rendering.

