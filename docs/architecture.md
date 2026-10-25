# HVL-Interact architecture

HVL-Interact implements the mathematical method in *Hierarchical Vision–Language Alignment for
Dual-Person Interaction Action Recognition*. Public identifiers remain NTU action IDs throughout
the pipeline. Local classifier indices exist only inside the training objective.

## Computational contract

1. Read NTU skeleton files by persistent body ID. Keep the two tracks with the most valid
   observations, using motion variance and body ID to break ties. Tracking state zero is invalid;
   a tracked coordinate at the origin is valid.
2. Trim empty boundary frames, resample with validity-aware linear interpolation, and subtract
   the shared midpoint of the two root joints. Normalize both people by the same median
   spine-base-to-spine-shoulder length. Missing roots use the available root or an interpolated
   shared origin. No valid roots or no positive body scale is an explicit error.
3. A residual graph/temporal-convolution encoder produces features with shape
   `[batch, person, time, joint, width]`. Separate temporal attention and masked pooling produce
   50 joint tokens and 10 body-group tokens. Combined token attention exchanges information
   across both people and both spatial resolutions.
4. Render uniformly sampled frames into a fixed sequence-level viewport. Frozen local CLIP
   image features are averaged before a trainable projection. CLIP uses evaluation mode even
   while the rest of the network is trained.
5. Independent text projections transform frozen CLIP descriptions at fine, coarse and global
   levels. Project each paraphrase before averaging and L2 normalization, as in the manuscript.
6. Fine and coarse pools produce normalized semantic queries. An observation-only gate fuses
   projected fine, coarse and visual features into the normalized global query. Neither labels nor
   label-indexed descriptions enter this computation.
7. Optimize smoothed classifier cross-entropy plus beta times weighted class-prototype
   cross-entropy. Semantic negatives are classes, not other instances in a minibatch. Strict
   zero-shot inference uses weighted cosine scores over the six declared unseen candidates.

## Experiment boundaries

Supervised experiments use all 26 interaction classes. Zero-shot experiments initialize a separate
network and use 20 seen classes for training and validation. The user supplies six distinct unseen
IDs and the official X-Sub or X-Set protocol. There is no automatic unseen-class selection.
Validation groups keep all cameras of a setup/performer/replication together and come exclusively
from the official training partition. Official test samples never select checkpoints.

The split manifest, dataset index, prompt bank, configuration, CLIP files and software versions are
fingerprinted. Checkpoints omit the external frozen CLIP tensors and verify their fingerprint when
loaded. Epoch-boundary resume restores optimizer, scaler, global step and per-rank random states.
Evaluation shards do not pad or duplicate samples.

## Implementation choices

The manuscript does not fix the skeleton backbone depth, exact grouping indices, image count,
augmentation magnitudes or per-level weights. These are explicit configuration choices, documented
in `method_mapping.md`; they are not inferred from an aggregate accuracy or parameter count.
The implementation provides the stated equations and protocol, not a claim that a historical
checkpoint or historical accuracy has been reconstructed.

