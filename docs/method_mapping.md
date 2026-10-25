# Manuscript-to-code mapping

| Manuscript operation | Implementation | Concrete behavior |
| --- | --- | --- |
| Shared normalization, Eqs. 2–3 | `data/preprocess.py:normalize_shared` | Frame-wise shared root midpoint; sequence-wide median body scale |
| Spatiotemporal encoder, Eq. 4 | `models/skeleton.py:SpatialTemporalBlock` | Validity-normalized skeleton neighbors, linear graph map, temporal convolution, residual |
| Temporal attention and pooling, Eqs. 5–6 | `models/skeleton.py:TemporalPool` | Sinusoidal time positions, masked self-attention, explicit masked mean |
| Group pooling, Eqs. 7–8 | `models/skeleton.py:HierarchicalSkeletonEncoder` | Five column-normalized groups; renormalization when joints are missing |
| Combined tokens, Eq. 9 | `models/skeleton.py:HierarchicalSkeletonEncoder` | 50 joint and 10 group tokens, learned token identities, joint token attention |
| Skeleton rendering, Eq. 11 | `rendering.py:SkeletonRenderer` | Shared XY front projection and square viewport over the entire sequence |
| Frozen image encoding, Eq. 12 | `models/clip.py:FrozenCLIP`, `models/network.py:HVLInteract` | Average raw per-frame CLIP features, trainable visual projection, L2 normalization |
| Text prototypes, Eq. 14 | `models/clip.py:encode_descriptions`, `models/network.py:project_text` | Frozen text features, per-level projection, paraphrase average, L2 normalization |
| Fine/coarse queries, Eq. 15 | `models/network.py:HVLInteract.forward` | Masked token pools and separate normalized projections |
| Hierarchical alignment, Eqs. 16–17 | `objectives.py:HierarchicalObjective` | Cross-entropy over permitted class prototypes, with no instance-level false negatives |
| Gated fusion, Eqs. 19–20 | `models/network.py:HVLInteract.forward` | Shared tanh gate over fine, coarse and visual streams; no target text input |
| Classification and total loss, Eqs. 21–22 | `objectives.py:HierarchicalObjective` | Smoothed CE plus beta times weighted hierarchical alignment |
| Zero-shot scoring, Eq. 23 | `objectives.py:semantic_scores` | Weighted cosine scores; nonnegative weights normalized to sum to one |
| Training protocol, Algorithm 1 | `engine/training.py`, `data/protocols.py` | Fresh seen-only model, seen-only validation, fixed unseen candidate scoring |

## Explicit defaults

The manuscript specifies 120 frames, 25 joints, two people, five groups, skeleton width 512,
CLIP ViT-B/16, AdamW, 110 epochs, 10 warmup epochs, learning rate 0.001, weight decay 0.05,
temperature 0.07, smoothing 0.1 and beta 0.5. These settings are retained.

The following choices complete details not determined by the manuscript:

| Setting | Default |
| --- | --- |
| Backbone widths | 128, 256, 512; one residual graph/temporal block per width |
| Temporal convolution | Kernel 9, stride 1, same-length padding |
| Fine and coarse temporal attention | One layer each; eight heads; feed-forward width 4D |
| Combined token attention | Two layers, eight heads |
| Projection heads | Linear, GELU, Linear; shared semantic width 512 |
| Normalization | Frame-wise shared center; median spine-base/spine-shoulder length |
| Interpolation | Linear only between observed endpoints; no interpolation through an invalid bracket |
| Rendered frames | Eight uniformly spaced frames; repeated indices allowed for short sequences |
| Rendering | XY, Y up; 224 px; white background; blue/orange participants; 8% margin |
| Text prompts | One description per class per level; same paraphrase count per level across classes |
| Alignment weights | Fine/coarse/global = 1/1/1, without implicit averaging |
| Semantic scoring weights | Fine/coarse/global = 1/3 each |
| Learning-rate minimum | 0.000001 |
| Validation | 15% of training capture groups, deterministic hash ordering |
| Augmentation | Shared yaw ±15 degrees; normalized noise std 0.005; masking 0.01; person swap 0.5; temporal crop ≥0.9; jitter 0.15 bins |

Body groups use zero-based indices:

| Group | Indices |
| --- | --- |
| Head | 2, 3 |
| Torso | 0, 1, 20 |
| Left arm | 4, 5, 6, 7, 21, 22 |
| Right arm | 8, 9, 10, 11, 23, 24 |
| Legs | 12, 13, 14, 15, 16, 17, 18, 19 |

These default groups form a partition. A configuration may specify five nonempty groups covering
all joints, including overlapping groups. Names remain `head`, `torso`, `left_arm`, `right_arm` and
`legs`; the model uses this fixed order regardless of YAML key order. Membership is saved in the configuration.
No left/right reflection is applied. Prompts use role-neutral, side-neutral descriptions.
Object names in global category descriptions denote the category semantics; the renderer does
not observe or reconstruct objects. Fine and coarse descriptions emphasize body motion.

## Controlled variants

For a skeleton-only supervised experiment, set `model.use_visual: false` and `loss.beta: 0`.
For a skeleton-plus-rendering supervised experiment, retain the visual branch and set beta to zero.
For global-only semantic alignment, use `level_weights: [0, 0, 1]` and
`semantic_weights: [0, 0, 1]`. For hierarchical alignment, enable all three weights.
Use a separate output directory for every variant. Frozen CLIP remains resident for the text
interface and identity check; parameter reporting explicitly distinguishes frozen and trainable
parameters. A conceptual branch ablation does not automatically reproduce any historical parameter
table or resolve an unspecified nearest-neighbor baseline.

An audit of original experiment logs is still needed before associating published aggregate
numbers with this implementation. Counts, timings and accuracies must come from actual runs.

