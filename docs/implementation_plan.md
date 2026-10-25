# HVL-Interact implementation plan

**Goal:** Deliver an installable, offline-capable PyTorch research repository with complete
NTU preparation, supervised and zero-shot training, evaluation and single-sequence inference.

**Architecture:** Separate dataset/protocol validation, geometry, skeleton tokens, frozen CLIP,
semantic objectives and experiment execution. YAML configurations describe every implementation
choice; immutable fingerprints connect checkpoints to their experiment inputs.

**Stack:** Python 3.10+, PyTorch 2.6+, Transformers 4.x, NumPy, Pillow, PyYAML, pytest and Ruff.

## Verification-driven tasks

- [x] Dataset: test persistent IDs, tracked zero coordinates, mask-preserving interpolation,
  shared-origin geometry and malformed-file rejection; implement `data/ntu.py` and
  `data/preprocess.py`. Run `pytest tests/test_data.py`.
- [x] Protocols: test official X-Sub/X-Set assignment, grouped validation, 20/6 requirements,
  duplicate rejection and forged split detection; implement `data/protocols.py`.
  Run `pytest tests/test_protocols.py`.
- [x] Geometry and network: test fixed viewport rendering, all-missing attention stability,
  query shapes, label-independent queries, trainable text projection gradients and frozen CLIP;
  implement `rendering.py` and `models/`. Run `pytest tests/test_model.py`.
- [x] Objectives: compare the alignment loss with hand-computed class-prototype cross-entropy;
  verify semantic candidate permutation and official-ID mapping. Implement `objectives.py`.
  Run `pytest tests/test_objectives.py`.
- [x] Experiment engine: test one-epoch training, checkpoint restore, independent zero-shot
  training, evaluation records and local CLIP loading using a real tiny CLIP test fixture.
  Implement `engine/`, configuration validation and CLI. Run `pytest tests/test_integration.py`.
- [x] Distribution: test disjoint unpadded evaluation shards and gradient accumulation;
  support `torchrun` with NCCL on Linux CUDA and Gloo on CPU.
- [x] Delivery: document setup, downloads, commands, method mapping and scientific limits;
  add CI, license, citation and contribution guides. Run Ruff, pytest, wheel/sdist builds,
  installed CLI checks and an archive integrity check.

