# Verification record

Date: 2026-10-25. Reference environment: Windows x86-64, Python 3.12.14,
PyTorch 2.6.0+cpu, Transformers 4.49.0, NumPy 2.3.5, Pillow 12.3.0,
PyYAML 6.0.2, safetensors 0.8.0 and pytest 8.3.5.

The automated suite completed with **28 passed, 0 failed, 0 skipped**. It checks:

- Shared-origin geometry, validity-aware interpolation and persistent body identities.
- Skeleton-file preparation and recorded quality exclusions.
- Official subject/setup boundaries, explicit 20/6 manifests and complete sample coverage.
- Fixed-viewport rendering, 50/10 token shapes and canonical body-group ordering.
- Finite half-precision zero embeddings and label-independent observation queries.
- Class-prototype loss values, candidate permutation and short-microbatch accumulation weights.
- Frozen local CLIP forward passes, training updates and per-sample evaluation artifacts.
- Bitwise equivalence of continuous and epoch-resumed CPU training in the reference environment.
- Two-process Gloo training of the full model and the skeleton-only variant, followed by
  distributed evaluation with unequal numbers of batches and exactly-once sample coverage.
- A 120-frame, 512-dimensional skeleton/fusion forward pass using explicit CLIP feature inputs.
- Invalid configuration keys, invalid stop epochs and stale resume-history rejection.
- Public package, model class and command naming.

The local CLIP integration fixture instantiates a small real Transformers CLIP network, tokenizer
and image processor. It does not use a downloaded pretrained ViT-B/16 checkpoint. Geometry
fixtures exist only inside temporary test directories. Test accuracies are not benchmark results.

Release checks use `ruff check src tests`, `ruff format --check src tests`, `python -m build`
and installed command-line entry points. Linux CI is configured for Ubuntu 22.04 with Python
3.10 and 3.12. That workflow has not been executed on a remote CI service in this workspace.

Full NTU120 training, CUDA/NCCL execution, pretrained ViT-B/16 behavior, benchmark accuracy and
hardware throughput require the actual local assets and training hardware. No such benchmark
result is asserted by this verification record.
