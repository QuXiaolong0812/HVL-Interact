# HVL-Interact

**Hierarchical Vision–Language Alignment for Dual-Person Interaction Action Recognition**

The name **HVL-Interact** combines **H**ierarchical **V**ision–**L**anguage alignment
with **Interact**ion action recognition.

PyTorch implementation of dual-resolution skeleton tokens, rendered-skeleton CLIP features,
hierarchical text supervision and observation-only fusion for NTU RGB+D 120 interaction recognition.

[中文说明](README.zh-CN.md) · [Method mapping](docs/method_mapping.md) ·
[Architecture](docs/architecture.md) · [Data protocol](docs/data_protocol.md)

## Features

- 50 joint tokens and 10 body-group tokens with masked temporal and cross-person token attention.
- Frozen local CLIP image/text encoders and three trainable semantic projection heads.
- Class-prototype contrastive learning, label smoothing and label-independent gated fusion.
- Official X-Sub/X-Set evaluation and explicit 20-seen/6-unseen experiments.
- Single-device or `torchrun` training, AMP, gradient accumulation and epoch-boundary resume.
- Content fingerprints, per-sample scores, confusion matrices and actual parameter accounting.

This repository implements the manuscript's mathematical formulation. Some architecture and
preprocessing details were not fixed by the manuscript; their concrete choices are documented and
configured explicitly. Benchmark accuracies must be obtained by training and evaluating the code.
The repository does not bundle historical experiment checkpoints or claim their reconstruction.

## Installation

Use Linux with Python 3.10–3.12. Install a PyTorch build matching the system CUDA driver, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[dev]'
hvl-interact --help
```

For CPU checks, use `https://download.pytorch.org/whl/cpu` for the PyTorch installation.
The dependency ranges are in `pyproject.toml`; `requirements/validated.txt` records the reference
package versions used for verification. Training configuration files are relative
to their `root` field, which is resolved against the directory containing the configuration file.

## Local assets

```text
data/
  raw/
    nturgb+d_skeletons/                  # extracted NTU 60 and NTU 120 skeleton files
    NTU_RGBD120_samples_with_missing_skeletons.txt
  processed/ntu120/                     # index.json and per-sequence NPZ files
  splits/                              # experiment manifests
weights/
  clip-vit-base-patch16/                # local Transformers CLIP checkpoint
  checkpoints/                         # separately distributed HVL-Interact checkpoints
runs/                                  # logs, checkpoints and evaluation outputs
resources/prompts/ntu120_interactions.json
```

Obtain the dataset from the [official NTU page](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
and its exclusion list from the
[official repository](https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt).
Use both the original and additional NTU skeleton archives. Only A050–A060 and A106–A120 are retained.

Download [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16) into
`weights/clip-vit-base-patch16/`. Include `config.json`, `model.safetensors` or
`pytorch_model.bin`, `preprocessor_config.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.json` and `merges.txt`; include `tokenizer.json` if distributed
with the model. Runtime loading uses `local_files_only=True`; it never downloads model assets.
The encoder fingerprint covers weight, configuration and tokenizer files.

NTU data and model checkpoints follow their respective distribution terms and are excluded from
source control. The code license does not apply to third-party datasets or model weights.

## Prepare the dataset

```bash
hvl-interact prepare \
  --skeleton-root data/raw/nturgb+d_skeletons \
  --excluded data/raw/NTU_RGBD120_samples_with_missing_skeletons.txt \
  --output data/processed/ntu120
```

The index records accepted samples, exclusion reasons, tracking identities and SHA-256 hashes.
Coordinates stay in their original units in NPZ files. Resampling, shared normalization and
training augmentation run in the dataset loader. All input archives use `allow_pickle=False`.
Existing dataset indices and manifests require `--overwrite` to replace them.

## Supervised training

```bash
hvl-interact split --index data/processed/ntu120/index.json \
  --output data/splits/supervised_xsub.json --mode supervised --protocol xsub
hvl-interact doctor --config configs/supervised_xsub.yaml
torchrun --standalone --nproc_per_node=2 -m hvl_interact train --config configs/supervised_xsub.yaml
hvl-interact evaluate --config configs/supervised_xsub.yaml --checkpoint runs/supervised_xsub/best.pt
```

Use `supervised_xset.yaml` and `--protocol xset` to create and run X-Set experiments.
For one device, run `hvl-interact train --config configs/supervised_xsub.yaml`.
`batch_size` is per process. The effective global batch size is
`batch_size × world_size × accumulation_steps`; the supplied two-process settings use 32 samples.
Reduce the per-process batch size and increase accumulation for smaller GPUs. Rendered CLIP
frames are encoded in chunks of 64 images. CPU execution uses FP32.

## Zero-shot training

Choose six official interaction IDs and one evaluation protocol before running an experiment.
The following commands request those choices explicitly:

```bash
read -r -p 'Protocol (xsub/xset): ' PROTOCOL
read -r -p 'Six unseen action IDs separated by spaces: ' -a UNSEEN_IDS
hvl-interact split --index data/processed/ntu120/index.json \
  --output "data/splits/zeroshot_${PROTOCOL}.json" \
  --mode zeroshot --protocol "$PROTOCOL" --unseen "${UNSEEN_IDS[@]}"
torchrun --standalone --nproc_per_node=2 -m hvl_interact train \
  --config configs/zeroshot.yaml --protocol "$PROTOCOL" \
  --split "data/splits/zeroshot_${PROTOCOL}.json" --output "runs/zeroshot_${PROTOCOL}"
hvl-interact evaluate --config configs/zeroshot.yaml --protocol "$PROTOCOL" \
  --split "data/splits/zeroshot_${PROTOCOL}.json" --output "runs/zeroshot_${PROTOCOL}" \
  --checkpoint "runs/zeroshot_${PROTOCOL}/best.pt"
```

Unseen samples in the official training partition are excluded. Training and validation use only
seen classes. Evaluation uses unseen classes in the official test partition and restricts candidates
to the six declared IDs. A fresh model is initialized for each experiment. A fully supervised
26-class skeleton checkpoint is not used to initialize a zero-shot experiment.

## Resume and inference

```bash
hvl-interact train --config configs/supervised_xsub.yaml --resume runs/supervised_xsub/last.pt
hvl-interact predict --checkpoint runs/supervised_xsub/best.pt \
  --clip weights/clip-vit-base-patch16 \
  --prompts resources/prompts/ntu120_interactions.json \
  --input data/processed/ntu120/samples/S001C001P003R001A050.npz \
  --output runs/prediction.json
```

The inference input must contain `coordinates: float32[2,T,25,3]` and
`valid: bool[2,T,25]`. No class label is required. The illustrated sample filename is an input
format example; use a sequence present in the local dataset. Inference uses preprocessing saved
in the checkpoint and does not require the training dataset index.

Resume is exact at epoch boundaries with the same configuration, inputs, process count and
software/hardware environment.
`--stop-after-epoch N` can end a job early while retaining the original total-epoch learning-rate
schedule. Resume from `last.pt` in the same run directory. External frozen CLIP tensors are not
duplicated in HVL-Interact checkpoints; the matching local CLIP directory is required when loading them.

## Outputs

Each run contains `run.json`, `history.jsonl`, `last.pt` and `best.pt`. The best checkpoint is
selected by seen-only validation Top-1 accuracy. Zero-shot validation uses semantic scores over
seen candidates. `evaluation/test/` contains:

- `metrics.json`: micro Top-1, macro accuracy, per-class support and provenance.
- `predictions.jsonl`: sample IDs, true/predicted official class IDs and raw candidate scores.
- `confusion.csv`: integer counts, with true classes as rows and predictions as columns.

Evaluation shards contain each sample exactly once, including uneven distributed partitions.
Parameter reports distinguish trainable, frozen and total resident parameters. They are computed
from the instantiated network rather than copied from a paper table.

## Development

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
ruff check src tests
ruff format --check src tests
pytest -q
python -m build
```

Tests include real local CLIP forward passes with a small test configuration, optimizer updates,
bitwise epoch-resume checks, class-leakage rejection and two-process Gloo training/evaluation.
They do not need the NTU dataset or a downloaded pretrained encoder.

## References

- [NTU RGB+D 120 and official subject/setup protocols](https://arxiv.org/abs/1905.04757)
- [CLIP](https://github.com/openai/CLIP)
- [Transformers CLIP implementation](https://huggingface.co/docs/transformers/model_doc/clip)

See `CITATION.cff` for software citation metadata and `LICENSE` for the MIT code license.

