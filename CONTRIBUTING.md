# Contributing

Install the development dependencies with `python -m pip install -e '.[dev]'`.
Keep changes scoped to one behavior and include a regression test for numerical or protocol changes.

Run `ruff check src tests`, `ruff format --check src tests`, `pytest -q` and `python -m build`
before proposing a change. Tests run with local small CLIP configurations and geometry fixtures;
they do not download pretrained weights or the NTU dataset. Distributed tests use two CPU processes.

Describe configuration, split, prompt or metric changes explicitly. Preserve official action IDs
in exported predictions. Record measured experimental results with their checkpoint and run
metadata. Keep datasets, model weights, credentials and machine-specific paths out of commits.

