# Contributing to harmonia

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/yourname/harmonia.git
cd harmonia
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Adding new PyTorch versions

When a new PyTorch version is released:

1. Check the [official compatibility page](https://pytorch.org/get-started/previous-versions/)
2. Edit `harmonia/data/pytorch.json`
3. Add the new version entry with Python, CUDA, and companion versions
4. Run `pytest` to verify
5. Submit a PR

Example entry:

```json
"2.6.0": {
  "python": ["3.9", "3.10", "3.11", "3.12", "3.13"],
  "cuda": ["11.8", "12.1", "12.4"],
  "companions": {
    "torchaudio": "2.6.0",
    "torchvision": "0.21.0"
  },
  "install_hint": "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
}
```

## Adding new transformers versions

1. Check [HuggingFace docs](https://huggingface.co/docs/transformers/installation) and the release notes
2. Edit `harmonia/data/transformers.json`
3. Add version with Python range, torch minimum, and companion constraints
4. Add any known conflicts to the `known_conflicts` array
5. Run `pytest` and submit a PR

## Adding a new known conflict

If you hit a version conflict that others will hit too:

1. Add it to the `known_conflicts` array in the relevant data file
2. Include the exact error message (helps users match their problem)
3. Include the fix
4. Add a test case in `tests/test_core.py`

```json
{
  "description": "Short description of the conflict",
  "packages": {"package_a": ">=X.Y.Z", "package_b": "<A.B.C"},
  "error": "The exact error message users see",
  "fix": "How to fix it"
}
```

## Adding a new ecosystem

To add support for a new package ecosystem (e.g., `jax`, `tensorflow`):

1. Create `harmonia/data/newecosystem.json` following the existing format
2. Add query functions in `database.py`
3. Add check logic in `checker.py`
4. Add CLI commands in `cli.py`
5. Add tests
6. Update the README

## Code style

- Zero runtime dependencies (stdlib only)
- Type hints everywhere
- Tests for every check path
- Keep JSON data files easy to read and edit by hand
