# harmonia

**Bring harmony to your ML dependency stack.**

`harmonia` detects your GPU, CUDA, driver, OS, Python, and installed packages — then tells you exactly what's compatible with what. Zero dependencies. Works offline.

```
pip install harmonia-ml
```

## The problem

You install `torch`, then `transformers`, then `accelerate` — and something breaks. You get a cryptic `AttributeError` or a silent CUDA mismatch. You spend an hour on GitHub issues to discover your driver doesn't support the CUDA version your PyTorch was built for.

**harmonia fixes this in one command.**

## Usage

### `harmonia check` — full environment scan

```bash
$ harmonia check

  _                                  _
 | |__   __ _ _ __ _ __ ___   ___  _ __ (_) __ _
 | '_ \ / _` | '__| '_ ` _ \ / _ \| '_ \| |/ _` |
 | | | | (_| | |  | | | | | | (_) | | | | | (_| |
 |_| |_|\__,_|_|  |_| |_| |_|\___/|_| |_|_|\__,_|

System:
  OS:       Ubuntu 22.04 (jammy)
  Arch:     x86_64
  glibc:    2.35

Python:
  Version:  3.10.12
  Path:     /home/user/.venv/bin/python
  Env:      venv:myproject

GPU (1 detected):
  [0] NVIDIA RTX 4090 (24564 MB)

CUDA:
  Driver:     535.54.03
  CUDA (smi): 12.2
  CUDA (nvcc):12.1
  CUDA (torch):12.1
  cuDNN:      8.9.7

Packages:
  torch==2.5.1
  torchaudio==2.3.0
  torchvision==0.20.1
  transformers==4.44.2
  accelerate==0.30.0

────────────────────────────────────────────────────────
❌ 1 error(s):
  [torchaudio] torchaudio==2.3.0 is NOT compatible with torch==2.5.1.
    ↳ Expected torchaudio==2.5.1

📦 Recommended compatible set:
  torch==2.5.1
  torchaudio==2.5.1
  torchvision==0.20.1
```

### `harmonia doctor` — deep system diagnostics

Inspects your GPU, CUDA driver chain, glibc, virtualenv, and checks for mismatches:

```bash
$ harmonia doctor
```

### `harmonia suggest` — find compatible versions

```bash
# What goes with torch 2.5.1?
harmonia suggest torch==2.5.1

# What goes with transformers 4.44.2?
harmonia suggest transformers==4.44.2

# Best stack for my Python + CUDA?
harmonia suggest transformers --python 3.11 --cuda 12.1
```

### `harmonia matrix` — compatibility tables

```bash
# PyTorch ecosystem
harmonia matrix pytorch

# Transformers ecosystem
harmonia matrix transformers

# As JSON (for scripts)
harmonia matrix pytorch --json
```

### `harmonia conflicts` — known conflict database

```bash
$ harmonia conflicts

  1. transformers>=4.38 breaks with torch<1.13
     When: transformers>=4.38.0 + torch<1.13.0
     Error: AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
     Fix: Upgrade torch to >=1.13.0 or downgrade transformers to <4.38.0

  2. transformers>=5.0 requires torch>=2.1 (is_torch_fx_available removed)
     ...
```

## What it checks

| Check | What it detects |
|---|---|
| **OS / glibc** | Distro, version, glibc compatibility with manylinux wheels |
| **Python** | Version, virtualenv, EOL warnings |
| **GPU** | Model, VRAM, count via `nvidia-smi --query-gpu` |
| **CUDA driver** | nvidia-smi driver version, max CUDA supported |
| **CUDA toolkit** | nvcc version, driver compatibility |
| **torch CUDA** | PyTorch's built-in CUDA vs driver mismatch |
| **cuDNN** | Version detection via torch.backends |
| **PyTorch** | torch ↔ torchvision ↔ torchaudio version lock |
| **Transformers** | torch minimum, accelerate, tokenizers constraints |
| **Known conflicts** | Pattern-matched against real-world bug reports |

## Use as a library

```python
from harmonia.checker import check, suggest
from harmonia.detector import detect

# Full environment scan
env = detect()
print(env.summary())
print(f"GPU: {env.gpus[0].name if env.gpus else 'none'}")
print(f"CUDA: {env.cuda_info.best_version}")

# Check for issues
result = check()
for issue in result.issues:
    print(f"{issue.severity}: [{issue.package}] {issue.message}")

# Suggest compatible versions
versions = suggest(package="transformers", version="4.44.2")
```

## JSON output

For CI pipelines and scripts:

```bash
harmonia check --json
```

Returns structured JSON with errors, warnings, and environment info.

## [OpenClaw Skill](https://clawhub.ai/ahmed-eladl/harmonia)

harmonia is available as an [OpenClaw](https://github.com/openclaw/openclaw) skill. Install it and your AI agent can diagnose ML environments through chat:

```bash
clawhub install harmonia
```

Then just tell your agent: *"check my ML environment"*, *"what torch works with transformers 5?"*, or *"I'm getting a CUDA error"* — it runs harmonia and gives you the answer.

## Contributing

The compatibility databases live in `data/*.json`. To add new versions or ecosystems:

1. Edit the relevant JSON file in `data/`
2. Run `pytest` to verify
3. Submit a PR

## Zero dependencies

harmonia has **no runtime dependencies**. It uses only the Python standard library. It detects packages through `importlib.metadata` and system tools through subprocess. It works offline, costs nothing, and installs instantly.

## License

MIT
