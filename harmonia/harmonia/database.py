"""Load and query the compatibility database."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_db(name: str = "pytorch") -> dict:
    """Load a compatibility database by name."""
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_all_dbs() -> dict[str, dict]:
    """Load every .json database file from the data directory."""
    dbs = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        with open(path) as f:
            dbs[path.stem] = json.load(f)
    return dbs


# ---------------------------------------------------------------------------
# PyTorch ecosystem
# ---------------------------------------------------------------------------

def get_torch_versions(db: dict) -> list[str]:
    versions = list(db["packages"]["torch"].keys())
    versions.sort(key=_version_tuple, reverse=True)
    return versions


def get_version_info(db: dict, torch_version: str) -> Optional[dict]:
    return db["packages"]["torch"].get(torch_version)


def find_compatible_torch(
    db: dict,
    python_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
) -> list[dict]:
    results = []
    for version, info in db["packages"]["torch"].items():
        if python_version:
            if _normalize_python(python_version) not in info["python"]:
                continue
        if cuda_version:
            if cuda_version not in info.get("cuda", []):
                continue
        results.append({"version": version, **info})
    results.sort(key=lambda r: _version_tuple(r["version"]), reverse=True)
    return results


def find_companions(db: dict, torch_version: str) -> Optional[dict]:
    info = get_version_info(db, torch_version)
    if not info:
        return None
    return {
        "torch": torch_version,
        **info["companions"],
        "python": info["python"],
        "cuda": info["cuda"],
        "install_hint": info.get("install_hint", ""),
    }


# ---------------------------------------------------------------------------
# Transformers ecosystem
# ---------------------------------------------------------------------------

def get_transformers_versions(db: dict) -> list[str]:
    versions = list(db["packages"]["transformers"].keys())
    versions.sort(key=_transformers_sort_key, reverse=True)
    return versions


def get_transformers_info(db: dict, version: str) -> Optional[dict]:
    packages = db["packages"]["transformers"]
    if version in packages:
        return packages[version]
    for key, info in packages.items():
        if "version_range" in info:
            if _version_in_range(version, info["version_range"]):
                return info
    return None


def find_compatible_transformers(
    db: dict,
    python_version: Optional[str] = None,
    torch_version: Optional[str] = None,
) -> list[dict]:
    results = []
    for version, info in db["packages"]["transformers"].items():
        if python_version:
            if _normalize_python(python_version) not in info["python"]:
                continue
        if torch_version:
            torch_min = info.get("torch_min", "0.0.0")
            if _version_tuple(torch_version) < _version_tuple(torch_min):
                continue
        results.append({"version": version, **info})
    results.sort(key=lambda r: _transformers_sort_key(r["version"]), reverse=True)
    return results


def get_known_conflicts(db: dict) -> list[dict]:
    return db.get("known_conflicts", [])


# ---------------------------------------------------------------------------
# Version utilities (exported for use by checker)
# ---------------------------------------------------------------------------

def normalize_python(version: str) -> str:
    """'3.10.12' -> '3.10'."""
    parts = version.split(".")
    return f"{parts[0]}.{parts[1]}"

# Keep underscore alias for internal compat
_normalize_python = normalize_python


def version_tuple(v: str) -> tuple[int, ...]:
    """Convert '2.5.1' to (2, 5, 1)."""
    parts = []
    for p in v.split("."):
        match = re.match(r"(\d+)", p)
        if match:
            parts.append(int(match.group(1)))
    return tuple(parts) if parts else (0,)

_version_tuple = version_tuple


def satisfies_constraint(version: str, constraint: str) -> bool:
    """Check if version satisfies a pip-style constraint: '>=1.0.0,<2.0.0'."""
    for part in constraint.split(","):
        part = part.strip()
        if not part:
            continue
        if not _version_in_range(version, part):
            return False
    return True


def _transformers_sort_key(v: str) -> tuple[int, ...]:
    if v.endswith(".x"):
        return (int(v.split(".")[0]), 999)
    return _version_tuple(v)


def _version_in_range(version: str, range_spec: str) -> bool:
    v = _version_tuple(version)
    match = re.match(r"(>=|<=|>|<|==)(.+)", range_spec)
    if not match:
        return False
    op, target = match.groups()
    t = _version_tuple(target)
    ops = {">=": v >= t, "<=": v <= t, ">": v > t, "<": v < t, "==": v == t}
    return ops.get(op, False)
