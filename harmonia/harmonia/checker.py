"""Check compatibility and suggest fixes across the ML stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from harmonia.database import (
    find_companions,
    find_compatible_torch,
    find_compatible_transformers,
    get_known_conflicts,
    get_transformers_info,
    get_version_info,
    load_db,
    normalize_python,
    satisfies_constraint,
    version_tuple,
)
from harmonia.detector import Environment, detect


class Severity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Issue:
    severity: Severity
    package: str
    message: str
    suggestion: str = ""


@dataclass
class CheckResult:
    environment: Environment
    issues: list[Issue] = field(default_factory=list)
    compatible_versions: Optional[dict] = None

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def summary(self) -> str:
        from harmonia import __banner__
        lines = [__banner__.rstrip()]
        lines.append("")
        lines.append(self.environment.summary())
        lines.append("")
        lines.append("─" * 56)

        if self.is_clean:
            lines.append("✅ All clear — no compatibility issues found!")
        else:
            errors = [i for i in self.issues if i.severity == Severity.ERROR]
            warnings = [i for i in self.issues if i.severity == Severity.WARNING]

            if errors:
                lines.append(f"❌ {len(errors)} error(s):")
                for issue in errors:
                    lines.append(f"  [{issue.package}] {issue.message}")
                    if issue.suggestion:
                        lines.append(f"    ↳ {issue.suggestion}")
                lines.append("")

            if warnings:
                lines.append(f"⚠️  {len(warnings)} warning(s):")
                for issue in warnings:
                    lines.append(f"  [{issue.package}] {issue.message}")
                    if issue.suggestion:
                        lines.append(f"    ↳ {issue.suggestion}")
                lines.append("")

        if self.compatible_versions:
            lines.append("📦 Recommended compatible set:")
            for pkg, ver in sorted(self.compatible_versions.items()):
                if pkg not in ("python", "cuda", "install_hint"):
                    lines.append(f"  {pkg}=={ver}")
            hint = self.compatible_versions.get("install_hint")
            if hint:
                lines.append(f"\n  Install command:\n  {hint}")

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check(env: Optional[Environment] = None) -> CheckResult:
    """Run a full compatibility check."""
    if env is None:
        env = detect()

    result = CheckResult(environment=env)

    _check_system(env, result)
    _check_cuda(env, result)
    _check_pytorch(env, result)
    _check_transformers(env, result)
    _check_known_conflicts(env, result)

    return result


# ---------------------------------------------------------------------------
# System-level checks
# ---------------------------------------------------------------------------

def _check_system(env: Environment, result: CheckResult) -> None:
    """Check OS-level compatibility."""
    os_info = env.os_info

    # glibc check — PyTorch wheels need glibc >= 2.17 (manylinux2014)
    if os_info.glibc_version:
        glibc = version_tuple(os_info.glibc_version)
        if glibc < (2, 17):
            result.issues.append(
                Issue(Severity.ERROR, "system",
                      f"glibc {os_info.glibc_version} is too old for PyTorch wheels.",
                      "PyTorch requires glibc >= 2.17 (manylinux2014). Upgrade your OS or build from source.")
            )

    # Python version sanity
    py = version_tuple(env.python_version)
    if py < (3, 8):
        result.issues.append(
            Issue(Severity.ERROR, "python",
                  f"Python {env.python_version} is end-of-life and unsupported by modern ML libraries.",
                  "Upgrade to Python 3.10+.")
        )
    elif py < (3, 9):
        result.issues.append(
            Issue(Severity.WARNING, "python",
                  f"Python {env.python_version} — most ML libraries are dropping 3.8 support.",
                  "Consider upgrading to Python 3.10+.")
        )

    # No virtualenv warning
    if not env.venv:
        result.issues.append(
            Issue(Severity.WARNING, "system",
                  "Not running inside a virtual environment.",
                  "Use `python -m venv .venv` or conda to isolate your project.")
        )


# ---------------------------------------------------------------------------
# CUDA / GPU checks
# ---------------------------------------------------------------------------

# Map: NVIDIA driver version → max CUDA version it supports
# Source: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
DRIVER_CUDA_MAP = [
    # (min_driver, max_cuda)
    # Source: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
    ("580.00", "13.0"),
    ("575.00", "12.9"),
    ("570.26", "12.8"),
    ("565.57", "12.7"),
    ("560.28", "12.6"),
    ("555.42", "12.5"),
    ("550.54", "12.4"),
    ("545.23", "12.3"),
    ("535.54", "12.2"),
    ("530.30", "12.1"),
    ("525.60", "12.0"),
    ("520.61", "11.8"),
    ("515.43", "11.7"),
    ("510.39", "11.6"),
    ("495.29", "11.5"),
    ("470.42", "11.4"),
    ("465.19", "11.3"),
    ("460.27", "11.2"),
    ("455.23", "11.1"),
    ("450.36", "11.0"),
]


def _check_cuda(env: Environment, result: CheckResult) -> None:
    """Check CUDA installation consistency and driver compatibility."""
    cuda = env.cuda_info

    # No GPU at all
    if not env.gpus and not cuda.best_version:
        result.issues.append(
            Issue(Severity.WARNING, "gpu",
                  "No NVIDIA GPU detected.",
                  "GPU-accelerated PyTorch requires an NVIDIA GPU. CPU-only mode will work but be slower.")
        )
        return

    # nvidia-smi works but no nvcc
    if cuda.smi_version and not cuda.nvcc_version:
        result.issues.append(
            Issue(Severity.WARNING, "cuda",
                  "NVIDIA driver found but CUDA toolkit (nvcc) not installed.",
                  "Install CUDA toolkit or use PyTorch pre-built wheels with bundled CUDA.")
        )

    # Driver vs nvcc mismatch
    if cuda.smi_version and cuda.nvcc_version:
        smi = version_tuple(cuda.smi_version)
        nvcc = version_tuple(cuda.nvcc_version)
        if nvcc > smi:
            result.issues.append(
                Issue(Severity.ERROR, "cuda",
                      f"CUDA toolkit ({cuda.nvcc_version}) is newer than driver supports ({cuda.smi_version}).",
                      f"Update NVIDIA driver to support CUDA {cuda.nvcc_version}, or install CUDA toolkit <={cuda.smi_version}.")
            )

    # torch.cuda vs nvidia-smi mismatch
    if cuda.torch_cuda_version and cuda.smi_version:
        torch_cuda = version_tuple(cuda.torch_cuda_version)
        smi = version_tuple(cuda.smi_version)
        if torch_cuda[:2] > smi[:2]:
            result.issues.append(
                Issue(Severity.ERROR, "cuda",
                      f"PyTorch was built for CUDA {cuda.torch_cuda_version} but driver only supports up to {cuda.smi_version}.",
                      f"Either update your NVIDIA driver or install a PyTorch build for CUDA {cuda.smi_version}.")
            )

    # Driver version → max CUDA check
    if cuda.driver_version:
        max_cuda = _max_cuda_for_driver(cuda.driver_version)
        if max_cuda:
            target_cuda = cuda.nvcc_version or cuda.torch_cuda_version
            if target_cuda and version_tuple(target_cuda) > version_tuple(max_cuda):
                result.issues.append(
                    Issue(Severity.ERROR, "driver",
                          f"Driver {cuda.driver_version} supports up to CUDA {max_cuda}, but CUDA {target_cuda} is in use.",
                          f"Update driver to support CUDA {target_cuda}.")
                )


def _max_cuda_for_driver(driver_ver: str) -> Optional[str]:
    """Look up max CUDA version for a given driver version."""
    dv = version_tuple(driver_ver)
    for min_driver, max_cuda in DRIVER_CUDA_MAP:
        if dv >= version_tuple(min_driver):
            return max_cuda
    return None


# ---------------------------------------------------------------------------
# PyTorch ecosystem
# ---------------------------------------------------------------------------

def _strip_local(version: str) -> str:
    """Strip local version specifier: '2.9.1+cu128' → '2.9.1'."""
    return version.split("+")[0]

def _check_pytorch(env: Environment, result: CheckResult) -> None:
    db = load_db("pytorch")
    torch_ver_raw = env.installed_packages.get("torch")

    if not torch_ver_raw:
        result.issues.append(
            Issue(Severity.WARNING, "torch",
                  "torch is not installed.",
                  "Install with: pip install torch")
        )
        compatible = find_compatible_torch(
            db,
            python_version=env.python_version,
            cuda_version=env.cuda_version,
        )
        if compatible:
            result.compatible_versions = find_companions(db, compatible[0]["version"])
        return

    # Strip local version specifier: "2.9.1+cu128" → "2.9.1"
    torch_ver = _strip_local(torch_ver_raw)

    info = get_version_info(db, torch_ver)
    if not info:
        result.issues.append(
            Issue(Severity.WARNING, "torch",
                  f"torch=={torch_ver_raw} is not in our database.",
                  "You may be using a nightly or very new release.")
        )
        return

    py_minor = normalize_python(env.python_version)
    if py_minor not in info["python"]:
        result.issues.append(
            Issue(Severity.ERROR, "python",
                  f"Python {py_minor} is NOT compatible with torch=={torch_ver}.",
                  f"Compatible: {', '.join(info['python'])}")
        )

    # CUDA compat (use best detected version)
    cuda_ver = env.cuda_version
    if cuda_ver:
        cuda_parts = cuda_ver.split(".")
        cuda_short = f"{cuda_parts[0]}.{cuda_parts[1]}" if len(cuda_parts) >= 2 else cuda_ver
        if cuda_short not in info.get("cuda", []):
            result.issues.append(
                Issue(Severity.ERROR, "cuda",
                      f"CUDA {cuda_short} is NOT compatible with torch=={torch_ver}.",
                      f"Compatible CUDA: {', '.join(info['cuda'])}")
            )

    # Companion packages
    for pkg, expected in info.get("companions", {}).items():
        installed = env.installed_packages.get(pkg)
        if installed and _strip_local(installed) != expected:
            result.issues.append(
                Issue(Severity.ERROR, pkg,
                      f"{pkg}=={installed} is NOT compatible with torch=={torch_ver}.",
                      f"Expected {pkg}=={expected}")
            )

    result.compatible_versions = find_companions(db, torch_ver)


# ---------------------------------------------------------------------------
# Transformers ecosystem
# ---------------------------------------------------------------------------

def _check_transformers(env: Environment, result: CheckResult) -> None:
    tf_ver = env.installed_packages.get("transformers")
    if not tf_ver:
        return

    try:
        db = load_db("transformers")
    except FileNotFoundError:
        return

    info = get_transformers_info(db, tf_ver)
    if not info:
        result.issues.append(
            Issue(Severity.WARNING, "transformers",
                  f"transformers=={tf_ver} is not in our database.")
        )
        return

    # Python
    py_minor = normalize_python(env.python_version)
    if py_minor not in info["python"]:
        result.issues.append(
            Issue(Severity.ERROR, "transformers",
                  f"Python {py_minor} is NOT compatible with transformers=={tf_ver}.",
                  f"Compatible: {', '.join(info['python'])}")
        )

    # Torch minimum
    torch_ver = env.installed_packages.get("torch")
    torch_min = info.get("torch_min")
    if torch_ver and torch_min and version_tuple(_strip_local(torch_ver)) < version_tuple(torch_min):
        result.issues.append(
            Issue(Severity.ERROR, "transformers",
                  f"torch=={torch_ver} is below minimum for transformers=={tf_ver}.",
                  f"Requires torch>={torch_min}. Recommended: >={info.get('torch_recommended', torch_min)}")
        )

    # Companion constraints
    for pkg, constraint in info.get("companions", {}).items():
        installed = env.installed_packages.get(pkg)
        if installed and not satisfies_constraint(installed, constraint):
            result.issues.append(
                Issue(Severity.ERROR, pkg,
                      f"{pkg}=={installed} doesn't satisfy {constraint} (needed by transformers=={tf_ver}).",
                      f"Fix: pip install '{pkg}{constraint}'")
            )

    # Notes
    notes = info.get("notes", "")
    if notes:
        result.issues.append(Issue(Severity.WARNING, "transformers", f"Note: {notes}"))


# ---------------------------------------------------------------------------
# Known conflicts
# ---------------------------------------------------------------------------

def _check_known_conflicts(env: Environment, result: CheckResult) -> None:
    for db_name in ("transformers",):
        try:
            db = load_db(db_name)
        except FileNotFoundError:
            continue

        py_minor = normalize_python(env.python_version)

        for conflict in get_known_conflicts(db):
            pkgs = conflict.get("packages", {})
            all_match = True
            for pkg_name, constraint in pkgs.items():
                if pkg_name == "python":
                    ver = py_minor + ".0"
                else:
                    ver = env.installed_packages.get(pkg_name)
                if ver is None or not satisfies_constraint(ver, constraint):
                    all_match = False
                    break

            if all_match:
                result.issues.append(
                    Issue(Severity.ERROR, "conflict",
                          conflict["description"],
                          conflict.get("fix", ""))
                )


# ---------------------------------------------------------------------------
# Suggest
# ---------------------------------------------------------------------------

def suggest(
    package: str = "torch",
    version: Optional[str] = None,
    python_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
) -> Optional[dict]:
    """Suggest a compatible set of packages."""
    if package == "torch":
        db = load_db("pytorch")
        if version:
            return find_companions(db, version)
        compatible = find_compatible_torch(db, python_version=python_version, cuda_version=cuda_version)
        return find_companions(db, compatible[0]["version"]) if compatible else None

    elif package == "transformers":
        db = load_db("transformers")
        torch_ver = None
        if cuda_version or not version:
            pytorch_db = load_db("pytorch")
            torch_matches = find_compatible_torch(pytorch_db, python_version=python_version, cuda_version=cuda_version)
            if torch_matches:
                torch_ver = torch_matches[0]["version"]

        if version:
            info = get_transformers_info(db, version)
            if info:
                rec = {"transformers": version, **info.get("companions", {})}
                rec["torch_min"] = info.get("torch_min", "")
                rec["torch_recommended"] = info.get("torch_recommended", "")
                rec["python"] = info["python"]
                if torch_ver:
                    rec["torch"] = torch_ver
                return rec
            return None

        compatible = find_compatible_transformers(db, python_version=python_version, torch_version=torch_ver)
        if compatible:
            best = compatible[0]
            rec = {"transformers": best["version"], **best.get("companions", {})}
            rec["torch_min"] = best.get("torch_min", "")
            rec["torch_recommended"] = best.get("torch_recommended", "")
            rec["python"] = best["python"]
            if torch_ver:
                rec["torch"] = torch_ver
            return rec
        return None

    return None
