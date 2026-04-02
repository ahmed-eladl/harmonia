"""Deep system and environment detection."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Every ML package we track across all ecosystems
TARGET_PACKAGES = {
    # PyTorch core
    "torch", "torchvision", "torchaudio", "torchtext", "torchdata",
    # HuggingFace
    "transformers", "accelerate", "tokenizers", "safetensors",
    "huggingface-hub", "datasets", "sentence-transformers", "peft", "trl",
    # Training / inference
    "pytorch-lightning", "lightning", "xformers", "triton",
    "deepspeed", "bitsandbytes", "vllm", "optimum",
    # Numerical
    "numpy", "scipy",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    name: str = ""
    memory_total_mb: int = 0
    driver_version: str = ""
    cuda_version: str = ""  # CUDA version reported by nvidia-smi

    def summary_line(self) -> str:
        mem = f"{self.memory_total_mb} MB" if self.memory_total_mb else "unknown"
        return f"{self.name} ({mem})"


@dataclass
class OSInfo:
    system: str = ""        # Linux, Darwin, Windows
    release: str = ""       # kernel version
    arch: str = ""          # x86_64, aarch64
    distro: str = ""        # Ubuntu, Debian, CentOS, Fedora, Arch, macOS, Windows
    distro_version: str = ""  # 22.04, 11, etc.
    distro_codename: str = ""  # jammy, bookworm, etc.
    glibc_version: str = ""

    @property
    def display_name(self) -> str:
        if self.distro:
            parts = [self.distro]
            if self.distro_version:
                parts.append(self.distro_version)
            if self.distro_codename:
                parts.append(f"({self.distro_codename})")
            return " ".join(parts)
        return f"{self.system} {self.release}"


@dataclass
class CUDAInfo:
    # nvidia-smi reported
    smi_version: str = ""
    driver_version: str = ""
    # nvcc reported
    nvcc_version: str = ""
    # torch reported
    torch_cuda_version: str = ""
    # cuDNN
    cudnn_version: str = ""

    @property
    def best_version(self) -> Optional[str]:
        """Return the most reliable CUDA version we found."""
        # nvcc is most accurate for compilation, smi for driver support
        return self.nvcc_version or self.smi_version or self.torch_cuda_version or None

    @property
    def driver_cuda_max(self) -> Optional[str]:
        """Max CUDA version supported by the installed driver (from nvidia-smi)."""
        return self.smi_version or None

    def summary_lines(self) -> list[str]:
        lines = []
        if self.driver_version:
            lines.append(f"  Driver:     {self.driver_version}")
        if self.smi_version:
            lines.append(f"  CUDA (smi): {self.smi_version}")
        if self.nvcc_version:
            lines.append(f"  CUDA (nvcc):{self.nvcc_version}")
        if self.torch_cuda_version:
            lines.append(f"  CUDA (torch):{self.torch_cuda_version}")
        if self.cudnn_version:
            lines.append(f"  cuDNN:      {self.cudnn_version}")
        if not lines:
            lines.append("  Not detected")
        return lines


@dataclass
class Environment:
    python_version: str = ""
    python_path: str = ""
    venv: str = ""  # virtualenv / conda env name
    os_info: OSInfo = field(default_factory=OSInfo)
    cuda_info: CUDAInfo = field(default_factory=CUDAInfo)
    gpus: list[GPUInfo] = field(default_factory=list)
    installed_packages: dict[str, str] = field(default_factory=dict)

    @property
    def cuda_version(self) -> Optional[str]:
        """Shorthand for best detected CUDA version."""
        return self.cuda_info.best_version

    def summary(self) -> str:
        lines = []

        # System
        lines.append("System:")
        lines.append(f"  OS:       {self.os_info.display_name}")
        lines.append(f"  Arch:     {self.os_info.arch}")
        if self.os_info.glibc_version:
            lines.append(f"  glibc:    {self.os_info.glibc_version}")

        # Python
        lines.append("")
        lines.append("Python:")
        lines.append(f"  Version:  {self.python_version}")
        lines.append(f"  Path:     {self.python_path}")
        if self.venv:
            lines.append(f"  Env:      {self.venv}")

        # GPU / CUDA
        lines.append("")
        if self.gpus:
            lines.append(f"GPU ({len(self.gpus)} detected):")
            for i, gpu in enumerate(self.gpus):
                lines.append(f"  [{i}] {gpu.summary_line()}")
        else:
            lines.append("GPU: none detected")

        lines.append("")
        lines.append("CUDA:")
        lines.extend(self.cuda_info.summary_lines())

        # Packages
        lines.append("")
        pytorch_pkgs = {k: v for k, v in self.installed_packages.items()
                        if k in ("torch", "torchaudio", "torchvision")}
        hf_pkgs = {k: v for k, v in self.installed_packages.items()
                   if k in ("transformers", "accelerate", "tokenizers", "safetensors", "huggingface-hub")}
        other_pkgs = {k: v for k, v in self.installed_packages.items()
                      if k not in pytorch_pkgs and k not in hf_pkgs}

        if pytorch_pkgs or hf_pkgs or other_pkgs:
            lines.append("Packages:")
            if pytorch_pkgs:
                for pkg, ver in sorted(pytorch_pkgs.items()):
                    lines.append(f"  {pkg}=={ver}")
            if hf_pkgs:
                for pkg, ver in sorted(hf_pkgs.items()):
                    lines.append(f"  {pkg}=={ver}")
            if other_pkgs:
                for pkg, ver in sorted(other_pkgs.items()):
                    lines.append(f"  {pkg}=={ver}")
        else:
            lines.append("Packages: none detected")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection entry point
# ---------------------------------------------------------------------------

def detect() -> Environment:
    """Run full environment detection."""
    env = Environment()
    env.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    env.python_path = sys.executable
    env.venv = _detect_venv()
    env.os_info = _detect_os()
    env.cuda_info = _detect_cuda()
    env.gpus = _detect_gpus()
    env.installed_packages = _detect_packages()
    return env


# ---------------------------------------------------------------------------
# OS / Distro detection
# ---------------------------------------------------------------------------

def _detect_os() -> OSInfo:
    info = OSInfo()
    info.system = platform.system()
    info.release = platform.release()
    info.arch = platform.machine()

    if info.system == "Linux":
        _detect_linux_distro(info)
        _detect_glibc(info)
    elif info.system == "Darwin":
        info.distro = "macOS"
        info.distro_version = platform.mac_ver()[0]
    elif info.system == "Windows":
        info.distro = "Windows"
        info.distro_version = platform.version()

    return info


def _detect_linux_distro(info: OSInfo) -> None:
    """Parse /etc/os-release for distro info (works on all modern Linux)."""
    os_release = Path("/etc/os-release")
    if os_release.exists():
        data = {}
        try:
            for line in os_release.read_text().strip().split("\n"):
                if "=" in line:
                    key, _, val = line.partition("=")
                    data[key.strip()] = val.strip().strip('"')
        except Exception:
            pass

        info.distro = data.get("NAME", data.get("ID", "Linux"))
        info.distro_version = data.get("VERSION_ID", "")
        info.distro_codename = data.get("VERSION_CODENAME", "")
        return

    # Fallback: try lsb_release
    try:
        out = subprocess.run(
            ["lsb_release", "-a"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            for line in out.stdout.split("\n"):
                if line.startswith("Description:"):
                    info.distro = line.split(":", 1)[1].strip()
                elif line.startswith("Release:"):
                    info.distro_version = line.split(":", 1)[1].strip()
                elif line.startswith("Codename:"):
                    info.distro_codename = line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info.distro = "Linux (unknown)"


def _detect_glibc(info: OSInfo) -> None:
    """Detect glibc version (important for wheel compatibility)."""
    try:
        # platform.libc_ver() is unreliable, use ctypes
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        gnu_get_libc_version = libc.gnu_get_libc_version
        gnu_get_libc_version.restype = ctypes.c_char_p
        info.glibc_version = gnu_get_libc_version().decode()
    except Exception:
        ver = platform.libc_ver()
        if ver and ver[1]:
            info.glibc_version = ver[1]


# ---------------------------------------------------------------------------
# CUDA detection (multi-source)
# ---------------------------------------------------------------------------

def _detect_cuda() -> CUDAInfo:
    info = CUDAInfo()

    # 1. nvidia-smi (driver-level CUDA support)
    _cuda_from_nvidia_smi(info)

    # 2. nvcc (toolkit installed)
    _cuda_from_nvcc(info)

    # 3. torch.version.cuda
    _cuda_from_torch(info)

    # 4. cuDNN
    _detect_cudnn(info)

    return info


def _cuda_from_nvidia_smi(info: CUDAInfo) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0:
            return

        for line in out.stdout.split("\n"):
            if "CUDA Version" in line:
                match = re.search(r"CUDA Version:\s*(\d+\.\d+)", line)
                if match:
                    info.smi_version = match.group(1)
            if "Driver Version" in line:
                match = re.search(r"Driver Version:\s*(\d+\.\d+(?:\.\d+)?)", line)
                if match:
                    info.driver_version = match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _cuda_from_nvcc(info: CUDAInfo) -> None:
    try:
        out = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            match = re.search(r"release\s+(\d+\.\d+)", out.stdout)
            if match:
                info.nvcc_version = match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _cuda_from_torch(info: CUDAInfo) -> None:
    try:
        import torch
        if hasattr(torch.version, "cuda") and torch.version.cuda:
            info.torch_cuda_version = torch.version.cuda
    except ImportError:
        pass


def _detect_cudnn(info: CUDAInfo) -> None:
    try:
        import torch
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            ver = torch.backends.cudnn.version()
            if ver:
                major = ver // 1000
                minor = (ver % 1000) // 100
                patch = ver % 100
                info.cudnn_version = f"{major}.{minor}.{patch}"
    except (ImportError, Exception):
        pass


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpus() -> list[GPUInfo]:
    """Detect GPUs via nvidia-smi --query-gpu."""
    gpus = []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append(GPUInfo(
                        name=parts[0],
                        memory_total_mb=int(float(parts[1])) if parts[1] else 0,
                        driver_version=parts[2],
                    ))
                elif len(parts) >= 1 and parts[0]:
                    gpus.append(GPUInfo(name=parts[0]))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return gpus


# ---------------------------------------------------------------------------
# Virtual environment detection
# ---------------------------------------------------------------------------

def _detect_venv() -> str:
    """Detect if running inside a virtualenv or conda env."""
    # Conda
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env:
        return f"conda:{conda_env}"

    # virtualenv / venv
    venv = os.environ.get("VIRTUAL_ENV", "")
    if venv:
        return f"venv:{Path(venv).name}"

    # Poetry
    poetry = os.environ.get("POETRY_ACTIVE", "")
    if poetry:
        return "poetry"

    return ""


# ---------------------------------------------------------------------------
# Package detection
# ---------------------------------------------------------------------------

def _detect_packages() -> dict[str, str]:
    """Detect installed ML-related packages."""
    installed = {}
    try:
        from importlib.metadata import distributions
        for dist in distributions():
            raw_name = dist.metadata.get("Name", "")
            if not raw_name:
                continue
            name = raw_name.lower().replace("_", "-")
            if name in TARGET_PACKAGES:
                installed[name] = dist.version
    except Exception:
        # Fallback
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode == 0:
                import json
                for pkg in json.loads(out.stdout):
                    name = pkg["name"].lower().replace("_", "-")
                    if name in TARGET_PACKAGES:
                        installed[name] = pkg["version"]
        except Exception:
            pass
    return installed
