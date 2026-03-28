"""Tests for harmonia."""

import pytest

from harmonia.checker import Severity, check, suggest
from harmonia.database import (
    find_companions,
    find_compatible_torch,
    find_compatible_transformers,
    get_known_conflicts,
    get_torch_versions,
    get_transformers_info,
    get_transformers_versions,
    load_db,
    normalize_python,
    satisfies_constraint,
    version_tuple,
)
from harmonia.detector import CUDAInfo, Environment, GPUInfo, OSInfo


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def pytorch_db():
    return load_db("pytorch")


@pytest.fixture
def transformers_db():
    return load_db("transformers")


def _env(**overrides) -> Environment:
    """Helper to build an Environment with sensible defaults."""
    defaults = dict(
        python_version="3.10.0",
        python_path="/usr/bin/python3",
        venv="venv:test",
        os_info=OSInfo(system="Linux", release="5.15.0", arch="x86_64",
                       distro="Ubuntu", distro_version="22.04",
                       distro_codename="jammy", glibc_version="2.35"),
        cuda_info=CUDAInfo(),
        gpus=[],
        installed_packages={},
    )
    defaults.update(overrides)
    return Environment(**defaults)


# ===================================================================
# Version utilities
# ===================================================================

class TestVersionUtils:
    def test_version_tuple(self):
        assert version_tuple("2.5.1") == (2, 5, 1)
        assert version_tuple("12.1") == (12, 1)
        assert version_tuple("1.0.0a1") == (1, 0, 0)

    def test_normalize_python(self):
        assert normalize_python("3.10.12") == "3.10"
        assert normalize_python("3.11") == "3.11"

    def test_satisfies_gte(self):
        assert satisfies_constraint("1.5.0", ">=1.0.0")
        assert satisfies_constraint("1.0.0", ">=1.0.0")
        assert not satisfies_constraint("0.9.0", ">=1.0.0")

    def test_satisfies_lt(self):
        assert satisfies_constraint("0.9.0", "<1.0.0")
        assert not satisfies_constraint("1.0.0", "<1.0.0")

    def test_satisfies_range(self):
        assert satisfies_constraint("0.20.5", ">=0.20.0,<0.22")
        assert not satisfies_constraint("0.22.0", ">=0.20.0,<0.22")
        assert not satisfies_constraint("0.19.0", ">=0.20.0,<0.22")

    def test_satisfies_eq(self):
        assert satisfies_constraint("4.46.0", "==4.46.0")
        assert not satisfies_constraint("4.46.1", "==4.46.0")


# ===================================================================
# PyTorch database
# ===================================================================

class TestPyTorchDB:
    def test_load(self, pytorch_db):
        assert "packages" in pytorch_db
        assert "torch" in pytorch_db["packages"]

    def test_versions_sorted(self, pytorch_db):
        versions = get_torch_versions(pytorch_db)
        assert len(versions) > 0
        assert version_tuple(versions[0]) >= version_tuple(versions[-1])

    def test_find_python_310(self, pytorch_db):
        results = find_compatible_torch(pytorch_db, python_version="3.10")
        assert all("3.10" in r["python"] for r in results)

    def test_find_cuda_118(self, pytorch_db):
        results = find_compatible_torch(pytorch_db, cuda_version="11.8")
        assert all("11.8" in r["cuda"] for r in results)

    def test_find_both_constraints(self, pytorch_db):
        results = find_compatible_torch(pytorch_db, python_version="3.11", cuda_version="12.1")
        assert len(results) > 0

    def test_companions(self, pytorch_db):
        c = find_companions(pytorch_db, "2.5.1")
        assert c is not None
        assert c["torch"] == "2.5.1"
        assert "torchaudio" in c
        assert "torchvision" in c

    def test_companions_unknown(self, pytorch_db):
        assert find_companions(pytorch_db, "99.99.99") is None


# ===================================================================
# Transformers database
# ===================================================================

class TestTransformersDB:
    def test_load(self, transformers_db):
        assert "transformers" in transformers_db["packages"]

    def test_exact_version(self, transformers_db):
        info = get_transformers_info(transformers_db, "4.44.2")
        assert info is not None
        assert "torch_min" in info

    def test_range_v5(self, transformers_db):
        info = get_transformers_info(transformers_db, "5.2.0")
        assert info is not None
        assert "3.10" in info["python"]

    def test_unknown(self, transformers_db):
        assert get_transformers_info(transformers_db, "1.0.0") is None

    def test_find_by_python(self, transformers_db):
        results = find_compatible_transformers(transformers_db, python_version="3.10")
        assert len(results) > 0

    def test_find_by_torch(self, transformers_db):
        results = find_compatible_transformers(transformers_db, torch_version="2.5.0")
        assert len(results) > 0

    def test_known_conflicts(self, transformers_db):
        conflicts = get_known_conflicts(transformers_db)
        assert len(conflicts) > 0
        assert all("description" in c for c in conflicts)


# ===================================================================
# System-level checks
# ===================================================================

class TestSystemChecks:
    def test_old_glibc(self):
        env = _env(
            os_info=OSInfo(system="Linux", glibc_version="2.12", distro="CentOS"),
            installed_packages={"torch": "2.5.1"},
        )
        result = check(env)
        assert any("glibc" in i.message for i in result.issues)

    def test_modern_glibc_no_warning(self):
        env = _env(
            os_info=OSInfo(system="Linux", glibc_version="2.35", distro="Ubuntu"),
            installed_packages={"torch": "2.5.1"},
        )
        result = check(env)
        assert not any("glibc" in i.message for i in result.issues)

    def test_old_python(self):
        env = _env(python_version="3.7.0", installed_packages={})
        result = check(env)
        assert any("end-of-life" in i.message for i in result.issues)

    def test_no_venv_warning(self):
        env = _env(venv="", installed_packages={})
        result = check(env)
        assert any("virtual environment" in i.message for i in result.issues)

    def test_in_venv_no_warning(self):
        env = _env(venv="venv:myproject", installed_packages={})
        result = check(env)
        assert not any("virtual environment" in i.message for i in result.issues)


# ===================================================================
# CUDA checks
# ===================================================================

class TestCUDAChecks:
    def test_no_gpu(self):
        env = _env(gpus=[], cuda_info=CUDAInfo(), installed_packages={"torch": "2.5.1"})
        result = check(env)
        assert any(i.package == "gpu" for i in result.issues)

    def test_driver_no_nvcc(self):
        env = _env(
            cuda_info=CUDAInfo(smi_version="12.1", driver_version="535.54"),
            gpus=[GPUInfo(name="RTX 4090", memory_total_mb=24576)],
            installed_packages={"torch": "2.5.1"},
        )
        result = check(env)
        assert any("nvcc" in i.message.lower() for i in result.issues)

    def test_nvcc_exceeds_driver(self):
        env = _env(
            cuda_info=CUDAInfo(smi_version="11.8", nvcc_version="12.4", driver_version="520.61"),
            gpus=[GPUInfo(name="RTX 3080")],
            installed_packages={"torch": "2.5.1"},
        )
        result = check(env)
        assert any(
            i.severity == Severity.ERROR and "newer than driver" in i.message
            for i in result.issues
        )

    def test_torch_cuda_exceeds_driver(self):
        env = _env(
            cuda_info=CUDAInfo(smi_version="11.7", torch_cuda_version="12.1", driver_version="515.43"),
            gpus=[GPUInfo(name="RTX 3070")],
            installed_packages={"torch": "2.5.1"},
        )
        result = check(env)
        assert any(
            i.severity == Severity.ERROR and "built for CUDA" in i.message
            for i in result.issues
        )

    def test_consistent_cuda_ok(self):
        env = _env(
            cuda_info=CUDAInfo(smi_version="12.1", nvcc_version="12.1",
                              torch_cuda_version="12.1", driver_version="535.54"),
            gpus=[GPUInfo(name="RTX 4090", memory_total_mb=24576)],
            installed_packages={
                "torch": "2.5.1", "torchaudio": "2.5.1", "torchvision": "0.20.1",
            },
        )
        result = check(env)
        # Should not have any CUDA-related errors
        cuda_errors = [i for i in result.issues
                       if i.severity == Severity.ERROR and i.package in ("cuda", "driver")]
        assert len(cuda_errors) == 0


# ===================================================================
# PyTorch checker
# ===================================================================

class TestPyTorchChecker:
    def test_no_torch(self):
        env = _env(installed_packages={})
        result = check(env)
        assert any(i.package == "torch" for i in result.issues)

    def test_compatible(self):
        env = _env(
            cuda_info=CUDAInfo(smi_version="12.1", driver_version="535.54"),
            gpus=[GPUInfo(name="GPU")],
            installed_packages={
                "torch": "2.5.1", "torchaudio": "2.5.1", "torchvision": "0.20.1",
            },
        )
        result = check(env)
        errors = [i for i in result.issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_wrong_companion(self):
        env = _env(installed_packages={"torch": "2.5.1", "torchaudio": "2.3.0"})
        result = check(env)
        assert any(i.package == "torchaudio" and i.severity == Severity.ERROR for i in result.issues)

    def test_bad_python(self):
        env = _env(python_version="3.7.0", installed_packages={"torch": "2.5.1"})
        result = check(env)
        assert result.has_errors

    def test_bad_cuda(self):
        env = _env(
            cuda_info=CUDAInfo(smi_version="10.2"),
            gpus=[GPUInfo(name="GPU")],
            installed_packages={"torch": "2.5.1"},
        )
        result = check(env)
        assert any(i.package == "cuda" and "NOT compatible" in i.message for i in result.issues)


# ===================================================================
# Transformers checker
# ===================================================================

class TestTransformersChecker:
    def test_ok(self):
        env = _env(
            cuda_info=CUDAInfo(smi_version="12.1", driver_version="535.54"),
            gpus=[GPUInfo(name="GPU")],
            installed_packages={
                "torch": "2.5.1", "transformers": "4.44.2",
                "accelerate": "0.30.0", "tokenizers": "0.20.0",
            },
        )
        result = check(env)
        tf_errors = [i for i in result.issues
                     if i.package == "transformers" and i.severity == Severity.ERROR]
        assert len(tf_errors) == 0

    def test_torch_too_old(self):
        env = _env(installed_packages={"torch": "1.12.0", "transformers": "4.44.2"})
        result = check(env)
        assert any("below minimum" in i.message for i in result.issues)

    def test_v5_needs_new_torch(self):
        env = _env(
            python_version="3.11.0",
            installed_packages={"torch": "2.0.0", "transformers": "5.2.0"},
        )
        result = check(env)
        assert any("below minimum" in i.message for i in result.issues)

    def test_bad_python_v5(self):
        env = _env(
            python_version="3.8.0",
            installed_packages={"transformers": "5.2.0", "torch": "2.5.1"},
        )
        result = check(env)
        assert any(
            i.package == "transformers" and "Python" in i.message and i.severity == Severity.ERROR
            for i in result.issues
        )

    def test_bad_accelerate(self):
        env = _env(installed_packages={
            "torch": "2.5.1", "transformers": "4.46.3", "accelerate": "0.20.0",
        })
        result = check(env)
        assert any(i.package == "accelerate" and i.severity == Severity.ERROR for i in result.issues)


# ===================================================================
# Known conflicts
# ===================================================================

class TestConflicts:
    def test_pytree_conflict(self):
        env = _env(installed_packages={"transformers": "4.40.0", "torch": "1.12.0"})
        result = check(env)
        assert any(i.package == "conflict" for i in result.issues)

    def test_no_false_conflict(self):
        env = _env(installed_packages={"transformers": "4.44.2", "torch": "2.4.0"})
        result = check(env)
        assert not any(i.package == "conflict" for i in result.issues)


# ===================================================================
# Suggest
# ===================================================================

class TestSuggest:
    def test_torch_version(self):
        r = suggest(package="torch", version="2.4.0")
        assert r is not None and r["torch"] == "2.4.0"

    def test_torch_by_python(self):
        r = suggest(package="torch", python_version="3.8")
        assert r is not None

    def test_torch_impossible(self):
        assert suggest(package="torch", python_version="2.7", cuda_version="99.0") is None

    def test_transformers_version(self):
        r = suggest(package="transformers", version="4.44.2")
        assert r is not None and r["transformers"] == "4.44.2"

    def test_transformers_auto(self):
        r = suggest(package="transformers", python_version="3.11")
        assert r is not None

    def test_unknown_package(self):
        assert suggest(package="nonexistent") is None


# ===================================================================
# OSInfo display
# ===================================================================

class TestOSInfo:
    def test_display_name_ubuntu(self):
        info = OSInfo(distro="Ubuntu", distro_version="22.04", distro_codename="jammy")
        assert info.display_name == "Ubuntu 22.04 (jammy)"

    def test_display_name_macos(self):
        info = OSInfo(system="Darwin", distro="macOS", distro_version="14.2")
        assert info.display_name == "macOS 14.2"

    def test_display_name_fallback(self):
        info = OSInfo(system="Linux", release="5.15.0")
        assert info.display_name == "Linux 5.15.0"


# ===================================================================
# CUDAInfo
# ===================================================================

class TestCUDAInfo:
    def test_best_version_nvcc(self):
        info = CUDAInfo(smi_version="12.4", nvcc_version="12.1")
        assert info.best_version == "12.1"

    def test_best_version_smi_fallback(self):
        info = CUDAInfo(smi_version="12.4")
        assert info.best_version == "12.4"

    def test_best_version_torch_fallback(self):
        info = CUDAInfo(torch_cuda_version="11.8")
        assert info.best_version == "11.8"

    def test_best_version_none(self):
        info = CUDAInfo()
        assert info.best_version is None
