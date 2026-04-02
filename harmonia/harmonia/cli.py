"""CLI entry point for harmonia."""

from __future__ import annotations

import argparse
import json
import sys

from harmonia import __version__, __banner__


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows.

    Windows consoles default to cp1252 which can't print emoji.
    """
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # Python < 3.7 or non-reconfigurable stream
            import io
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )


def main() -> None:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        prog="harmonia",
        description="Bring harmony to your ML dependency stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  harmonia check                          # full environment scan\n"
               "  harmonia doctor                         # system-level diagnostics\n"
               "  harmonia suggest torch==2.5.1           # compatible packages for torch 2.5.1\n"
               "  harmonia suggest transformers --cuda 12.1\n"
               "  harmonia matrix pytorch                 # PyTorch compatibility table\n"
               "  harmonia matrix transformers            # Transformers compatibility table\n"
               "  harmonia conflicts                      # known conflict patterns\n",
    )
    parser.add_argument("--version", action="version", version=f"harmonia {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # --- check ---
    check_p = subparsers.add_parser("check", help="Check environment for compatibility issues")
    check_p.add_argument("--json", dest="as_json", action="store_true", help="Output as JSON")

    # --- doctor ---
    subparsers.add_parser("doctor", help="Deep system diagnostics (GPU, CUDA, driver, OS)")

    # --- suggest ---
    suggest_p = subparsers.add_parser("suggest", help="Suggest compatible package versions")
    suggest_p.add_argument(
        "target", nargs="?", default=None,
        help="Package or package==version (e.g. torch, torch==2.5.1, transformers==4.44.2)")
    suggest_p.add_argument("--python", dest="python_version", help="Python version (e.g. 3.11)")
    suggest_p.add_argument("--cuda", dest="cuda_version", help="CUDA version (e.g. 12.1)")

    # --- matrix ---
    matrix_p = subparsers.add_parser("matrix", help="Print compatibility matrix")
    matrix_p.add_argument(
        "ecosystem", nargs="?", default="pytorch",
        choices=["pytorch", "transformers"],
        help="Which ecosystem (default: pytorch)")
    matrix_p.add_argument("--json", dest="as_json", action="store_true")

    # --- conflicts ---
    subparsers.add_parser("conflicts", help="List known conflict patterns")

    # --- fix ---
    fix_p = subparsers.add_parser("fix", help="Generate or run pip commands to fix compatibility issues")
    fix_p.add_argument("--yes", "-y", action="store_true", help="Run fix commands automatically (no confirmation)")
    fix_p.add_argument("--dry-run", action="store_true", help="Show commands without running them")

    args = parser.parse_args()

    if args.command is None:
        print(__banner__)
        parser.print_help()
        sys.exit(0)

    commands = {
        "check": lambda: _cmd_check(args),
        "doctor": _cmd_doctor,
        "suggest": lambda: _cmd_suggest(args),
        "matrix": lambda: _cmd_matrix(args),
        "conflicts": _cmd_conflicts,
        "fix": lambda: _cmd_fix(args),
    }
    commands[args.command]()


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def _cmd_check(args: argparse.Namespace) -> None:
    from harmonia.checker import check

    result = check()

    if args.as_json:
        data = {
            "errors": result.error_count,
            "warnings": result.warning_count,
            "issues": [
                {"severity": i.severity.value, "package": i.package,
                 "message": i.message, "suggestion": i.suggestion}
                for i in result.issues
            ],
            "environment": {
                "python": result.environment.python_version,
                "os": result.environment.os_info.display_name,
                "cuda": result.environment.cuda_version,
                "packages": result.environment.installed_packages,
            },
        }
        print(json.dumps(data, indent=2))
    else:
        print(result.summary())

    sys.exit(1 if result.has_errors else 0)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def _cmd_doctor() -> None:
    from harmonia.detector import detect

    env = detect()

    print(__banner__)
    print("System Diagnostics")
    print("=" * 56)
    print()

    # OS
    os_info = env.os_info
    print(f"  OS:         {os_info.display_name}")
    print(f"  Kernel:     {os_info.system} {os_info.release}")
    print(f"  Arch:       {os_info.arch}")
    if os_info.glibc_version:
        print(f"  glibc:      {os_info.glibc_version}")
        from harmonia.database import version_tuple
        if version_tuple(os_info.glibc_version) < (2, 17):
            print("              ⚠️  Too old for manylinux2014 wheels")
        else:
            print("              ✅ OK for manylinux2014")
    print()

    # Python
    print(f"  Python:     {env.python_version}")
    print(f"  Executable: {env.python_path}")
    if env.venv:
        print(f"  Env:        {env.venv} ✅")
    else:
        print("  Env:        system (⚠️  consider using a virtualenv)")
    print()

    # GPU
    if env.gpus:
        print(f"  GPUs:       {len(env.gpus)} detected")
        for i, gpu in enumerate(env.gpus):
            print(f"    [{i}] {gpu.name}")
            print(f"        Memory: {gpu.memory_total_mb} MB")
            if gpu.driver_version:
                print(f"        Driver: {gpu.driver_version}")
    else:
        print("  GPUs:       None detected")
    print()

    # CUDA
    cuda = env.cuda_info
    print("  CUDA:")
    if cuda.driver_version:
        print(f"    Driver version: {cuda.driver_version}")
    if cuda.smi_version:
        print(f"    CUDA (nvidia-smi): {cuda.smi_version}  (max supported by driver)")
    else:
        print("    nvidia-smi: not available")
    if cuda.nvcc_version:
        print(f"    CUDA (nvcc): {cuda.nvcc_version}  (toolkit installed)")
    else:
        print("    nvcc: not installed")
    if cuda.torch_cuda_version:
        print(f"    CUDA (torch): {cuda.torch_cuda_version}  (PyTorch build)")
    if cuda.cudnn_version:
        print(f"    cuDNN: {cuda.cudnn_version}")

    # CUDA consistency
    if cuda.smi_version and cuda.nvcc_version:
        from harmonia.database import version_tuple
        if version_tuple(cuda.nvcc_version) <= version_tuple(cuda.smi_version):
            print(f"    ✅ nvcc ({cuda.nvcc_version}) <= driver max ({cuda.smi_version})")
        else:
            print(f"    ❌ nvcc ({cuda.nvcc_version}) > driver max ({cuda.smi_version}) — MISMATCH")
    if cuda.torch_cuda_version and cuda.smi_version:
        from harmonia.database import version_tuple
        tc = version_tuple(cuda.torch_cuda_version)[:2]
        sc = version_tuple(cuda.smi_version)[:2]
        if tc <= sc:
            print(f"    ✅ torch CUDA ({cuda.torch_cuda_version}) <= driver max ({cuda.smi_version})")
        else:
            print(f"    ❌ torch CUDA ({cuda.torch_cuda_version}) > driver max ({cuda.smi_version}) — MISMATCH")

    print()

    # Packages
    pkgs = env.installed_packages
    if pkgs:
        print("  Installed ML packages:")
        for pkg, ver in sorted(pkgs.items()):
            print(f"    {pkg}=={ver}")
    else:
        print("  No ML packages detected.")

    print()


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------

def _cmd_suggest(args: argparse.Namespace) -> None:
    from harmonia.checker import suggest
    from harmonia.detector import detect

    package = "torch"
    version = None

    if args.target:
        if "==" in args.target:
            package, version = args.target.split("==", 1)
        else:
            package = args.target

    python_ver = args.python_version
    cuda_ver = args.cuda_version

    if not version and not python_ver and not cuda_ver:
        env = detect()
        python_ver = python_ver or env.python_version
        cuda_ver = cuda_ver or env.cuda_version

    result = suggest(package=package, version=version,
                     python_version=python_ver, cuda_version=cuda_ver)

    if result is None:
        print(f"No compatible {package} version found for the given constraints.")
        sys.exit(1)

    skip_keys = {"python", "cuda", "install_hint", "notes", "torch_min", "torch_recommended"}
    print(f"📦 Recommended stack for {package}:")
    for pkg, ver in sorted(result.items()):
        if pkg in skip_keys:
            continue
        if isinstance(ver, list):
            print(f"  {pkg}: {', '.join(ver)}")
        else:
            print(f"  {pkg}: {ver}")

    torch_rec = result.get("torch_recommended")
    if torch_rec and "torch" not in result:
        print(f"  torch (recommended): >={torch_rec}")

    py = result.get("python")
    if py:
        print(f"  python: {', '.join(py) if isinstance(py, list) else py}")

    hint = result.get("install_hint")
    if hint:
        print(f"\nInstall:\n  {hint}")


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------

def _cmd_matrix(args: argparse.Namespace) -> None:
    if args.ecosystem == "pytorch":
        _matrix_pytorch(args)
    else:
        _matrix_transformers(args)


def _matrix_pytorch(args: argparse.Namespace) -> None:
    from harmonia.database import get_torch_versions, load_db

    db = load_db("pytorch")

    if args.as_json:
        print(json.dumps(db["packages"]["torch"], indent=2))
        return

    versions = get_torch_versions(db)
    header = f"{'torch':<10} {'Python':<28} {'CUDA':<18} {'torchvision':<15} {'torchaudio':<15}"
    print(header)
    print("─" * len(header))
    for v in versions:
        info = db["packages"]["torch"][v]
        py = ", ".join(info["python"])
        cuda = ", ".join(info.get("cuda", []))
        tv = info["companions"].get("torchvision", "-")
        ta = info["companions"].get("torchaudio", "-")
        print(f"{v:<10} {py:<28} {cuda:<18} {tv:<15} {ta:<15}")


def _matrix_transformers(args: argparse.Namespace) -> None:
    from harmonia.database import get_transformers_versions, load_db

    db = load_db("transformers")

    if args.as_json:
        print(json.dumps(db["packages"]["transformers"], indent=2))
        return

    versions = get_transformers_versions(db)
    header = f"{'transformers':<16} {'Python':<26} {'torch min':<12} {'torch rec':<12} {'accelerate':<18}"
    print(header)
    print("─" * len(header))
    for v in versions:
        info = db["packages"]["transformers"][v]
        py = ", ".join(info["python"])
        tmin = info.get("torch_min", "-")
        trec = info.get("torch_recommended", "-")
        accel = info.get("companions", {}).get("accelerate", "-")
        print(f"{v:<16} {py:<26} {tmin:<12} {trec:<12} {accel:<18}")


# ---------------------------------------------------------------------------
# conflicts
# ---------------------------------------------------------------------------

def _cmd_conflicts() -> None:
    from harmonia.database import get_known_conflicts, load_db

    print("Known compatibility conflicts:\n")
    idx = 1
    for db_name in ("transformers",):
        try:
            db = load_db(db_name)
        except FileNotFoundError:
            continue
        for c in get_known_conflicts(db):
            print(f"  {idx}. {c['description']}")
            pkgs = c.get("packages", {})
            parts = [f"{k}{v}" for k, v in pkgs.items()]
            print(f"     When: {' + '.join(parts)}")
            if c.get("error"):
                print(f"     Error: {c['error']}")
            if c.get("fix"):
                print(f"     Fix: {c['fix']}")
            print()
            idx += 1


# ---------------------------------------------------------------------------
# fix
# ---------------------------------------------------------------------------

def _cmd_fix(args: argparse.Namespace) -> None:
    import subprocess

    from harmonia.checker import Severity, check

    result = check()

    if result.is_clean:
        print("✅ No compatibility issues found — nothing to fix!")
        sys.exit(0)

    # Collect pip-fixable commands
    fix_commands = []
    manual_fixes = []

    for issue in result.issues:
        if issue.severity != Severity.ERROR:
            continue

        suggestion = issue.suggestion

        # Companion version mismatch: "Expected torchaudio==2.9.1"
        if issue.package in ("torchaudio", "torchvision", "torchtext") and "Expected" in suggestion:
            expected = suggestion.split("Expected ")[-1].strip()
            fix_commands.append(f"pip install {expected}")

        # accelerate/tokenizers/safetensors constraint: "pip install 'accelerate>=1.0.0'"
        elif issue.package in ("accelerate", "tokenizers", "safetensors", "huggingface-hub") and "pip install" in suggestion:
            cmd = suggestion.split("pip install ")[-1].strip().strip("'\"")
            fix_commands.append(f"pip install '{cmd}'")

        # torch too old for transformers
        elif issue.package == "transformers" and "below minimum" in issue.message:
            if "Recommended:" in suggestion:
                rec = suggestion.split("Recommended: ")[-1].strip()
                fix_commands.append(f"pip install 'torch{rec}'")
            elif "Requires" in suggestion:
                req = suggestion.split("Requires ")[-1].split(".")[0].strip()
                fix_commands.append(f"pip install 'torch{req}'")

        # Python version / CUDA / driver / system issues → can't pip-fix
        elif issue.package in ("python", "cuda", "driver", "system", "gpu"):
            manual_fixes.append(f"[{issue.package}] {issue.message}\n    ↳ {suggestion}")

        # Known conflicts
        elif issue.package == "conflict":
            if issue.suggestion:
                manual_fixes.append(f"[conflict] {issue.message}\n    ↳ {issue.suggestion}")

    # Use recommended install hint if available and we have companion issues
    install_hint = None
    if result.compatible_versions:
        install_hint = result.compatible_versions.get("install_hint", "")

    if not fix_commands and not manual_fixes:
        print("⚠️  Issues found but no automatic fixes available.")
        print("Run `harmonia check` for details.")
        sys.exit(1)

    # Show what we'll do
    if fix_commands:
        print("🔧 Fix commands:\n")
        for i, cmd in enumerate(fix_commands, 1):
            print(f"  {i}. {cmd}")
        print()

        if install_hint:
            print("  Or install the full recommended set:")
            print(f"  {install_hint}")
            print()

    if manual_fixes:
        print("⚠️  Manual fixes needed (cannot be auto-fixed):\n")
        for fix in manual_fixes:
            print(f"  {fix}")
        print()

    if not fix_commands:
        sys.exit(1)

    # Dry run — just show commands
    if args.dry_run:
        sys.exit(0)

    # Auto-run or ask
    if args.yes:
        run = True
    else:
        try:
            answer = input("Run fix commands? [y/N] ").strip().lower()
            run = answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)

    if not run:
        print("Aborted. You can copy and run the commands manually.")
        sys.exit(0)

    # Execute fixes
    failed = False
    for cmd in fix_commands:
        print(f"\n→ Running: {cmd}")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"  ❌ Command failed with exit code {ret.returncode}")
            failed = True
        else:
            print("  ✅ Done")

    print()
    if failed:
        print("Some fixes failed. Run `harmonia check` to see remaining issues.")
        sys.exit(1)
    else:
        print("All fixes applied! Run `harmonia check` to verify.")
        sys.exit(0)


if __name__ == "__main__":
    main()
