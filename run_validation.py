#!/usr/bin/env python3
"""
Full validation run: tests, graph check, and real gate execution.
Run from Windows PowerShell:
    wsl.exe -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/brian/Desktop/projects/niel_landa && python run_validation.py"
Or directly from Windows:
    python run_validation.py
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent


def run_command(cmd, name):
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    start = time.time()
    result = subprocess.run(cmd, shell=True, cwd=REPO)
    elapsed = time.time() - start
    status = "✓ PASS" if result.returncode == 0 else "✗ FAIL"
    print(f"\n{status} ({elapsed:.1f}s)\n")
    return result.returncode


def main():
    """Run validation pipeline."""
    results = {}

    # Step 1: pytest
    results["pytest"] = run_command(
        f"{sys.executable} -m pytest tests/ -v --tb=short",
        "STEP 1: Run pytest",
    )

    # Step 2: validate_graph
    results["graph"] = run_command(
        f"{sys.executable} tests/validate_graph.py --strict",
        "STEP 2: Validate graph",
    )

    # Step 3: Real gate execution
    print("\n" + "=" * 60)
    print("STEP 3: Execute BBQ existence gate (REAL)")
    print("=" * 60)
    print("⚠️  This step runs inference on 47996 items (~2 hours)")
    print("Starting at", time.strftime("%H:%M:%S"))
    start_time = time.time()

    results["gate"] = run_command(
        f"{sys.executable} scripts/02_bbq_existence_gate.py --verbose",
        "STEP 3: BBQ Existence Gate"
    )

    elapsed = time.time() - start_time
    print(f"Gate execution took {elapsed/60:.1f} minutes")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for name, code in results.items():
        status = "✓" if code == 0 else "✗"
        print(f"{status} {name}: {'PASS' if code == 0 else 'FAIL'}")

    all_pass = all(code == 0 for code in results.values())
    print()
    if all_pass:
        print("🎉 ALL VALIDATIONS PASSED")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
