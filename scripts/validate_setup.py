#!/usr/bin/env python3
"""Validate that the project structure is correctly set up."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

print("[1/5] Checking pyproject.toml...")
pyproject = repo_root / "pyproject.toml"
if pyproject.exists():
    print(f"      OK: {pyproject}")
else:
    print(f"      FAIL: {pyproject} not found")
    sys.exit(1)

print("[2/5] Checking src/bbq_gate structure...")
for path in [
    "src/bbq_gate/__init__.py",
    "src/bbq_gate/domain/__init__.py",
    "src/bbq_gate/application/__init__.py",
    "src/bbq_gate/infrastructure/__init__.py",
]:
    full_path = repo_root / path
    if full_path.exists():
        print(f"      OK: {path}")
    else:
        print(f"      FAIL: {path} not found")
        sys.exit(1)

print("[3/5] Testing imports (no GPU)...")
try:
    import bbq_gate
    print(f"      OK: import bbq_gate")
    import bbq_gate.domain
    print(f"      OK: import bbq_gate.domain")
    import bbq_gate.application
    print(f"      OK: import bbq_gate.application")
    import bbq_gate.infrastructure
    print(f"      OK: import bbq_gate.infrastructure")
except ImportError as e:
    print(f"      FAIL: {e}")
    sys.exit(1)

print("[4/5] Testing domain entities...")
try:
    from bbq_gate.domain.entities import BBQItem, BBQOption, AnswerRole
    print(f"      OK: imported domain entities")
except ImportError as e:
    print(f"      FAIL: {e}")
    sys.exit(1)

print("[5/5] Testing domain resolution...")
try:
    from bbq_gate.domain.resolution import resolve_unknown_option, tag_matches_group
    print(f"      OK: imported resolution functions")
except ImportError as e:
    print(f"      FAIL: {e}")
    sys.exit(1)

print("\nSETUP VALIDATION PASSED")
