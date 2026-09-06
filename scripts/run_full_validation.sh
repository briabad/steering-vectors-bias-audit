#!/bin/bash
# Full validation: tests, graph check, and real gate execution
# Run in WSL with: bash scripts/run_full_validation.sh

set -e  # Exit on error

REPO="/mnt/c/Users/brian/Desktop/projects/niel_landa"
VENV="$HOME/.venvs/niel_landa/bin"
PYTHON="$VENV/python"

cd "$REPO"

echo "=========================================="
echo "STEP 1: Run pytest"
echo "=========================================="
$PYTHON -m pytest tests/ -v --tb=short
PYTEST_EXIT=$?

echo ""
echo "=========================================="
echo "STEP 2: Validate graph"
echo "=========================================="
$PYTHON tests/validate_graph.py --strict
GRAPH_EXIT=$?

echo ""
echo "=========================================="
echo "STEP 3: Execute BBQ existence gate (REAL)"
echo "=========================================="
echo "Starting at $(date '+%H:%M:%S')"
START_TIME=$(date +%s)
$PYTHON scripts/02_bbq_existence_gate.py --verbose
GATE_EXIT=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Completed at $(date '+%H:%M:%S') (${ELAPSED}s elapsed)"

echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="
echo "pytest: $([ $PYTEST_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "graph: $([ $GRAPH_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "gate: $([ $GATE_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"

if [ $PYTEST_EXIT -ne 0 ] || [ $GRAPH_EXIT -ne 0 ] || [ $GATE_EXIT -ne 0 ]; then
    exit 1
fi

exit 0
