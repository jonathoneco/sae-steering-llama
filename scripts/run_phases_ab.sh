#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
source ~/.local/secrets/secrets.env

echo "=== Phase A: Layer Sweep (resuming) ==="
python scripts/10_layer_sweep.py

echo ""
echo "=== Phase A: Generating plots ==="
python scripts/11_plot_layer_sweep.py

echo ""
echo "=== Phase B: Concept Discovery ==="
python scripts/12_concept_discovery.py

echo ""
echo "=== Phase B: Concept Sweep ==="
python scripts/13_concept_sweep.py

echo ""
echo "=== Phases A+B complete! ==="
