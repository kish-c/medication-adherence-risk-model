#!/usr/bin/env bash
set -e

echo "=============================="
echo "  Medication Adherence Pipeline"
echo "=============================="

echo ""
echo "[1/5] Generating synthetic data..."
python -m src.data_generate

echo ""
echo "[2/5] Engineering features..."
python -m src.feature_engineering

echo ""
echo "[3/5] Preparing train/test split..."
python -m src.data_prep

echo ""
echo "[4/5] Training models..."
python -m src.train

echo ""
echo "[5/5] Evaluating and generating figures..."
python -m src.evaluate

echo ""
echo "=============================="
echo "  Pipeline complete!"
echo "  Figures → reports/figures/"
echo "  Metrics → reports/metrics.json"
echo "=============================="
