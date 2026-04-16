#!/bin/bash
# ============================================================
# setup.sh — One-time venv environment setup
# Usage: bash setup.sh
# ============================================================

set -e

echo "======================================"
echo " nexus — Environment Setup"
echo "======================================"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[OK] venv created"
else
    echo "[SKIP] venv already exists"
fi

source venv/bin/activate
pip install --upgrade pip -q

pip install \
    mlflow==2.13.0 \
    tbparse==0.0.8 \
    tensorboard==2.16.2 \
    tensorboardX \
    pandas \
    rich \
    -q

echo ""
echo "[OK] All packages installed"
echo ""
echo "Activate with:  source venv/bin/activate"
echo "Then see:       README.md"
