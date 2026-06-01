#!/bin/bash
# Deploy audio-edge to Jetson Orin NX via rsync.
#
# Usage:
#   ./scripts/deploy_to_jetson.sh [jetson-ip] [--restart]
#
# Examples:
#   ./scripts/deploy_to_jetson.sh 192.168.1.100
#   ./scripts/deploy_to_jetson.sh 192.168.1.100 --restart
#   JETSON_USER=nvidia ./scripts/deploy_to_jetson.sh 192.168.1.100

set -euo pipefail

JETSON_IP="${1:-}"
if [ -z "$JETSON_IP" ]; then
    echo "Usage: $0 <jetson-ip> [--restart]"
    echo "  Example: $0 192.168.1.100"
    exit 1
fi

JETSON_USER="${JETSON_USER:-jetson}"
JETSON_PATH="${JETSON_PATH:-/home/$JETSON_USER/audio-edge}"
RESTART="${2:-}"

echo "=== Deploying audio-edge to Jetson ==="
echo "  Target: ${JETSON_USER}@${JETSON_IP}:${JETSON_PATH}"

# Sync source code (exclude large/generated files)
rsync -avz --progress \
    --exclude 'models/vad/' \
    --exclude 'models/kws/' \
    --exclude 'models/sed/' \
    --exclude 'models/asc/' \
    --exclude 'models/engines/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'venv/' \
    --exclude 'logs/' \
    --exclude '.pytest_cache/' \
    ./ "${JETSON_USER}@${JETSON_IP}:${JETSON_PATH}/"

echo ""
echo "Deployed to ${JETSON_USER}@${JETSON_IP}:${JETSON_PATH}"

# Optional: restart systemd service
if [ "$RESTART" = "--restart" ]; then
    echo "Restarting audio-edge service..."
    ssh "${JETSON_USER}@${JETSON_IP}" "sudo systemctl restart audio-edge.service"
    echo "Service restarted."
fi

echo ""
echo "=== Next steps on Jetson ==="
echo "  ssh ${JETSON_USER}@${JETSON_IP}"
echo "  cd ${JETSON_PATH}"
echo "  pip install -e ."
echo "  python scripts/download_models.py"
echo "  python scripts/build_trt_engines.py"
echo "  python -m src.cli run --config configs/jetson_trt.yaml"
