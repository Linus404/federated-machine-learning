#!/usr/bin/env bash
# Submit one training run from the server VM after gce-bootstrap.sh has completed.
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p "$HOME/.flwr"
cat >"$HOME/.flwr/config.toml" <<'EOF'
[superlink.gce-fml]
address = "127.0.0.1:9093"
insecure = true
EOF

exec docker run --rm \
  --network host \
  -v "$PWD:/workspace" \
  -v "$HOME/.flwr:/root/.flwr" \
  -w /workspace \
  federated-machine-learning:latest \
  sh -c '
    uv run --no-sync python -m src.flower_readiness \
      --address 127.0.0.1:9093 \
      --expected-online 4 \
      --timeout 120 &&
    exec uv run --no-sync flwr run . gce-fml --stream \
      --run-config "embedding-dim=100"
  '
