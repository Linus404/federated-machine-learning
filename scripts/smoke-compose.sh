#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null || ! docker info >/dev/null 2>&1; then
  echo "SKIP: Docker and a reachable daemon are required for the Compose smoke test."
  exit 0
fi

prepared_root="${FML_PREPARED_ARTIFACT_ROOT:-artifacts/.prepared-current}"
if [[ ! -d "$prepared_root/public" || ! -d "$prepared_root/client/client-3" ]]; then
  echo "SKIP: run src.data_prep first or set FML_PREPARED_ARTIFACT_ROOT."
  exit 0
fi

project="fml-smoke-$$"
home_dir="$(mktemp -d)"
network="${project}-default"
export FML_CONTROL_BIND="127.0.0.1:0"
export FML_DASHBOARD_BIND="127.0.0.1:0"
export FML_PREPARED_ARTIFACT_ROOT
FML_PREPARED_ARTIFACT_ROOT="$(realpath "$prepared_root")"
compose=(
  docker compose
  -f compose.yaml
  -f "$home_dir/network.yaml"
  -p "$project"
)
cleanup() {
  "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
  docker network remove "$network" >/dev/null 2>&1 || true
  rm -r "$home_dir"
}
trap cleanup EXIT

subnet=""
for octet in {240..255}; do
  candidate="10.$octet.0.0/24"
  if docker network create --subnet "$candidate" "$network" >/dev/null 2>&1; then
    subnet="$candidate"
    break
  fi
done
if [[ -z "$subnet" ]]; then
  echo "ERROR: no isolated smoke-test subnet is available." >&2
  exit 1
fi

cat >"$home_dir/network.yaml" <<EOF
networks:
  default:
    external: true
    name: "$network"
EOF

"${compose[@]}" config --quiet
"${compose[@]}" build
"${compose[@]}" up --detach --wait
control_address="$("${compose[@]}" port superlink 9093)"

mkdir -p "$home_dir/.flwr"
cat >"$home_dir/.flwr/config.toml" <<EOF
[superlink]
default = "compose"

[superlink.compose]
address = "$control_address"
insecure = true
EOF

HOME="$home_dir" uv run flwr run . compose --stream --run-config \
  'num-server-rounds=1 expected-client-count=4 local-epochs=1 batch-size=16384 artifact-retention-runs=1' \
  | tee "$home_dir/run.log"
grep -q '"event":"fit_round_completed"' "$home_dir/run.log"
echo "PASS: all Compose roles became healthy and four clients completed one Flower round."
