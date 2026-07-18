#!/usr/bin/env bash
# Provision and start the complete five-VM Flower deployment from a checked-out repo.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-europe-west3-a}"
CLIENT_COUNT="${CLIENT_COUNT:-4}"
SERVER_VM="${SERVER_VM:-fml-server}"
CLIENT_PREFIX="${CLIENT_PREFIX:-fml-client}"
REMOTE_APP="${REMOTE_APP:-/opt/federated-machine-learning}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: ./deploy/gce/gce-bootstrap.sh --project PROJECT_ID [options]

Options:
  --project PROJECT_ID   Google Cloud project (or set PROJECT_ID)
  --zone ZONE            Compute Engine zone (default: europe-west3-a)
  --clients COUNT        Number of client VMs (default: 4)
EOF
}

while (($#)); do
  case "$1" in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --clients) CLIENT_COUNT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$PROJECT_ID" ]] || { echo "--project is required" >&2; exit 2; }
[[ "$CLIENT_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "--clients must be a positive integer" >&2; exit 2; }

gcloud_cmd=(gcloud --quiet --project="$PROJECT_ID")

remote() {
  local vm="$1"
  shift
  "${gcloud_cmd[@]}" compute ssh --tunnel-through-iap "$vm" --zone="$ZONE" --command="$*"
}

copy_to() {
  local source="$1" vm="$2" destination="$3"
  "${gcloud_cmd[@]}" compute scp --tunnel-through-iap "$source" "$vm:$destination" --zone="$ZONE"
}

copy_from() {
  local vm="$1" source="$2" destination="$3"
  "${gcloud_cmd[@]}" compute scp --tunnel-through-iap "$vm:$source" "$destination" --zone="$ZONE"
}

copy_tree_to() {
  local source="$1" vm="$2" destination="$3"
  "${gcloud_cmd[@]}" compute scp --tunnel-through-iap --recurse "$source" "$vm:$destination" --zone="$ZONE"
}

copy_tree_from() {
  local vm="$1" source="$2" destination="$3"
  "${gcloud_cmd[@]}" compute scp --tunnel-through-iap --recurse "$vm:$source" "$destination" --zone="$ZONE"
}

ensure_vm() {
  local vm="$1" tag="$2" disk_size="$3"
  if "${gcloud_cmd[@]}" compute instances describe "$vm" --zone="$ZONE" >/dev/null 2>&1; then
    local status
    status="$("${gcloud_cmd[@]}" compute instances describe "$vm" --zone="$ZONE" --format='value(status)')"
    if [[ "$status" != "RUNNING" ]]; then
      "${gcloud_cmd[@]}" compute instances start "$vm" --zone="$ZONE"
    fi
  else
    "${gcloud_cmd[@]}" compute instances create "$vm" \
      --zone="$ZONE" \
      --machine-type=e2-standard-4 \
      --boot-disk-size="$disk_size" \
      --image-family=ubuntu-2404-lts-amd64 \
      --image-project=ubuntu-os-cloud \
      --tags="$tag"
  fi
}

ensure_firewall() {
  local name="$1"
  shift
  if ! "${gcloud_cmd[@]}" compute firewall-rules describe "$name" >/dev/null 2>&1; then
    "${gcloud_cmd[@]}" compute firewall-rules create "$name" "$@"
  fi
}

echo "==> Enabling Google Cloud services"
"${gcloud_cmd[@]}" services enable compute.googleapis.com iap.googleapis.com

echo "==> Creating or starting VMs"
ensure_vm "$SERVER_VM" fml-server 50GB
for i in $(seq 0 $((CLIENT_COUNT - 1))); do
  ensure_vm "${CLIENT_PREFIX}-${i}" fml-client 40GB
done

echo "==> Configuring firewalls"
ensure_firewall fml-allow-iap-ssh \
  --network=default --direction=INGRESS --action=ALLOW --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 --target-tags=fml-server,fml-client
ensure_firewall fml-allow-flower-fleet \
  --network=default --direction=INGRESS --action=ALLOW --rules=tcp:9092 \
  --source-tags=fml-client --target-tags=fml-server

HOST_SETUP="$(mktemp)"
SOURCE_ARCHIVE="$(mktemp --suffix=.tar.gz)"
TRANSFER_DIR="$(mktemp -d)"
trap 'rm -f "$HOST_SETUP" "$SOURCE_ARCHIVE"; rm -rf "$TRANSFER_DIR"' EXIT
cat >"$HOST_SETUP" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo apt-get remove -y \
  docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc \
  || true
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<DOCKER_REPO
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
DOCKER_REPO
sudo apt-get update
sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
sudo systemctl daemon-reload
sudo systemctl stop docker.service docker.socket || true
sudo systemctl reset-failed docker.service docker.socket
sudo systemctl enable containerd.service docker.socket
sudo systemctl start containerd.service docker.socket
sudo systemctl start docker.service
sudo docker info >/dev/null
EOF

git -C "$REPO_ROOT" archive --format=tar.gz --output="$SOURCE_ARCHIVE" HEAD

echo "==> Installing Docker and copying the checked-out application"
for vm in "$SERVER_VM" $(seq 0 $((CLIENT_COUNT - 1)) | sed "s|^|${CLIENT_PREFIX}-|"); do
  copy_to "$HOST_SETUP" "$vm" /tmp/fml-host-setup.sh
  remote "$vm" 'bash /tmp/fml-host-setup.sh'
  copy_to "$SOURCE_ARCHIVE" "$vm" /tmp/fml-source.tar.gz
  remote "$vm" "
    set -e
    sudo mkdir -p /opt
    sudo rm -rf '$REMOTE_APP'
    sudo mkdir -p '$REMOTE_APP'
    sudo chown \"\$USER:\$USER\" '$REMOTE_APP'
    tar -xzf /tmp/fml-source.tar.gz -C '$REMOTE_APP'
    find '$REMOTE_APP/deploy' -type f -name '*.sh' -exec sed -i 's/\r$//' {} +
  "
done

echo "==> Building the server image and preparing client shards"
remote "$SERVER_VM" "
  set -e
  cd '$REMOTE_APP'
  mkdir -p artifacts/public artifacts/server artifacts/clients
  docker compose -f deploy/gce/server.compose.yaml build
  docker run --rm -v \"\$PWD/artifacts:/app/artifacts\" \
    federated-machine-learning:latest \
    uv run --no-sync python -m src.data_prep \
      --partitions '$CLIENT_COUNT' \
      --client-shard-dir artifacts/clients \
      --public-artifact-dir artifacts/public
"

echo "==> Distributing one private shard to each client"
copy_tree_from "$SERVER_VM" "$REMOTE_APP/artifacts/public" "$TRANSFER_DIR/"
for i in $(seq 0 $((CLIENT_COUNT - 1))); do
  copy_from "$SERVER_VM" "$REMOTE_APP/artifacts/clients/client-${i}.tar.gz" "$TRANSFER_DIR/"
  copy_tree_to "$TRANSFER_DIR/public" "${CLIENT_PREFIX}-${i}" "$REMOTE_APP/artifacts/"
  copy_to "$TRANSFER_DIR/client-${i}.tar.gz" "${CLIENT_PREFIX}-${i}" "$REMOTE_APP/artifacts/"
done

SERVER_INTERNAL_IP="$("${gcloud_cmd[@]}" compute instances describe "$SERVER_VM" --zone="$ZONE" --format='get(networkInterfaces[0].networkIP)')"

echo "==> Starting the Flower server and clients"
remote "$SERVER_VM" "
  set -e
  cd '$REMOTE_APP'
  sudo rm -rf artifacts/clients
  docker compose -f deploy/gce/server.compose.yaml up -d
"
for i in $(seq 0 $((CLIENT_COUNT - 1))); do
  remote "${CLIENT_PREFIX}-${i}" "
    set -e
    cd '$REMOTE_APP'
    mkdir -p artifacts/client-data/client-${i}
    tar -xzf artifacts/client-${i}.tar.gz -C artifacts/client-data/client-${i}
    rm artifacts/client-${i}.tar.gz
    cat > .env <<EOF
CLIENT_ID=${i}
SUPERLINK_ADDRESS=${SERVER_INTERNAL_IP}:9092
CLIENT_SHARD_DIR=$REMOTE_APP/artifacts/client-data/client-${i}
EOF
    docker compose --env-file .env -f deploy/gce/client.compose.yaml up -d --build
  "
done

echo "==> Deployment is ready"
echo "Connect: gcloud compute ssh --tunnel-through-iap $SERVER_VM --project=$PROJECT_ID --zone=$ZONE"
echo "Run:     cd $REMOTE_APP && ./deploy/gce/gce-run.sh"
echo "Dashboard tunnel: gcloud compute ssh --tunnel-through-iap $SERVER_VM --project=$PROJECT_ID --zone=$ZONE --ssh-flag='-N' --ssh-flag='-L 8501:localhost:8501'"
