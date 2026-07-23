#!/usr/bin/env bash
set -euo pipefail

output="${1:-secrets/flower}"
days="${FLOWER_CERTIFICATE_DAYS:-30}"

if [[ ! "$days" =~ ^[1-9][0-9]*$ ]]; then
  printf 'FLOWER_CERTIFICATE_DAYS must be a positive integer\n' >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  printf 'refusing to replace existing credential directory: %s\n' "$output" >&2
  exit 2
fi

umask 077
mkdir -p "$output"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

openssl ecparam -name secp384r1 -genkey -noout -out "$output/ca.key"
openssl req -x509 -new -sha384 -key "$output/ca.key" -days "$days" \
  -subj "/CN=fml-development-ca" -out "$output/ca.crt"

issue_certificate() {
  local name="$1"
  local san="$2"
  openssl ecparam -name secp384r1 -genkey -noout -out "$output/$name.key"
  openssl req -new -sha384 -key "$output/$name.key" \
    -subj "/CN=$name" -out "$work/$name.csr"
  printf 'subjectAltName=%s\nextendedKeyUsage=serverAuth\n' "$san" \
    > "$work/$name.ext"
  openssl x509 -req -sha384 -in "$work/$name.csr" \
    -CA "$output/ca.crt" -CAkey "$output/ca.key" -CAcreateserial \
    -days "$days" -extfile "$work/$name.ext" -out "$output/$name.pem"
}

issue_certificate "fleet-control" "DNS:superlink,DNS:localhost,IP:127.0.0.1"
issue_certificate "serverappio" "DNS:superlink"
for client_id in 0 1 2 3; do
  issue_certificate "supernode-${client_id}-appio" "DNS:supernode-${client_id}"
  ssh-keygen -q -t ecdsa -b 384 -N "" \
    -C "fml-supernode-${client_id}" -f "$output/supernode-${client_id}"
done

chmod 0600 "$output"/*.key "$output"/supernode-[0-3]
chmod 0644 "$output"/*.crt "$output"/*.pem "$output"/*.pub
printf 'development credentials written to %s\n' "$output"
