# Local Operations

These procedures cover the supported single-host Compose demonstration. They do
not establish production disaster recovery.

## Secure Flower transport and SuperNode authentication

The default Compose file is loopback-only and intentionally insecure. The
secure override encrypts the Fleet, Control, ServerAppIo, and ClientAppIo
channels and requires a registered P-384 identity for every SuperNode.

Generate disposable local-development credentials:

```bash
./scripts/generate-dev-flower-credentials.sh
docker compose -f compose.yaml -f compose.secure.yaml up -d superlink
```

Add this profile to `~/.flwr/config.toml` using an absolute CA path:

```toml
[superlink.local-secure]
address = "127.0.0.1:9093"
root-certificates = "/absolute/path/to/secrets/flower/ca.crt"
```

Register each public identity before starting its SuperNode:

```bash
for id in 0 1 2 3; do
  uv run flwr supernode register "secrets/flower/supernode-${id}.pub" local-secure
done
docker compose -f compose.yaml -f compose.secure.yaml up --build -d
```

The generator is for local development only. Production deployments must use a
managed CA, distinct leaf keys, restricted secret delivery, expiry monitoring,
and audited rotation. Flower 1.32.1 does not document certificate hot reload or
revocation. Rotate by first distributing a CA bundle that trusts old and new
CAs, replace and restart each endpoint, validate peers, and remove the old CA
only after migration. Never commit `secrets/`; private keys are mounted
read-only at runtime.

## Backup

Stop writers and archive both persistent volumes:

```bash
docker compose stop
mkdir -p backups
docker compose run --rm --no-deps -T serverapp \
  tar -C /app/artifacts/server -czf - . > backups/server-artifacts.tgz
docker compose run --rm --no-deps -T superlink \
  tar -C /app/state -czf - . > backups/superlink-state.tgz
sha256sum backups/*.tgz > backups/SHA256SUMS
```

Store the archives, `SHA256SUMS`, the Git commit, and the image digest outside
the Docker host. A backup is not verified until it has been restored into
separate disposable volumes and the checksums and dashboard load succeed.

## Restore

Restore only while the stack is stopped. These commands erase the target
volumes before extraction:

```bash
sha256sum --check backups/SHA256SUMS
docker compose run --rm --no-deps -T serverapp sh -c \
  'find /app/artifacts/server -mindepth 1 -delete; tar -C /app/artifacts/server -xzf -' \
  < backups/server-artifacts.tgz
docker compose run --rm --no-deps -T superlink sh -c \
  'find /app/state -mindepth 1 -delete; tar -C /app/state -xzf -' \
  < backups/superlink-state.tgz
docker compose up -d
```

Verify `docker compose ps`, the service health checks, the dashboard, and the
selected run's artifact checksums after restoration.

## Checkpoints and resume

The server writes `checkpoint-round-XXXXXX.npz` after every successful
aggregation. A checkpoint is immutable once its run is complete and is covered
by the run's artifact checksums.

Resume from an explicit checkpoint with:

```bash
uv run --env-file .env.protocol flwr run . --stream \
  --run-config "resume-from-checkpoint='artifacts/server/runs/<run-id>/checkpoint-round-000010.npz'"
```

Resume starts a new run, resets Flower round and metric numbering, and executes
the configured number of additional rounds. It restores global model tensors
only; client-local state and optimizer state are intentionally not shared by
the current federated protocol. The source run is never modified.

## Rollback and disaster recovery

Before an update, record `git rev-parse HEAD` and
`docker image inspect federated-machine-learning:latest --format '{{index .RepoDigests 0}}'`.
If the update fails, stop the stack, restore the matching code and image, and
restore the pre-update volumes when the artifact compatibility policy requires
it. Never load newer artifacts with older code unless
[`COMPATIBILITY.md`](COMPATIBILITY.md) explicitly permits that schema pair.

For host loss, provision a clean Docker host, check out the recorded commit,
build the pinned image, create the Compose volumes, restore both verified
archives, and run the same health and artifact checks. Recovery time and
recovery point objectives are not guaranteed because this local demo has no
automated or replicated backups.

## Cost and teardown safeguards

The supported deployment uses existing local hardware, so its incremental
infrastructure price is zero aside from electricity and storage. The configured
ceilings total 22 CPU cores and 44 GiB of memory across 11 services; actual use
is workload-dependent. No cloud estimate is published because no cloud
architecture, region, service class, or availability target is supported.

`docker compose down` removes containers and the network but preserves both
persistent volumes. `docker compose down -v` permanently deletes experiment and
SuperLink state; run and verify a backup first.
