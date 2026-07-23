# Local Operations

These procedures cover the supported single-host Compose demonstration. They do
not establish production disaster recovery.

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
