# Compatibility and versioning policy

## Application interfaces

The project version in `pyproject.toml` follows semantic versioning. Before
`1.0.0`, a minor release may change documented command-line flags, Flower run
configuration keys, environment variables, or artifact contracts. Patch releases
must remain backward compatible. Modules under `src` are internal unless the
README documents them as an entry point.

Breaking interface changes require a project-version bump, release notes, and an
updated example command. Renaming or removing a documented setting is breaking;
adding an optional setting with an unchanged default is compatible.

## Artifact schemas

Artifact schemas are scoped by artifact kind. Public manifests, client shards,
and server artifact manifests use schema `2`; the prepared-generation index,
evaluation artifacts, run provenance, and current-run index use schema `1`:

- `artifacts/public/manifest.json` uses `schema_version: 2` and describes the
  frozen train-dataset identity, checksummed vocabulary, and model dimensions.
- Each `client-N/client_metadata.json` uses `schema_version: 2`, describes its
  client-scoped raw-review shard, and binds its records to the exact public
  manifest used by the consumer.
- `.prepared-current/index.json` uses `schema_version: 1`; the atomic
  `.prepared-current` directory link selects one
  immutable directory under `.prepared-generations/<generation-id>`. Its client,
  public, and evaluation children are always selected as one generation.
- `artifacts/evaluation/manifest.json` uses `schema_version: 1` and strictly
  versions and checksums the immutable official test-split JSONL artifact.
- Each completed `server/runs/<run_id>` directory has schema-2
  `artifact_manifest.json`, which binds the Keras model and both metrics CSV
  layouts to the exact public manifest, vocabulary, and model dimensions as one
  consistent, checksummed artifact set.
- Each run directory has an immutable schema-1 `run_manifest.json`, which versions
  run identity, configuration, environment, code, seed, and public-dataset
  provenance as one schema-checked record.
- Schema-1 `server/current.json` atomically selects a completed run and binds its
  artifact manifest by SHA-256 checksum.

Each consumer accepts only its artifact kind's current schema. Missing,
non-integer, older, and newer versions are rejected before the artifact is used.
Extra metadata fields are compatible, but changing required field names, types,
meanings, filenames, CSV columns, or the model serialization format requires an
increment of that artifact kind's schema without changing unrelated schemas.

## Migration and retention

### Version 0.2.0 public-manifest migration

Public schema `1` manifests and client schema `1` shards predate the mandatory
dataset, record, and public-manifest bindings. They are rejected explicitly as
older and cannot be upgraded by editing a version field. Regenerate public schema
`2`, client schema `2`, and the atomic prepared generation from the frozen source
dataset. Server schema `1` artifacts are unbound to a public vocabulary and are
also rejected; rerun local or federated training against the regenerated public
artifacts to produce server schema `2`. Evaluation, run-provenance,
prepared-generation-index, and current-run schema `1` artifacts do not require a
schema migration.

Artifacts are derived outputs, so regeneration into another root remains
non-destructive:

```bash
uv run --env-file .env.protocol python -m src.data_prep --partitions 4 --client-shard-dir artifacts/regenerated-public-v2/clients --public-artifact-dir artifacts/regenerated-public-v2/public --evaluation-artifact-dir artifacts/regenerated-public-v2/evaluation
uv run --env-file .env.protocol flwr run . --stream --federation-config "num-supernodes=4" --run-config "client-data-dir='artifacts/regenerated-public-v2/clients/client-{partition}' public-artifact-dir='artifacts/regenerated-public-v2/public' server-artifact-dir='artifacts/regenerated-public-v2/server'"
```

Do not edit a schema version or checksum by hand: that bypasses compatibility
checks without converting the data. A newer artifact requires newer application
code.

An interrupted preparation can leave an immutable, unselected directory under
`.prepared-generations`; consumers ignore it because only `.prepared-current`
selects a generation. These stale generations are not deleted automatically,
because a process may still hold a resolved path. Remove them only while all data
consumers are stopped, retaining the generation named by the current index.

`artifact-retention-runs` defaults to `10`. Valid run manifests are ordered by
their UTC `created_at` value and then UUID, so pruning is deterministic. The
currently selected run and the active writer are always protected, even when that
temporarily exceeds the configured count. Directories with missing, malformed, or
mismatched provenance are never deleted automatically. Pruning also stops without
deleting anything when `current.json` is missing or does not select a fully
checksum-valid completed run.

For a schema change, update the producer, every shared loader, rejection and
acceptance tests, this policy, and the schema constant in the same pull request.
