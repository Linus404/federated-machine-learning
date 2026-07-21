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

The current artifact schema is `1`. Every persisted contract carries
`schema_version: 1`:

- `artifacts/public/manifest.json` describes the vocabulary and model dimensions.
- Each `client-N/client_metadata.json` describes its private raw-review shard.
- Each server output directory has `artifact_manifest.json`, which versions the
  Keras model and both metrics CSV layouts as one consistent artifact set.
- Each server output directory has an immutable `run_manifest.json`, which versions
  run identity, configuration, environment, code, seed, and public-dataset
  provenance as one schema-checked record.

Consumers accept schema `1` only. Missing, non-integer, older, and newer versions
are rejected before the artifact is used. Extra metadata fields are compatible,
but changing required field names, types, meanings, filenames, CSV columns, or
the model serialization format requires a schema increment.

## Migration and retention

There is no in-place migration for schema `1`. Artifacts are derived outputs, so
the supported migration is to keep any historical directory needed for comparison
and regenerate a new directory with the target checkout:

```bash
uv run python -m src.data_prep --partitions 4 --client-shard-dir artifacts/regenerated-schema-1/clients --public-artifact-dir artifacts/regenerated-schema-1/public
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "client-data-dir='artifacts/regenerated-schema-1/clients/client-{partition}' public-artifact-dir='artifacts/regenerated-schema-1/public' server-artifact-dir='artifacts/regenerated-schema-1/server'"
```

Do not edit a schema version by hand: that bypasses compatibility checks without
converting the data. A newer artifact requires newer application code. Automated
retention and migration tools should be added only when a second supported schema
or durable experiment registry exists.

For a schema change, update the producer, every shared loader, rejection and
acceptance tests, this policy, and the schema constant in the same pull request.
