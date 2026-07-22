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

Artifact schemas are scoped by artifact kind. Public manifests and client shards
use schema `2`; server artifact manifests use schema `4`; the prepared-generation
index uses schema `3`; evaluation artifacts, run provenance, and the current-run
index use schema `1`:

- `artifacts/public/manifest.json` uses `schema_version: 2` and describes the
  frozen train-dataset identity, checksummed vocabulary, and model dimensions.
- Each `client-N/client_metadata.json` uses `schema_version: 2`, describes its
  client-scoped raw-review shard, and binds its records to the exact public
  manifest used by the consumer.
- `.prepared-current/index.json` uses `schema_version: 3`; the atomic
  `.prepared-current` directory link selects one
  immutable directory under `.prepared-generations/<generation-id>`. Its client,
  public, and evaluation children are always selected as one generation. The
  canonical index bytes must record the same UUIDv4 as the selected directory and
  bind the artifact-affecting preparation request and the exact size and SHA-256
  digest of every public, client, and evaluation file. The durable migration
  journal carries the same request and independently binds the canonical index
  checksum.
- `artifacts/evaluation/manifest.json` uses `schema_version: 1` and strictly
  versions and checksums the immutable official test-split JSONL artifact.
- Each completed `server/runs/<run_id>` directory has schema-4
  `artifact_manifest.json`, which binds the Keras model and both metrics CSV
  layouts to the exact public manifest, vocabulary, and model dimensions as one
  consistent artifact set. The canonical public `manifest.json` and `vocab.txt`
  bytes are retained and checksummed with every output. A completed directory has
  exactly `artifact_manifest.json` plus the checksummed regular files; no other
  entry type or unmanifested name is accepted.
- Each run directory has an immutable schema-1 `run_manifest.json`, which versions
  run identity, configuration, environment, code, seed, and public-dataset
  provenance as one schema-checked record. Its effective unsigned 32-bit
  `master-seed` and SHA-256 namespaced derivation contract reproduce model,
  Dropout/training-order, and client/round update-noise streams. Its
  string-encoded public-dataset identity must itself be strict, canonical JSON
  with the exact train-identity fields and valid field types and values. An
  available private shard records exactly the SHA-256 bindings for
  `client_metadata.json` and `reviews.jsonl`; empty, partial, or extended checksum
  inventories are invalid.
- Schema-1 `server/current.json` atomically selects a completed run and binds its
  artifact manifest by SHA-256 checksum.

Both current-pointer and explicit historical consumers validate completed-run
provenance bytes, directory/run identity, frozen public dataset and vocabulary
bindings, checksums, and exact inventory before returning artifact bytes. Current,
historical, and evaluation loaders retain the complete no-follow directory chain for
the whole load, perform reads and inventory checks descriptor-relatively, and
revalidate the visible chain immediately before returning; an exact byte-for-byte tree
replacement is therefore not accepted as the selected path.
Publication retains no-follow descriptors and exact directory-edge identities from
the filesystem root through the artifact root, canonical `runs/` directory, and
selected run, plus every captured file. Creation flushes each owning directory before
using a newly created artifact-root or `runs/` edge. Publication and pruning revalidate
the complete visible chain around every mutation, durability barrier, rollback, and
successful return; retaining or locking a detached root inode never proves visible-path
ownership. Publication flushes the retained files, completed run directory, and `runs/`
directory, revalidates the exact entry inventory, identities, and bytes, replaces
`current.json` descriptor-relatively, and finally flushes the retained artifact root.
The exclusive temporary pointer remains open through replacement; the installed entry
must retain that inode and exact bytes before and after the root flush. Finalizers
serialize on that retained root inode rather than a replaceable lock entry. Before the
final barriers, a private,
strict-canonical state file is atomically replaced and flushed with the exact candidate
pointer plus the exact previous pointer identity, bytes, and checksum, or its absence.
A restart accepts only those two recorded pointer states: it retries publication from
the exact previous pointer or completes recovery from the exact candidate. Any other
pointer is rejected. State transitions use exclusive temporary files, file and root
flushes, and atomic descriptor-relative replacement, so a failed transition cannot
truncate the prior durable state. Successful publication and one-time recovery still
reject later refinalization.

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
artifacts to produce server schema `4`. Evaluation, run-provenance, and current-run
schema `1` artifacts do not require a schema migration. Prepared-generation schema
`1` did not bind the requested partition count and is no longer selectable. A
pending schema-1 preparation is safely rolled back and its journal-owned candidate
discarded before regeneration; an already selected schema-1 generation remains
immutable until a schema-3 generation atomically supersedes it.

Prepared-generation schema `2` did not independently bind the selected file
inventory. Pending schema-2 preparations are rolled back rather than recovered;
regenerate them as schema `3`. Server schema `2` completion did not retain file
sizes or guarantee one retained byte snapshot for provenance validation and
checksumming. Server schema `3` did not retain canonical public evidence or reject
unmanifested completed-run entries. Rerun training to produce server schema `4`;
editing an older manifest cannot reconstruct the missing evidence.

Artifacts are derived outputs, so regeneration into another root remains
non-destructive:

```bash
uv run --env-file .env.protocol python -m src.data_prep --partitions 4 --client-shard-dir artifacts/regenerated-public-v2/clients --public-artifact-dir artifacts/regenerated-public-v2/public --evaluation-artifact-dir artifacts/regenerated-public-v2/evaluation
uv run --env-file .env.protocol flwr run . --stream --federation-config "num-supernodes=4" --run-config "client-data-dir='artifacts/regenerated-public-v2/clients/client-{partition}' public-artifact-dir='artifacts/regenerated-public-v2/public' server-artifact-dir='artifacts/regenerated-public-v2/server'"
```

Do not edit a schema version or checksum by hand: that bypasses compatibility
checks without converting the data. A newer artifact requires newer application
code.

Direct artifact preparation and validation require Linux. Windows operators must
run the Linux `data-prep` Compose profile from a WSL2 Linux-filesystem checkout;
NTFS bind mounts are not supported for prepared links. macOS operators use the
same Linux-container profile. Unsupported native execution fails before
filesystem mutation. The profile binds the repository `artifacts` path unchanged,
accepts the invoking Linux user's UID/GID, and keeps `artifacts/server` read-only.

When generation publication first encounters real legacy logical directories,
it retains them under `.prepared-legacy/<generation-id>/` and replaces the
logical names with links to `.prepared-current`. This controlled migration never
deletes legacy bytes. Unrecognized legacy files remain archive-only and are never
promoted into a selected client, public, or evaluation generation. Public
generations contain exactly `manifest.json` and the checksummed `vocab.txt`;
client and evaluation loaders likewise enforce their exact inventories. Inspect
and remove an archive only with all consumers stopped. Compose mounts public and
per-client inputs directly through `.prepared-current`; evaluation remains
unmounted. Because bind sources are resolved when containers are created, stop
containers before regeneration and recreate them afterward.

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
Complete-state pruning is serialized on the same retained root inode as publication.
It removes a pruned run's complete or provably obsolete finalization state and flushes
the root, but protects recoverable pending or malformed state and its run. A failed run
deletion leaves its state untouched. Recursive deletion first detaches only the name
that still selects the retained inode, traverses from retained descriptors without
following links, flushes child removals before parent removal, and never deletes a
replacement installed at the public name.

For a schema change, update the producer, every shared loader, rejection and
acceptance tests, this policy, and the schema constant in the same pull request.
