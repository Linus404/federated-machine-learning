# Federated Machine Learning

This research demo simulates competing streaming platforms that jointly train a
sentiment classifier. The local data-preparation command centrally creates all
demo shards; it does not demonstrate ingestion from independent data owners.

## Project setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python versions and dependencies.

### Install uv

**Windows (PowerShell)**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If execution is blocked, run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` first.

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

Every documented local Python entry point uses `--env-file .env.protocol` so the
frozen hash seed, Keras backend, and TensorFlow determinism settings exist before
the interpreter starts. Missing or conflicting startup values are rejected; do
not set them later from Python code.

Artifact publication and validation are supported directly on Linux only. They
depend on Linux descriptor-relative and no-follow filesystem primitives; native
Windows and macOS execution fails before creating files instead of using weaker
fallback checks. On Windows, use Docker Desktop from a WSL2 Linux-filesystem
checkout as documented below; do not prepare through an NTFS bind mount whose
link semantics cannot satisfy this contract. macOS operators likewise use the
Linux-container workflow. Other dependency-management commands remain usable on
their documented platforms.

## Artifact preparation

On Linux, prepare the raw client shards, shared public vocabulary, and immutable
untouched evaluation dataset directly:

```bash
uv run --env-file .env.protocol python -m src.data_prep --partitions 4 --client-shard-dir artifacts/clients --public-artifact-dir artifacts/public --evaluation-artifact-dir artifacts/evaluation
```

This command loads `stanfordnlp/imdb` configuration `plain_text` at the frozen
revision, verifies every official split and raw/content SHA-256, creates every raw
review/label shard from the training split only, and publishes a vocabulary
adapted on that same split. It writes the official test split in ascending source
order to the selected prepared generation, with stable `test:<row-index>`
identities and a strict checksum manifest. One atomic `.prepared-current`
switch selects the matching client, public, and evaluation directories together;
after an interruption during later preparation, either the prior generation or
the fully prepared new generation remains selected, depending on whether the
atomic pointer replacement completed. During the first legacy migration,
preparation may be interrupted after archiving the legacy roots but before
all aliases and the pointer are installed, leaving the logical roots only
partially visible and no generation selected. The durable journal makes that
state recoverable without byte loss, but preparation must be rerun to complete
selection. The unsupervised split is
verified and excluded. The
generated directories simulate client-scoped
storage only after this centralized preparation; they are not evidence that data
originated at or remained hidden within independent organizations. Clients
tokenize their own mounted reviews, and no centrally tokenized partitions are
generated. The evaluation directory is immutable and evaluation-only: it must not
be mounted into ClientApp, the dashboard, or the current training ServerApp. PR14
and PR15 will add the registered consumers and metrics; this command performs no
evaluation.

The first generation created over real legacy `artifacts/clients`,
`artifacts/public`, or `artifacts/evaluation` directories migrates those complete
directories by same-filesystem rename into
`artifacts/.prepared-legacy/<generation-id>/`. It then installs the logical paths
as aliases to `.prepared-current`. Nothing in the legacy directories is deleted;
inspect and remove an archived copy only while every consumer is stopped.
Unrecognized legacy files remain only in that archive and are never copied into a
selected client, public, or evaluation generation. Public generations expose
exactly the canonical `manifest.json` and checksummed `vocab.txt` inventory.

The server does not read raw shard files during training, but it receives each
client's resulting model parameters, sample counts, training metrics, and
per-client evaluation loss, accuracy, and client ID before or during aggregation.
Those values can leak information about client data. The model uses a normal
trainable embedding, not pretrained GloVe vectors or an embedding matrix.

Train one client locally from its raw shard:

```bash
uv run --env-file .env.protocol python -m src.local_training --client-data-dir artifacts/clients/client-0 --public-artifact-dir artifacts/public --run-artifact-dir artifacts/local-runs
```

The local command treats `--run-artifact-dir` as a reusable history root. Every
training invocation writes to a new `runs/<run_id>` directory; finalization binds
its regular files into a checksum-verified artifact snapshot.

## Direct Flower simulation (Linux)

On Linux, run the Flower app directly using the per-partition raw directories
generated above. This is the fast local development path and does not use Docker:

```bash
uv run --env-file .env.protocol flwr run . --stream --federation-config "num-supernodes=4"
```

Run the dashboard against the selected server artifact directory:

```bash
FML_SERVER_ARTIFACT_DIR=artifacts/server uv run --env-file .env.protocol streamlit run dashboard.py
```

On Windows and macOS, use the containerized dashboard in the local distributed
runtime below so artifact validation remains inside Linux.

### Runtime paths

The Flower configuration uses separate paths for public artifacts, server output,
and raw client shards. Data preparation also uses the separate evaluation-only
path `artifacts/evaluation`, overridable with `--evaluation-artifact-dir` or
`FML_EVALUATION_ARTIFACT_DIR`. Its default client path is
`artifacts/clients/client-{partition}`. You can override paths with Flower run
config, for example:

```bash
uv run --env-file .env.protocol flwr run . --stream --federation-config "num-supernodes=4" --run-config "client-data-dir='artifacts/clients/client-{partition}' server-artifact-dir='artifacts/server' public-artifact-dir='artifacts/public'"
```

The dashboard reads the completed run selected by the atomic `current.json` index
under `FML_SERVER_ARTIFACT_DIR` and reads public artifacts from
`FML_PUBLIC_ARTIFACT_DIR`. Each ClientApp uses `CLIENT_DATA_DIR` for its one
mounted client-scoped shard.

The server never clears its configured artifact root. It writes each experiment to
`runs/<run_id>` and advances `current.json` only after the run has a model,
metrics, provenance, and verified SHA-256 checksums. Existing completed runs remain
available for comparison:

```bash
uv run --env-file .env.protocol flwr run . --stream --federation-config "num-supernodes=4" --run-config "use-update-noise=false use-huber=false"
uv run --env-file .env.protocol flwr run . --stream --federation-config "num-supernodes=4" --run-config "use-update-noise=false use-huber=true huber-threshold=10.0"
uv run --env-file .env.protocol flwr run . --stream --federation-config "num-supernodes=4" --run-config "use-update-noise=true use-huber=false"
```

`use-huber` enables an experimental robust aggregation path for outlier-resistance
experiments; it is not a Byzantine-security guarantee. `use-update-noise` is an
illustrative ablation, not formal differential privacy: it has no privacy
accountant, composition or sensitivity model, or published epsilon/delta and must
not be presented as a privacy guarantee.

Every server run and the documented local-training command create an immutable
`run_manifest.json` before training. It
contains a UUID, the Flower run ID, creation time, complete run configuration,
Python/OS/package versions, Git revision and worktree state when available, known
seeds, and SHA-256 checksums for the public manifest and vocabulary. The effective
`master-seed` defaults to `67`; SHA-256 domain-separated namespaces deterministically
derive distinct model-construction, client/round Dropout and training-order, and
client/round update-noise streams. Each completed
`artifact_manifest.json` binds the saved model, metrics, and provenance file bytes
to their SHA-256 checksums. Completed runs also retain the canonical public
`manifest.json` and `vocab.txt`; consumers validate those bytes against the frozen
protocol, require the provenance and model dimensions to match them, and reject
any unmanifested directory entry before loading. Current and explicit historical
loads apply the same completed-run provenance, directory-identity, public-evidence,
vocabulary, checksum, and exact-inventory validation boundary.
Set
`FML_CODE_REVISION` to the full Git object ID in images that do not contain `.git`.
Client shard identities and checksums are not collected by the server manifest.
In this demo the operator still created all shards centrally; the boundary only
describes what ServerApp reads at runtime.

Completed-run publication retains no-follow descriptors and filesystem identities
for the artifact root, canonical `runs/` directory, selected run, and every captured
file. Finalizers lock the retained root inode, so replacing a visible lock pathname
cannot split serialization across same-run or different-run publishers. Before the
final barriers, an atomically replaced private state file records the exact candidate
pointer and the exact previous pointer identity, bytes, and checksum, or absence. It
is written through an exclusive temporary file, flushed, renamed descriptor-relatively,
and followed by a retained-root flush without truncating prior durable state. A retry
accepts only the recorded previous pointer, from which it republishes, or the exact
candidate pointer, from which it completes recovery after full revalidation and every
barrier. Divergent pointers are rejected, and successful publication or one-time
recovery rejects later retries. No publication call reports success unless every
required barrier completes.

`artifact-retention-runs` defaults to `10`. Retention orders validated run
manifests by `(created_at, run_id)`, keeps the newest configured count, and never
deletes the active or currently selected run. Malformed or unrecognized run
directories are left untouched for manual recovery.

## Local distributed Docker runtime

Stop the runtime before every preparation. Already-running containers retain the
immutable directory resolved when their bind mount was created, so they do not
follow a later `.prepared-current` switch. `docker compose down`, prepare, and
then recreate the stack; this prevents a run from mixing generations.

On Linux, direct preparation remains available as shown above. Set the host IDs
when using the Linux-container workflow so generated files retain host ownership:

```bash
docker compose down
mkdir -p artifacts artifacts/server
FML_HOST_UID="$(id -u)" FML_HOST_GID="$(id -g)" docker compose --profile prepare run --rm --build data-prep
```

On Windows, enable Docker Desktop's WSL2 integration and keep the checkout in the
WSL2 Linux filesystem (for example `~/src/federated-machine-learning-independent`,
not `/mnt/c/...`). Run the same commands from the WSL shell. This preserves Linux
links, repository-relative paths, and the WSL user's ownership:

```bash
docker compose down
mkdir -p artifacts artifacts/server
FML_HOST_UID="$(id -u)" FML_HOST_GID="$(id -g)" docker compose --profile prepare run --rm --build data-prep
```

Build the single application image and start the separate SuperLink, ServerApp,
four SuperNode/ClientApp pairs, and dashboard services:

```bash
FML_CODE_REVISION="$(git rev-parse HEAD)" docker compose up --build -d
```

The launch command passes the exact host Git revision to ServerApp and every
ClientApp without adding repository metadata to the image.

Submit the Flower app to the running local federation:

```bash
uv run --env-file .env.protocol python -m src.flower_config
uv run --env-file .env.protocol flwr run . local-docker --stream
```

Flower 1.32.1 reads SuperLink connection profiles from `~/.flwr/config.toml` rather
than the project configuration. The configuration command preserves unrelated
profiles, rejects an existing `local-docker` profile unless it already points to
the loopback Control API with insecure local transport, and waits until exactly
four SuperNodes are registered online. The insecure connection is appropriate
only for this loopback-only local runtime; do not use it for a remote or public
SuperLink.

The dashboard is available at <http://127.0.0.1:8501>. Compose resolves every
public and client bind source below `.prepared-current`, so a legacy directory
cannot shadow the selected generation. Each ClientApp receives only its matching
client shard as a read-only mount. Public artifacts are read-only in every
consuming service; only ServerApp can write `artifacts/server`, which the
dashboard and offline preparation service mount read-only. The untouched
evaluation directory is not mounted into any PR13 runtime service. Stop the
runtime with `docker compose down`.

### Security and privacy scope

This repository does not implement TLS, client or SuperNode authentication,
secure aggregation, or formal differential privacy. The dashboard has no
application authentication. Model parameters, metrics, sample counts, and
artifacts can leak information even though ServerApp does not read raw shard
files. The [secure-aggregation evaluation](docs/adr/0001-secure-aggregation.md)
records why Flower SecAgg+ adoption is deferred. See the explicit
[threat model](THREAT_MODEL.md) for assets, trust boundaries, attack surfaces,
and requirements that must be met before production or privacy claims.

### Deployment scope

The previous multi-host Google Cloud deployment was removed because the project
currently has no cloud access for live validation. The supported deployment path
is the local Docker runtime above. Reintroduce a cloud deployment only when it can
be tested end to end against real infrastructure.

## Development

Add dependencies with:

```bash
uv add package-name
```

Upgrade locked dependencies with:

```bash
uv lock --upgrade
```

Before committing, format and check the codebase:

```bash
uv run --env-file .env.protocol ruff format .
uv run --env-file .env.protocol ruff check .
```

See [COMPATIBILITY.md](COMPATIBILITY.md) for the application and artifact
versioning policy and supported regeneration path.

Open a new branch for each feature and use pull requests for review.

## Tech stack

1. Data preparation: Hugging Face Datasets and pandas
2. Local machine learning: Keras with TensorFlow
3. Federated machine learning: Flower
4. Dashboard and analysis: Streamlit
