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

## Artifact preparation

Prepare the raw client shards, shared public vocabulary, and immutable untouched
evaluation dataset:

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
an interruption leaves the prior generation selected. The unsupervised split is
verified and excluded. The
generated directories simulate client-scoped
storage only after this centralized preparation; they are not evidence that data
originated at or remained hidden within independent organizations. Clients
tokenize their own mounted reviews, and no centrally tokenized partitions are
generated. The evaluation directory is immutable and evaluation-only: it must not
be mounted into ClientApp, the dashboard, or the current training ServerApp. PR14
and PR15 will add the registered consumers and metrics; this command performs no
evaluation.

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

## Direct Flower simulation

Run the Flower app directly using the per-partition raw directories generated
above. This is the fast local development path and does not use Docker:

```bash
uv run --env-file .env.protocol flwr run . --stream --federation-config "num-supernodes=4"
```

Run the dashboard against the selected server artifact directory:

```bash
FML_SERVER_ARTIFACT_DIR=artifacts/server uv run --env-file .env.protocol streamlit run dashboard.py
```

On PowerShell:

```powershell
$env:FML_SERVER_ARTIFACT_DIR = "artifacts/server"
uv run --env-file .env.protocol streamlit run dashboard.py
```

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

```powershell
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
seeds, and SHA-256 checksums for the public manifest and vocabulary. Each completed
`artifact_manifest.json` binds the saved model, metrics, and provenance file bytes
to their SHA-256 checksums, which consumers validate and snapshot before loading.
Set
`FML_CODE_REVISION` to the full Git object ID in images that do not contain `.git`.
Client shard identities and checksums are not collected by the server manifest.
In this demo the operator still created all shards centrally; the boundary only
describes what ServerApp reads at runtime.

`artifact-retention-runs` defaults to `10`. Retention orders validated run
manifests by `(created_at, run_id)`, keeps the newest configured count, and never
deletes the active or currently selected run. Malformed or unrecognized run
directories are left untouched for manual recovery.

## Local distributed Docker runtime

Prepare the four client shards, public artifacts, and host-only evaluation
artifact before starting the runtime:

```bash
uv run --env-file .env.protocol python -m src.data_prep --partitions 4 --client-shard-dir artifacts/clients --public-artifact-dir artifacts/public --evaluation-artifact-dir artifacts/evaluation
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

The dashboard is available at <http://127.0.0.1:8501>. Each ClientApp receives
only its matching `artifacts/clients/client-N` shard as a read-only mount. Public
artifacts are read-only in every consuming service; only ServerApp can write
`artifacts/server`, which the dashboard mounts read-only. The untouched
`artifacts/evaluation` directory is not mounted into this PR13 runtime. Stop the
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
