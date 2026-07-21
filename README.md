# Federated Machine Learning

Competing streaming platforms collect customer reviews. Each client represents one platform; clients jointly train a sentiment classifier while each platform's feedback data stays in-house.

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

## Artifact preparation

Prepare the raw client shards and shared public vocabulary:

```bash
uv run python -m src.data_prep --partitions 4 --client-shard-dir artifacts/clients --public-artifact-dir artifacts/public
```

This command loads IMDB once for the demo setup, creates one raw review/label
directory per client, and publishes the shared vocabulary. Clients
always tokenize their own reviews; no centrally tokenized partitions are generated.
The server sees only model updates. The model uses a normal trainable embedding,
not pretrained GloVe vectors or an embedding matrix.

Train one client locally from its raw shard:

```bash
uv run python -m src.local_training --client-data-dir artifacts/clients/client-0 --public-artifact-dir artifacts/public --run-artifact-dir artifacts/local-runs
```

The local command treats `--run-artifact-dir` as a reusable history root. Every
training invocation writes to a new immutable `runs/<run_id>` directory.

## Direct Flower simulation

Run the Flower app directly using the per-partition raw directories generated
above. This is the fast local development path and does not use Docker:

```bash
uv run flwr run . --stream --federation-config "num-supernodes=4"
```

Run the dashboard against the selected server artifact directory:

```bash
FML_SERVER_ARTIFACT_DIR=artifacts/baseline uv run streamlit run dashboard.py
```

On PowerShell:

```powershell
$env:FML_SERVER_ARTIFACT_DIR = "artifacts/baseline"
uv run streamlit run dashboard.py
```

### Runtime paths

The Flower configuration uses separate paths for public artifacts, server output,
and raw client shards. Its default client path is
`artifacts/clients/client-{partition}`. You can override paths with Flower run
config, for example:

```bash
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "client-data-dir='artifacts/clients/client-{partition}' server-artifact-dir='artifacts/server' public-artifact-dir='artifacts/public'"
```

The dashboard reads the completed run selected by the atomic `current.json` index
under `FML_SERVER_ARTIFACT_DIR` and reads public artifacts from
`FML_PUBLIC_ARTIFACT_DIR`. Deployed clients use `CLIENT_DATA_DIR` for their one
mounted private shard.

The server never clears its configured artifact root. It writes each experiment to
`runs/<run_id>` and advances `current.json` only after the run has a model,
metrics, provenance, and verified SHA-256 checksums. Existing completed runs remain
available for comparison:

```powershell
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "use-update-noise=false use-huber=false"
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "use-update-noise=false use-huber=true huber-threshold=10.0"
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "use-update-noise=true use-huber=false"
```

`use-huber` enables the robust aggregation path for outlier-resistant experiments. `use-update-noise` enables a small illustrative client update-noise ablation; it is not a production differential-privacy guarantee.

Every server run and the documented local-training command create an immutable
`run_manifest.json` before training. It
contains a UUID, the Flower run ID, creation time, complete run configuration,
Python/OS/package versions, Git revision and worktree state when available, known
seeds, and SHA-256 checksums for the public manifest and vocabulary. Completed
run manifests also bind the saved model, metrics, and provenance files to their
SHA-256 checksums, which consumers validate before loading. Set
`FML_CODE_REVISION` to the full Git object ID in images that do not contain `.git`.
Private client datasets stay outside the server trust boundary, so their identity
and checksums are not collected by the server manifest.

`artifact-retention-runs` defaults to `10`. Retention orders validated run
manifests by `(created_at, run_id)`, keeps the newest configured count, and never
deletes the active or currently selected run. Malformed or unrecognized run
directories are left untouched for manual recovery.

## Local distributed Docker runtime

Prepare the four client shards and public artifacts on the host before starting
the runtime:

```bash
uv run python -m src.data_prep --partitions 4 --client-shard-dir artifacts/clients --public-artifact-dir artifacts/public
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
uv run python -m src.flower_config
uv run flwr run . local-docker --stream
```

Flower 1.29 reads SuperLink connection profiles from `~/.flwr/config.toml` rather
than the project configuration. The configuration command preserves unrelated
profiles, rejects an existing `local-docker` profile unless it already points to
the loopback Control API with insecure local transport, and waits until exactly
four SuperNodes are registered online. The insecure connection is appropriate
only for this loopback-only local runtime; do not use it for a remote or public
SuperLink.

The dashboard is available at <http://127.0.0.1:8501>. Each ClientApp receives
only its matching `artifacts/clients/client-N` shard as a read-only mount. Public
artifacts are read-only in every consuming service; only ServerApp can write
`artifacts/server`, which the dashboard mounts read-only. Stop the runtime with
`docker compose down`.

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
uv run ruff format .
uv run ruff check .
```

See [COMPATIBILITY.md](COMPATIBILITY.md) for the application and artifact
versioning policy and supported regeneration path.

Open a new branch for each feature and use pull requests for review.

## Tech stack

1. Data preparation: Hugging Face Datasets and pandas
2. Local machine learning: Keras with TensorFlow
3. Federated machine learning: Flower
4. Dashboard and analysis: Streamlit
