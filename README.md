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
uv run python -m src.local_training --client-data-dir artifacts/clients/client-0 --public-artifact-dir artifacts/public
```

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

The dashboard reads server artifacts from `FML_SERVER_ARTIFACT_DIR` and public
artifacts from `FML_PUBLIC_ARTIFACT_DIR`. Deployed clients use `CLIENT_DATA_DIR`
for their one mounted private shard.

The server clears its configured artifact directory at the start of each run so stale metrics or models cannot bleed into fresh experiments. Use a distinct `server-artifact-dir` for each experiment you want to compare:

```powershell
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "server-artifact-dir='artifacts/baseline' use-update-noise=false use-huber=false"
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "server-artifact-dir='artifacts/huber' use-update-noise=false use-huber=true huber-threshold=10.0"
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "server-artifact-dir='artifacts/update-noise' use-update-noise=true use-huber=false"
```

`use-huber` enables the robust aggregation path for outlier-resistant experiments. `use-update-noise` enables a small illustrative client update-noise ablation; it is not a production differential-privacy guarantee.

## Local distributed Docker runtime

Prepare the four client shards and public artifacts on the host before starting
the runtime:

```bash
uv run python -m src.data_prep --partitions 4 --client-shard-dir artifacts/clients --public-artifact-dir artifacts/public
```

Build the single application image and start the separate SuperLink, ServerApp,
four SuperNode/ClientApp pairs, and dashboard services:

```bash
docker compose up --build -d
```

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

Open a new branch for each feature and use pull requests for review.

## Tech stack

1. Data preparation: Hugging Face Datasets and pandas
2. Local machine learning: Keras with TensorFlow
3. Federated machine learning: Flower
4. Dashboard and analysis: Streamlit
