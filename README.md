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

### Common commands

Prepare the raw client shards and shared public vocabulary:

```bash
uv run python -m src.data_prep --partitions 4 --client-shard-dir artifacts/clients --public-artifact-dir artifacts/public
```

This command loads IMDB once for the demo setup, creates one raw review/label
directory and archive per client, and publishes the shared vocabulary. Clients
always tokenize their own reviews; no centrally tokenized partitions are generated.
The server sees only model updates. The model uses a normal trainable embedding,
not pretrained GloVe vectors or an embedding matrix.

Train one client locally from its raw shard:

```bash
uv run python -m src.local_training --client-data-dir artifacts/clients/client-0 --public-artifact-dir artifacts/public
```

Run the Flower app using the per-partition raw directories generated above:

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

## Docker

Build the image:

```bash
docker build -t federated-machine-learning .
```

Run the Flower simulation with generated raw shards and artifacts mounted into the container:

```bash
docker compose up --build
```

Prepare data on the host first with `uv run python -m src.data_prep`. The Compose
command runs the four-supernode simulation and schedules one TensorFlow client at
a time to avoid low-memory Docker Desktop failures.

## Google Cloud deployment

The top-level `compose.yaml` flow runs a single-VM shared-filesystem simulation.

For realistic multi-VM Flower Runtime on Google Cloud (one server plus client VMs), use the deployment guide in [deploy/README.md](deploy/README.md).

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
