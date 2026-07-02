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

Prepare the local data partitions:

```bash
uv run python -m src.data_prep
```

Train one local baseline model:

```bash
uv run python -m src.local_training
```

Run the Flower app using the configuration in `pyproject.toml`:

```bash
uv run flwr run . --stream --federation-config "num-supernodes=4"
```

Run the dashboard against the selected artifact directory:

```bash
FML_ARTIFACT_DIR=artifacts/baseline uv run streamlit run dashboard.py
```

On PowerShell:

```powershell
$env:FML_ARTIFACT_DIR = "artifacts/baseline"
uv run streamlit run dashboard.py
```

### Runtime paths

The default data and artifact directories are configured in `[tool.flwr.app.config]` as `data-dir = "data"` and `artifact-dir = "artifacts"`. You can override them with Flower run config, for example:

```bash
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "data-dir='data' artifact-dir='artifacts'"
```

Local scripts also respect the environment variables `FML_DATA_DIR` and `FML_ARTIFACT_DIR` for their default paths.

The server clears the configured artifact directory at the start of each run so stale metrics or models cannot bleed into fresh experiments. Use a distinct `artifact-dir` for each experiment you want to compare:

```powershell
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "artifact-dir='artifacts/baseline' use-update-noise=false use-huber=false"
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "artifact-dir='artifacts/huber' use-update-noise=false use-huber=true huber-threshold=10.0"
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "artifact-dir='artifacts/update-noise' use-update-noise=true use-huber=false"
```

`use-huber` enables the robust aggregation path for outlier-resistant experiments. `use-update-noise` enables a small illustrative client update-noise ablation; it is not a production differential-privacy guarantee.

## Docker

Build the image:

```bash
docker build -t federated-machine-learning .
```

Run the full Flower simulation with local `data/` and `artifacts/` mounted into the container:

```bash
docker compose up --build
```

Prepare data on the host first with `uv run python -m src.data_prep`, or run the same command inside the image by overriding the compose command. The Compose command still runs the full four-supernode simulation, but mounts data outside the packaged app directory and schedules one TensorFlow client at a time to avoid noisy Ray packaging warnings and low-memory Docker Desktop failures.

## Google Cloud deployment

The configured cloud deployment runs this project on a Google Compute Engine VM with Docker Compose. The VM runs the Flower training container and the Streamlit dashboard container, while `data/` and `artifacts/` are mounted into both containers.

### Local setup

Install and authenticate the Google Cloud CLI, then select the project that owns the VM:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Connect to the VM

```powershell
gcloud compute ssh fml-vm
```

On the VM, enter the repository:

```bash
cd federated-machine-learning
```

### Run the training and dashboard on the VM

Prepare the data partitions once:

```bash
docker compose run --rm federated uv run --no-sync python -m src.data_prep
```

Run the federated training:

```bash
docker compose up --build federated
```

Start the dashboard:

```bash
docker compose up -d dashboard
```

Inspect running containers and logs:

```bash
docker compose ps
docker compose logs -f federated
docker compose logs -f dashboard
```

### Open the dashboard from Windows

Keep this PowerShell session open while using the dashboard:

```powershell
gcloud compute ssh fml-vm --ssh-flag="-L 8502:localhost:8501"
```

Then open:

```text
http://localhost:8502
```

The tunnel maps Windows `localhost:8502` to the Streamlit dashboard running on VM port `8501`. Use another local port, for example `8510`, if `8502` is already in use:

```powershell
gcloud compute ssh fml-vm --ssh-flag="-L 8510:localhost:8501"
```

### Stop cloud costs after the demo

Stop the VM when it is not needed:

```powershell
gcloud compute instances stop fml-vm
```

Delete the VM when the deployment is no longer needed:

```powershell
gcloud compute instances delete fml-vm
```

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
