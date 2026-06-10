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

### Runtime paths

The default data and artifact directories are configured in `[tool.flwr.app.config]` as `data-dir = "data"` and `artifact-dir = "artifacts"`. You can override them with Flower run config, for example:

```bash
uv run flwr run . --stream --federation-config "num-supernodes=4" --run-config "data-dir='data' artifact-dir='artifacts'"
```

Local scripts also respect the environment variables `FML_DATA_DIR` and `FML_ARTIFACT_DIR` for their default paths.

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
