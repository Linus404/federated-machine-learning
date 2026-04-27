# federated_machine_learning
Competing streaming platforms collect customer reviews. Each client represents one platform; clients jointly train a sentiment classifier while each platform's feedback data stays in-house. 

# 🚀 Project Startup

This project utilizes **uv** for efficient management of Python versions and dependencies. This ensures a consistent development environment across all team members.

---

## 🛠 1. Install uv
Before you begin, `uv` must be installed globally on your system:

* **Windows (PowerShell):**
    ```powershell
    powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
    ```
    *Note: If execution is blocked, run `Set-ExecutionPolicy RemoteSigned -scope CurrentUser` first.*

* **macOS / Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

---

## 📦 2. Project Setup
After cloning the repository, navigate to the project directory and run:

```bash
uv sync
```

### Adding Packages
```bash
uv add package1 package2
```

### Update all Packages to their latest Version
```bash
uv lock --upgrade
```

## 🤝 Contributing
Open a new Branch for a new feature. Open a pull request and let someone else review it instead of merging directly.  
Before commiting run following command to format and lint the codebase  
Linux:
```
uv run ruff check --fix && uv run ruff format
```
bzw (ich glaube) following on Windows:
```
uv run ruff check --fix ; uv run ruff format
```
