# federated_machine_learning
Competing streaming platforms collect customer reviews. Each client represents one platform; clients jointly train a sentiment classifier while each platform's feedback data stays in-house. 

# 🚀 Project Startup & Maintainence

This project utilizes **uv** for efficient management of Python versions and dependencies. This ensures a consistent development environment across all team members.

---

## 🛠 1. Install uv
Before you begin, `uv` must be installed globally on your system:

* **Windows (PowerShell):**
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
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

### Running the Project
```
uv run src/main.py
```
Or running the guided walkthrough `research.ipynb` (Does not save data)

## 🤝 Contributing
Open a new Branch for a new feature. Open a pull request and let someone else review it instead of merging directly.  
Before commiting run following command to format and check the codebase  
```
uv run ruff check --fix .
uv run ruff format .
```

---

# 🧱 Tech Stack
1. Data Preparation: HuggingFace, pandas
2. Local Machine Learning: Keras with TensorFlow (?)
3. Federated Machine Learning: Flower
4. Containerization: Docker
5. Cloud Deployment: Google Cloud
6. Dashboard & Analysis: Streamlit (?)
