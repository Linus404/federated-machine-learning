# Completed Academic Project Roadmap

This file records the completed scope of the academic project. A checked item
means that the capability was implemented or that its design decision and
limitation were documented. It does not assert production readiness.

## P0 — Portfolio and engineering foundation

- [x] Add a GitHub Actions CI workflow for unit tests, Ruff linting, and formatting.
- [x] Add type checking and enforce it in CI.
- [x] Measure test coverage and define a meaningful minimum coverage threshold.
- [x] Add Docker build and Compose validation smoke tests to CI.
- [x] Add dependency, secret, and container vulnerability scanning.
- [x] Choose an open-source license and add a `LICENSE` file.
- [x] Add `CONTRIBUTING.md`, `SECURITY.md`, and pull-request/issue templates.
- [x] Move notebook-only dependencies such as `ipykernel` and `nbformat` out of runtime dependencies.
- [x] Align Ruff's target Python version with the supported Python versions.

## P0 — Scientific validation

- [x] Define a fixed, untouched global test set.
- [x] Add a centralized-training baseline.
- [x] Compare local-only, FedAvg, FedProx, and FedProx plus Huber aggregation.
- [x] Compare IID and multiple non-IID partition settings.
- [x] Run experiments with multiple random seeds and report mean and variance.
- [x] Report accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
- [x] Measure convergence time, communication volume, and client training time.
- [x] Evaluate robustness with controlled outlier and malicious-client scenarios.
- [x] Publish reproducible commands and a results table in the README.
- [x] Store the code version, run configuration, dataset version, seeds, and artifact checksums for every experiment.

## P0 — Privacy and security claims

- [x] Write an explicit threat model covering trusted and untrusted components.
- [x] Clearly distinguish the simulated demo's centralized shard preparation from real client-owned data.
- [x] Document what information model updates can leak.
- [x] Add TLS for Flower communication.
- [x] Add SuperNode/client authentication and certificate lifecycle documentation.
- [x] Evaluate and document the secure-aggregation decision.
  - Decision: defer Flower 1.32.1 SecAgg+ until its documented integration,
    incompatibilities, and end-to-end prerequisites are resolved; see
    `docs/adr/0001-secure-aggregation.md`.
- [x] Resolve the differential-privacy claim requirements.
  - Resolved by explicit non-claim: the current update-noise ablation has no
    accountant, composition or sensitivity model, so no epsilon/delta or formal
    differential-privacy claim is made.
- [x] Implement membership-inference and model-update leakage evaluators.
  - Exact evaluators and golden-vector tests are implemented. The published campaign
    does not include attack results because it did not retain the required candidate
    probabilities, gradients, and individual client updates.
- [x] Keep the current update-noise feature labeled as an illustrative ablation, not production differential privacy.

## P1 — Reliability and testing

- [x] Add a real end-to-end Flower test with multiple clients and at least one training round.
- [x] Add a containerized end-to-end smoke test.
- [x] Make the required client count configurable throughout server and deployment code.
- [x] Test partial client participation, disconnects, timeouts, and failed rounds.
- [x] Test malformed, missing, incompatible, and corrupted manifests and artifacts.
- [x] Test invalid model updates, incompatible tensor shapes, NaN values, and empty aggregation inputs.
- [x] Add dashboard tests for metric loading, inference errors, and missing artifacts.
- [x] Test model save/load compatibility.
- [x] Add a supported Python and operating-system test matrix.
- [x] Add deployment-script tests or dry-run validation.

## P1 — Observability and experiment lifecycle

- [x] Assign a unique ID to every training run.
- [x] Persist the complete run configuration and environment metadata with each run.
- [x] Add structured application logs with consistent levels and context.
- [x] Record client availability, failures, round duration, and communication metrics.
- [x] Add model checkpoints and documented resume behavior.
- [x] Introduce model and artifact versioning with retention rules.
- [x] Preserve experiment history instead of relying on manually selected artifact directories.
- [x] Add health checks for containers and deployed services.
- [x] Add monitoring dashboards and alerts for deployed runs.
- [x] Define backup, restore, rollback, and disaster-recovery procedures.

## P1 — Demonstration hardening

- [x] Run containers as a non-root user.
- [x] Pin base images by digest and define an update policy.
- [x] Add read-only root filesystems where possible.
- [x] Drop unnecessary Linux capabilities and enable `no-new-privileges`.
- [x] Add CPU, memory, and process limits.
- [x] Protect the Streamlit dashboard with authentication or an authenticated proxy.
- [x] Re-enable and correctly configure CORS and XSRF protection.
- [x] Define the supported infrastructure boundary.
  - Reviewed Compose configuration is the only supported infrastructure target.
    Provider-specific cloud infrastructure is outside the academic project scope.
- [x] Separate development, staging, and production configuration.
- [x] Add centralized log collection and deployment audit trails.
- [x] Document cost estimates and teardown safeguards.

## P1 — Portfolio presentation

- [x] Rewrite the README around the problem, architecture, results, and demo before installation details.
- [x] Add an architecture diagram showing server, clients, artifacts, and trust boundaries.
- [x] Add dashboard screenshots and a short demo GIF or video.
- [x] Add a concise results table and explain the main findings.
- [x] Add CI, coverage, Python, license, and Docker badges.
- [x] Add a quick-start path that produces a small result in a few minutes.
- [x] Explain key design decisions and rejected alternatives.
- [x] Add a limitations and responsible-use section.
- [x] Add a model card and dataset card.
- [x] Standardize all dashboard text to one language.
- [x] Remove stale branch instructions and clean up the research notebook.
- [x] Add a completed roadmap and concise project summary.

## P2 — Extended engineering capabilities

- [x] Add scalable client sampling and configurable participation policies.
- [x] Benchmark behavior with substantially more simulated clients.
  - The published aggregation microbenchmark covers 4, 16, and 64 clients; it is
    not presented as a 64-client end-to-end accuracy result.
- [x] Add robust aggregation alternatives and document their assumptions.
- [x] Add model-update validation and anomaly detection.
- [x] Define API and artifact compatibility/versioning policies.
- [x] Add canary deployment and model rollback support.
- [x] Add performance profiling and optimize serialization and communication overhead.
- [x] Conduct a documented security and privacy review before making production claims.

## Completion statement

The academic project is complete: results are reproducible and published, the
architecture and limitations are documented, CI enforces the repository contracts,
and a reviewer can run a small end-to-end demonstration from a clean checkout.

Production readiness is outside this project's scope and must not be inferred from
the demonstration hardening or security experiments.
