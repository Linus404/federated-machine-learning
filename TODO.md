# Production Readiness TODO

This roadmap tracks the work required to turn the project from a strong academic
prototype into a polished portfolio project and, eventually, a production-ready
federated learning system.

## P0 — Portfolio and engineering foundation

- [x] Add a GitHub Actions CI workflow for unit tests, Ruff linting, and formatting.
- [x] Add type checking and enforce it in CI.
- [x] Measure test coverage and define a meaningful minimum coverage threshold.
- [x] Add Docker build and Compose validation smoke tests to CI.
- [x] Add dependency, secret, and container vulnerability scanning.
- [ ] Choose an open-source license and add a `LICENSE` file.
- [x] Add `CONTRIBUTING.md`, `SECURITY.md`, and pull-request/issue templates.
- [x] Move notebook-only dependencies such as `ipykernel` and `nbformat` out of runtime dependencies.
- [x] Align Ruff's target Python version with the supported Python versions.

## P0 — Scientific validation

- [ ] Define a fixed, untouched global test set.
- [ ] Add a centralized-training baseline.
- [ ] Compare local-only, FedAvg, FedProx, and FedProx plus Huber aggregation.
- [ ] Compare IID and multiple non-IID partition settings.
- [ ] Run experiments with multiple random seeds and report mean and variance.
- [ ] Report accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
- [ ] Measure convergence time, communication volume, and client training time.
- [ ] Evaluate robustness with controlled outlier and malicious-client scenarios.
- [ ] Publish reproducible commands and a results table in the README.
- [ ] Store the code version, run configuration, dataset version, seeds, and artifact checksums for every experiment.

## P0 — Privacy and security claims

- [ ] Write an explicit threat model covering trusted and untrusted components.
- [ ] Clearly distinguish the synthetic demo's centralized shard preparation from real client-owned data.
- [ ] Document what information model updates can leak.
- [ ] Add TLS for Flower communication.
- [ ] Add SuperNode/client authentication and certificate lifecycle documentation.
- [ ] Evaluate secure aggregation.
- [ ] If differential privacy is claimed, implement formal privacy accounting and publish epsilon/delta values.
- [ ] Add membership-inference and model-update leakage experiments.
- [ ] Keep the current update-noise feature labeled as an illustrative ablation, not production differential privacy.

## P1 — Reliability and testing

- [ ] Add a real end-to-end Flower test with multiple clients and at least one training round.
- [ ] Add a containerized end-to-end smoke test.
- [ ] Make the required client count configurable throughout server and deployment code.
- [ ] Test partial client participation, disconnects, timeouts, and failed rounds.
- [ ] Test malformed, missing, incompatible, and corrupted manifests and artifacts.
- [ ] Test invalid model updates, incompatible tensor shapes, NaN values, and empty aggregation inputs.
- [ ] Add dashboard tests for metric loading, inference errors, and missing artifacts.
- [ ] Test model save/load compatibility.
- [ ] Add a supported Python and operating-system test matrix.
- [ ] Add deployment-script tests or dry-run validation.

## P1 — Observability and experiment lifecycle

- [ ] Assign a unique ID to every training run.
- [ ] Persist the complete run configuration and environment metadata with each run.
- [ ] Add structured application logs with consistent levels and context.
- [ ] Record client availability, failures, round duration, and communication metrics.
- [ ] Add model checkpoints and documented resume behavior.
- [ ] Introduce model and artifact versioning with retention rules.
- [ ] Preserve experiment history instead of relying on manually selected artifact directories.
- [ ] Add health checks for containers and deployed services.
- [ ] Add monitoring dashboards and alerts for deployed runs.
- [ ] Define backup, restore, rollback, and disaster-recovery procedures.

## P1 — Deployment hardening

- [ ] Run containers as a non-root user.
- [ ] Pin base images by digest and define an update policy.
- [ ] Add read-only root filesystems where possible.
- [ ] Drop unnecessary Linux capabilities and enable `no-new-privileges`.
- [ ] Add CPU, memory, and process limits.
- [ ] Protect the Streamlit dashboard with authentication or an authenticated proxy.
- [ ] Re-enable and correctly configure CORS and XSRF protection.
- [ ] Replace ad-hoc cloud provisioning with reviewed infrastructure as code.
- [ ] Separate development, staging, and production configuration.
- [ ] Add centralized log collection and deployment audit trails.
- [ ] Document cost estimates and teardown safeguards.

## P1 — Portfolio presentation

- [ ] Rewrite the README around the problem, architecture, results, and demo before installation details.
- [ ] Add an architecture diagram showing server, clients, artifacts, and trust boundaries.
- [ ] Add dashboard screenshots and a short demo GIF or video.
- [ ] Add a concise results table and explain the main findings.
- [ ] Add CI, coverage, Python, license, and Docker badges.
- [ ] Add a quick-start path that produces a small result in a few minutes.
- [ ] Explain key design decisions and rejected alternatives.
- [ ] Add a limitations and responsible-use section.
- [ ] Add a model card and dataset card.
- [ ] Standardize all dashboard text to one language.
- [ ] Remove stale branch instructions and clean up the research notebook.
- [ ] Add a roadmap and a resume-ready project summary.

## P2 — Advanced production capabilities

- [ ] Add scalable client sampling and configurable participation policies.
- [ ] Benchmark behavior with substantially more simulated clients.
- [ ] Add robust aggregation alternatives and document their assumptions.
- [ ] Add model-update validation and anomaly detection.
- [x] Define API and artifact compatibility/versioning policies.
- [ ] Add canary deployment and model rollback support.
- [ ] Add performance profiling and optimize serialization and communication overhead.
- [ ] Conduct a documented security and privacy review before making production claims.

## Definition of done

The project can be presented as an exceptional portfolio project when CI is green,
results are reproducible and published, the architecture and limitations are clear,
and a reviewer can run a small end-to-end demonstration from a clean checkout.

The project should only be described as production-ready after authenticated and
encrypted communication, operational monitoring, hardened deployment, failure
recovery, and evidence-backed privacy guarantees are implemented and tested.
