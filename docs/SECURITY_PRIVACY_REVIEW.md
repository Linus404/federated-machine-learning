# Security and privacy review

**Review date:** 2026-07-23  
**Decision:** suitable for a controlled local portfolio/research demonstration;
not approved or described as production-ready or privacy-preserving.

## Reviewed controls and evidence

| Area | Implemented control | Evidence |
| --- | --- | --- |
| Transport identity | TLS on configured Flower endpoints and P-384 SuperNode authentication | `compose.secure.yaml`, certificate generator tests, secure Compose validation |
| Dashboard access | Digest-pinned non-root Nginx Basic Auth proxy; no direct production dashboard port | `compose.production.yaml`, 401/200 proxy smoke, deployment contract tests |
| Update integrity | Client-ID/count, tensor count/shape/dtype/finite validation; failed rounds hard-fail | server aggregation tests and real Flower E2E |
| Poisoning visibility | L2/MAD update anomaly report; Huber, median, and trimmed-mean alternatives | aggregation tests and controlled attack benchmark |
| Artifact integrity | Atomic publication, immutable run history, SHA-256 manifests, provenance, checkpoints | artifact corruption, compatibility, resume, and retention tests |
| Runtime hardening | Non-root/read-only containers, dropped capabilities, resource limits, health checks | Compose contract tests and container E2E |
| Audit/recovery | Structured logs, release labels, rotated Docker logs, backup/restore, canary/rollback | `OPERATIONS.md`, deployment tests |
| Privacy evidence | Frozen membership and update-leakage metric evaluators | privacy golden-vector and input-validation tests |

The verification suite also includes dependency/container vulnerability scans,
secret scanning, Python/OS CI matrices, model compatibility tests, malformed
artifact tests, participation/failure tests, and real multi-client training.

## Residual risks

- The server sees individual updates because secure aggregation is not implemented.
- Basic Auth requires external TLS termination for remote browser access.
- Client data is centrally prepared for the demo; it is not independently owned.
- At-rest artifacts and Docker volumes are not encrypted by this repository.
- Docker-local logs are not immutable off-host compliance records.
- A compromised authenticated client can poison data, metrics, or valid-looking
  updates; anomaly reports are evidence, not automated attribution.
- The update-noise ablation has no accountant, composition, sensitivity model,
  epsilon, or delta and is not formal differential privacy.
- Empirical membership/update attacks cover only supplied candidates and cannot
  prove privacy or absence of leakage.
- No cloud provider, organization IAM, multi-host availability target, incident
  response organization, or regulated-data contract is defined.

## Production claim gate

A production claim requires a concrete target architecture and a new review that
adds off-host monitoring/audit retention, encrypted storage, incident response,
availability and rate-limit objectives, organization-specific authorization and
secret management, and network-adversary testing. A privacy-preserving claim also
requires secure aggregation where applicable and/or a formally accounted privacy
mechanism with published parameters.
