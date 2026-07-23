# Security Policy

## Supported versions

This project has not published stable releases. Security fixes are applied to
the current `main` branch; older commits and forks are not supported.

## System security posture

The supported runtime is a local research demo, not a production security
boundary. The secure Compose overlay implements TLS for Flower endpoints and
P-384 SuperNode identity authentication; the production overlay adds an
authenticated dashboard proxy. The base loopback profile remains intentionally
insecure. This repository does not implement secure aggregation, encryption at
rest, or formal differential privacy. `use-update-noise` is an illustrative
ablation, not formal differential privacy, and has no accountant, composition or
sensitivity model, or epsilon/delta guarantee.

Read [THREAT_MODEL.md](THREAT_MODEL.md) for the actual data flow, trusted and
untrusted components, information exposed through parameters, metrics, and
artifacts, and controls required before production or privacy claims.
The [secure-aggregation evaluation](docs/adr/0001-secure-aggregation.md) records
the deferred Flower 1.32.1 integration decision.
The [security and privacy review](docs/SECURITY_PRIVACY_REVIEW.md) records
implemented controls, verification evidence, and residual risks.

## Report a vulnerability

Do not open a public issue, pull request, discussion, or proof-of-concept for a
suspected vulnerability. Submit a private report through
[GitHub Security Advisories](https://github.com/Linus404/federated-machine-learning/security/advisories/new).

Include, when available:

- the affected commit and component;
- the vulnerability's impact and required preconditions;
- minimal reproduction steps or a proof of concept;
- suggested mitigations; and
- whether the report may be credited publicly.

Do not include real credentials, private datasets, personal data, or data from
systems you do not own. Test only against environments you control.

## Response process

Maintainers will aim to acknowledge a complete report within seven calendar
days. They will validate the issue, assess its severity and affected versions,
and coordinate a fix and disclosure timeline with the reporter. Complex issues
may take longer; material status changes will be shared through the private
advisory.

Please keep the report confidential until maintainers confirm that a fix or
mitigation is available. A public advisory will credit the reporter if desired
and if disclosure is safe.

## Scope

Reports about code or configuration maintained in this repository are in
scope. Vulnerabilities that exist only in an upstream dependency should be
reported to that dependency's maintainers; reports showing a project-specific
impact or unsafe integration remain welcome here.

This policy does not authorize access to third-party systems, denial-of-service
testing, social engineering, or collection of other people's data.

## Automated scanning policy

CI uses Gitleaks to reject detected secrets and Trivy to inspect locked runtime
dependencies and the application image. Trivy first prints UNKNOWN, LOW,
MEDIUM, HIGH, and CRITICAL findings, including unfixed findings. The dependency
gate rejects every unsuppressed HIGH or CRITICAL finding; the container gate
rejects unsuppressed HIGH or CRITICAL findings for which a fix is available.

Exceptions must identify one finding and affected package in
`.trivyignore.yaml`, explain why remediation is not currently possible, and
expire within 30 days. CI uses Trivy's suppressed-findings output so accepted
risks remain visible in logs. Renewing an exception requires a fresh review;
blanket, unbounded, and undocumented exceptions are not accepted. Secret
findings cannot be excepted: revoke the credential, remove it from the
repository and history where practical, and rerun the scan.

Run equivalent scans locally with:

```bash
docker run --rm -v "$PWD:/repo" zricethezav/gitleaks:v8.30.1 git --redact --verbose /repo
docker run --rm -v "$PWD:/repo" aquasec/trivy:0.72.0 fs --scanners vuln --pkg-types library --ignorefile /repo/.trivyignore.yaml --show-suppressed --severity HIGH,CRITICAL --exit-code 1 /repo
image_archive="$(mktemp)"
docker save --output "$image_archive" federated-machine-learning:latest
docker run --rm -v "$image_archive:/image.tar:ro" -v "$PWD/.trivyignore.yaml:/.trivyignore.yaml:ro" aquasec/trivy:0.72.0 image --input /image.tar --scanners vuln --pkg-types os,library --ignorefile /.trivyignore.yaml --show-suppressed --ignore-unfixed --severity HIGH,CRITICAL --exit-code 1
```
