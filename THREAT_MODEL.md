# Threat Model

## Scope and security posture

This threat model describes the repository at the current `main` branch: a
local research demo built with Flower, Keras, Streamlit, and Docker Compose. It
is not a production deployment assessment.

This repository does not implement TLS, client or SuperNode authentication,
secure aggregation, encryption at rest, or formal differential privacy. The
local Compose topology binds host-facing ports to loopback, but its Flower
connections use insecure plaintext transport inside the host and Compose
network. Process and volume separation reduce accidental access; they do not
provide cryptographic confidentiality, integrity, or identity.

## Actual data flow

1. The operator runs `src.data_prep` on one preparation host. That process
   downloads the public IMDB training split, reads every review and label,
   adapts one vocabulary over the complete training split, and writes all
   client shard directories. The operator and preparation host can therefore
   access every demo record.
2. Each ClientApp reads one client-scoped shard and the shared public
   vocabulary. It tokenizes reviews, creates a local train/validation split,
   trains, and evaluates locally.
3. Flower sends each client's resulting model parameters to the server before
   aggregation. Fit results also carry the number of training examples and
   training metrics. Evaluation results carry loss, accuracy, validation sample
   count, and the configured client ID.
4. ServerApp aggregates the parameters and metrics. It writes the global model,
   aggregate round metrics, per-client evaluation metrics, run configuration,
   environment and code provenance, and checksums under the server artifact
   root.
5. The dashboard reads the selected server run and public vocabulary. In the
   supported Compose topology it is unauthenticated and bound to host loopback.

### Demo shards are not evidence of client-owned ingestion

The generated `artifacts/clients/client-N` directories simulate separate data
owners only after centralized preparation. Calling them client-scoped means
that each ClientApp is mounted one directory and ServerApp does not read those
raw files during training. It does not mean the source data originated at
independent organizations or remained hidden from the demo operator.

A real deployment would need each organization to acquire, validate, retain,
and expose its own data inside its own administrative boundary. It would also
need authenticated and encrypted transport, credential lifecycle management,
access controls, and operational monitoring that this repository does not
provide.

## Assets and observations

| Asset or observation | Present location or recipient | Confidentiality concern |
| --- | --- | --- |
| Raw reviews and labels | Preparation host; one client shard per ClientApp | Plain JSON records reveal the complete examples and labels. The demo operator has all shards. |
| Client shard metadata | Client shard directory | Sample count, label histogram, split seed, Dirichlet alpha, and manifest checksum reveal distribution information. |
| Public vocabulary and manifest | ServerApp, each ClientApp, and the dashboard | The vocabulary was adapted on the complete demo training split; token inclusion and ordering can reveal corpus properties. |
| Client model parameters | Flower server before aggregation | Parameters can encode information about local examples and distributions. |
| Fit and evaluation metadata | Flower server | Sample counts, loss, accuracy, client ID, participation, and timing can expose client size, behavior, and data-distribution differences. |
| Global model | Server artifacts and dashboard process | A released model can memorize training information and be queried for membership or property signals. |
| Server artifacts | ServerApp and dashboard mounts | Per-client metrics, aggregate metrics, configuration, environment, package versions, code revision, and checksums disclose experiment and operational details. |
| SuperLink state | Named Docker volume | Run and node orchestration state may disclose participation and topology information. |

Artifact SHA-256 checksums detect unexpected byte changes after publication.
They are not signatures, access control, encryption, or proof that an artifact
came from a trusted client.

## Trust boundaries and threat actors

| Component or actor | Trust assumption in this demo | Threats considered |
| --- | --- | --- |
| Operator and preparation host | Trusted with all demo input and generated shards | Accidental disclosure, malicious preparation, poisoned data or vocabulary, overly broad filesystem access. |
| ClientApp and its shard | Trusted to protect its own local files; untrusted by the server | A compromised or malicious client can disclose its shard, report false metrics, poison parameters, or send malformed values. |
| SuperNode, SuperLink, and ServerApp | Trusted to execute the demo; observable by the operator and honest-but-curious from a data owner's perspective | Individual updates and metadata can be inspected; compromised services can alter runs or artifacts. |
| Network and other containers | Untrusted | Plaintext Flower traffic can be observed or modified by an actor with host or network access. No peer identity is verified. |
| Dashboard user or process | Untrusted unless the operator controls host access | The dashboard has no application authentication and can expose model and metric artifacts available to its process. |
| Dataset, packages, images, and CI actions | External supply chain | A compromised input or dependency can execute code, poison data, or exfiltrate accessible files. Pinning and vulnerability scans reduce but do not eliminate this risk. |

Host compromise is not contained by this topology: the host operator can read
bind mounts, named volumes, container memory, environment variables, and local
plaintext traffic.

## Threats and current limitations

### Confidentiality and privacy

- Individual client parameters are visible to the server because secure
  aggregation is absent. Model inversion, gradient or parameter inversion,
  property inference, and membership inference may recover or infer information
  about local training examples.
- Sample counts, client IDs, label histograms, losses, accuracies, participation,
  and timing can reveal client population and distribution characteristics even
  when raw reviews are not transmitted by the application.
- Raw shards and artifacts are not encrypted at rest. Filesystem permissions and
  read-only mounts are the only repository-provided storage boundaries.
- The global model and vocabulary can leak information when copied, queried, or
  published. This repository has no membership-inference or model-update leakage
  evaluation yet.
- `use-update-noise` is an illustrative ablation, not formal differential
  privacy. It clips each tensor update separately and adds Gaussian noise, but
  defines no adjacency relation or sampling analysis, uses no privacy accountant,
  and publishes no epsilon or delta. It must not be used as a privacy guarantee.

### Integrity

- There is no client identity verification or message authentication. An actor
  able to reach or control the local Flower services may impersonate a component
  or tamper with traffic.
- Clients can submit poisoned, malformed, non-finite, or incompatible parameters
  and dishonest metrics. The optional Huber path is an experimental robust
  aggregation ablation, not proof of Byzantine security and not a substitute for
  validation or anomaly detection.
- Public manifests and completed artifacts are schema-checked and checksummed,
  but they are unsigned. A writer that can replace both an artifact and its
  checksum remains trusted.
- Centralized preparation can introduce biased partitions, malicious records, or
  a poisoned vocabulary for every client.

### Availability

- The supported Compose topology provides four SuperNode/ClientApp pairs. The
  server waits for at least four available clients and, with both participation
  fractions set to `1.0`, selects every available client for fit and evaluation;
  with the four-service topology, that means all four. FedProx retains its
  `accept_failures=True` default, so it can aggregate the successful results from
  a round with selected-client failures. A missing or slow client can delay
  selection when fewer than four are available, and a round with no successful
  results cannot be aggregated.
- There are no production retry, timeout, disaster-recovery, rate-limit, or
  denial-of-service controls.
- Model and artifact history improve recovery from completed local runs, but do
  not protect against host, volume, or operator failure.

## Existing safeguards

- Each Compose ClientApp receives one client shard through a read-only mount;
  ServerApp receives no client-shard mount.
- In Compose, public artifacts are mounted read-only into ServerApp, all four
  ClientApp services, and the dashboard. The dashboard's server-artifact mount
  is also read-only.
- Host-facing SuperLink and dashboard ports bind to loopback in the supported
  local Compose topology.
- Artifact schema validation, path containment checks, regular-file checks,
  atomic selection, and checksums detect several corruption and path-confusion
  failures.
- CI runs tests, type and format checks, secret scanning, dependency scanning,
  container scanning, and Compose validation.

These are engineering safeguards, not evidence of privacy preservation or a
production security boundary.

## Required before production or privacy claims

At minimum, a future deployment must add and test TLS for every Flower channel;
mutual component authentication and certificate rotation; authorization and
secret management; secure aggregation if the server must not inspect individual
updates; formal differential privacy with an explicit mechanism, adjacency
definition, sampling model, accountant, and published epsilon/delta when DP is
claimed; update validation and poisoning defenses; authenticated dashboard
access; encrypted storage; audit logs; incident response; backup and recovery;
and leakage, membership-inference, malicious-client, and network-adversary tests.
