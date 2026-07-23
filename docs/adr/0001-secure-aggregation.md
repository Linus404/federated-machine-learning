# ADR 0001: Defer secure aggregation

- **Status:** Accepted
- **Decision:** Defer implementation
- **Evaluated version:** Flower 1.32.1, as resolved by `uv.lock`
- **Date:** 2026-07-22

## Context

The server currently receives every client's fit parameters and sample count before
`SentimentServer.aggregate_fit` runs. Flower 1.32.1 provides SecAgg+, but the
presence of its Python API does not mean this repository runs or validates the
protocol. Secure aggregation remains unimplemented.

Flower's SecAgg+ workflow masks each client's `[sample_count, sample_count *
parameters]` aggregation vector and reveals the recovered weighted aggregate to
the strategy. In the 1.32.1 `NumPyClient` compatibility path, the fit response's
declared `num_examples` and metrics remain separate records and are still visible
to ServerApp. The protocol adds setup, key-sharing, masked-vector collection, and
unmasking exchanges to each fit round.

## Decision

Defer SecAgg+ until the prerequisites and protocol tests below are complete. This
repository continues to use ordinary FedProx transport and must not claim that
individual updates are hidden from ServerApp.

SecAgg+ is the preferred future protocol because Flower supports it in both the
Simulation and Deployment Engines and it tolerates a configured amount of client
dropout. Adoption is not a small configuration change, however: it changes the
ServerApp execution model, numerical aggregation, failure behavior, and which
server-side defenses are possible.

## Required Flower 1.32.1 integration

The implementation must follow the paired integration documented by Flower:

1. Replace `ServerApp(server_fn=server_fn)` with `ServerApp()` and an
   `@app.main()` function receiving `Grid` and `Context`.
2. In that function, construct the existing strategy and `ServerConfig`, wrap them
   in `LegacyContext`, then execute
   `DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...))`.
3. Import `secaggplus_mod` from `flwr.client.mod` and construct
   `ClientApp(client_fn=client_fn, mods=[secaggplus_mod])`. The similarly named
   `flwr.clientapp.mod` namespace does not export this compatibility mod in 1.32.1.
4. Add and validate run configuration for `num_shares`,
   `reconstruction_threshold`, `max_weight`, `clipping_range`,
   `quantization_range`, `modulus_range`, and `timeout`.
5. Preserve the current artifact-lock acquisition, release-on-error behavior,
   final-round publication, and evaluation workflow while moving from `server_fn`
   to `@app.main()`.
6. Reject `use-huber=true` whenever SecAgg+ is enabled.

The current Huber aggregation is incompatible with SecAgg+. Huber compares
individual client vectors to reduce outlier influence, while the workflow hides
those vectors and supplies only the recovered weighted aggregate to
`Strategy.aggregate_fit`. Flower's compatibility path places that same aggregate
in each synthetic fit result; applying Huber there would compare duplicate
aggregate vectors, not client updates, and would provide no robust-aggregation
evidence.

## Candidate configuration and trade-offs

For the current four-client topology, the first tested candidate is
`num_shares=3` and `reconstruction_threshold=2`; it is a test hypothesis, not a
security guarantee. `max_weight` must be at least the largest permitted client
sample count. The clipping range must be selected from observed parameter ranges,
then accuracy and convergence must be compared with the unprotected baseline.

- More shares improve dropout robustness but increase communication and
  computation. A higher reconstruction threshold makes secret reconstruction
  harder for a small coalition but reduces dropout tolerance.
- Clipping and integer quantization add numerical error. A larger `max_weight` can
  reduce quantization precision.
- `modulus_range` must be a power of two and greater than
  `quantization_range`; the aggregate must also remain within the arithmetic
  bounds.
- A finite timeout bounds stalled protocol stages but can turn slow participants
  into dropouts. No timeout can wait indefinitely.
- Secure aggregation prevents inspection of individual fit updates, so the current
  Huber defense and future per-update validation or anomaly detection cannot run
  on those hidden vectors without a different design.

## Protection boundary

SecAgg+ would protect individual fit parameter vectors from ServerApp, revealing
their weighted aggregate. In this compatibility integration it would not protect
the separately encoded `num_examples`, fit metrics, raw reviews or labels on
clients, centrally created demo shards, public vocabulary, global model,
evaluation parameters or metrics, client identity and participation, message
timing and size, logs, or persisted artifacts. It is not transport encryption,
authentication, differential privacy, poisoning resistance, or protection against
a compromised client endpoint.

The Simulation Engine can establish functional and dropout behavior with virtual
clients, but it does not demonstrate independent administrative boundaries or
multi-host confidentiality. Deployment validation must use separate SuperNodes
with TLS and authentication. The base loopback profile has neither; the reviewed
secure overlay provides both without changing the secure-aggregation decision.

## Prerequisites and acceptance evidence

Before implementation:

- complete a real multi-client Flower round and the containerized end-to-end smoke
  test without SecAgg+;
- make required participation configurable and test disconnects, timeouts, and
  failed rounds;
- define the maximum sample weight and calibrate clipping and quantization against
  the baseline;
- decide whether secure aggregation or individual-update Huber/validation is the
  required security property; and
- add TLS and SuperNode authentication before any remote deployment claim.

Adoption requires end-to-end tests in both Simulation and Deployment Engines,
successful and threshold-breaking dropout cases, numerical parity within a stated
tolerance, artifact publication on success, cleanup on failure, and evidence that
ServerApp cannot observe an individual fit vector. The small API probe in this
repository verifies availability, not protocol validation.

## Differential privacy condition

No differential-privacy claim is made. The current update-noise ablation has no
defined adjacency relation, sensitivity model, privacy accountant, or composition
across rounds. Its per-tensor clipping and fixed Gaussian noise therefore do not
support an epsilon/delta value. Flower 1.32.1 provides preview DP mechanisms and
configuration helpers, but no complete privacy accounting for this repository's
mechanism, sampling, repeated rounds, and releases. Formal differential privacy
remains unimplemented unless a complete mechanism and accountant are added and
epsilon/delta are published.

## Official sources

- [Flower 1.32.1 secure-aggregation example](https://flower.ai/docs/examples/flower-secure-aggregation.html)
- [Flower secure-aggregation protocol explanation](https://flower.ai/docs/framework/1.32/en/explanation-ref-secure-aggregation-protocols.html)
- [`SecAggPlusWorkflow` 1.32 API](https://flower.ai/docs/framework/1.32/en/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html)
- [`secaggplus_mod` 1.32 API](https://flower.ai/docs/framework/1.32/en/ref-api/flwr.client.mod.secaggplus_mod.html)
- [Flower 1.32.1 server example source](https://github.com/adap/flower/blob/framework-1.32.1/examples/flower-secure-aggregation/secaggexample/server_app.py)
- [Flower 1.32.1 client example source](https://github.com/adap/flower/blob/framework-1.32.1/examples/flower-secure-aggregation/secaggexample/client_app.py)
- [Flower 1.32.1 `secaggplus_mod` source](https://github.com/adap/flower/blob/framework-1.32.1/framework/py/flwr/client/mod/secure_aggregation/secaggplus_mod.py)
- [Flower differential-privacy guide](https://flower.ai/docs/framework/1.32/en/how-to-use-differential-privacy.html)
