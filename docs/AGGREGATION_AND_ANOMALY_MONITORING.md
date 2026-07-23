# Aggregation, participation, and update monitoring

## Participation policy

The deployment server registers `expected-client-count` clients and requires all of
them to be online. `fit-participation-fraction` controls the fraction sampled by
Flower for training; `minimum-fit-client-count` sets the lower bound. When the
minimum is omitted, it is the ceiling of the fraction multiplied by the registered
client count. Evaluation still uses every registered client.

Every returned fit result must have a unique registered client ID, a positive sample
count, and the exact finite `float32` model shape. Any reported client failure fails
the round. Sampling therefore reduces participation, not validation.

## Registered aggregation alternatives

The experiment runner supports these existing alternatives:

- **FedAvg / FedProx:** sample-count-weighted arithmetic mean. This assumes sample
  counts are trustworthy and large updates should retain proportional influence.
- **FedProx + multidimensional Huber:** robust location estimate over flattened model
  vectors with sample-count base weights. Its threshold is scale-dependent, so it
  must be selected before comparison and interpreted in model-parameter units.
- **FedMedian:** coordinate-wise median. It limits coordinate outliers but can combine
  coordinates that never occurred together in one client model.
- **FedTrimmedAvg:** coordinate-wise trimmed mean. It assumes enough participating
  clients remain after symmetric trimming; its registered trim fraction is fixed by
  the scientific protocol.

Robust aggregation mitigates some outliers; it does not establish privacy, Byzantine
security, or correctness under coordinated model poisoning.

## Model-update anomaly report

For deployment rounds, the server computes each valid client's L2 distance from the
round-start global model. It reports clients above
`median + multiplier × 1.4826 × MAD`, where the multiplier is configured by
`update-anomaly-threshold-multiplier` and defaults to `3.0`.

The report is diagnostic only: flagged valid updates are still aggregated. Invalid
shapes, dtypes, or non-finite tensors remain contract violations and fail the round.
A zero MAD makes the threshold equal to the median; with very small client samples,
the statistic has limited detection power and must not be treated as proof of an
attack.
