# Portfolio experiment matrix

Status: **complete**

This preregistered academic campaign ran four requested strategies over
IID, moderate non-IID (`Dirichlet α=0.5`), and severe non-IID
(`Dirichlet α=0.1`) partitions with seeds 67, 101, 211, 307, and 401. Every cell used 20
epochs/rounds and the same untouched 25,000-review test set. Metric cells show mean
(sample variance).

| Strategy | Partition | Seeds | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| local_only | iid_stratified | 5 | 0.852748 (0.000003) | 0.846030 (0.000015) | 0.863104 (0.000072) | 0.854197 (0.000007) | 0.930961 (0.000001) |
| local_only | dirichlet_0.5 | 5 | 0.675160 (0.002560) | 0.683353 (0.024100) | 0.687144 (0.019186) | 0.614267 (0.011742) | 0.839731 (0.001775) |
| local_only | dirichlet_0.1 | 5 | 0.576584 (0.003322) | 0.394397 (0.002619) | 0.624524 (0.024429) | 0.473037 (0.005429) | 0.620527 (0.004713) |
| fedavg | iid_stratified | 5 | 0.886984 (0.000001) | 0.887850 (0.000050) | 0.886000 (0.000088) | 0.886864 (0.000002) | 0.955399 (0.000000) |
| fedavg | dirichlet_0.5 | 5 | 0.863480 (0.000267) | 0.883731 (0.000642) | 0.839744 (0.004425) | 0.859192 (0.000583) | 0.943952 (0.000028) |
| fedavg | dirichlet_0.1 | 5 | 0.773784 (0.014868) | 0.863636 (0.007028) | 0.642176 (0.067920) | 0.715204 (0.038451) | 0.881375 (0.013119) |
| fedprox | iid_stratified | 5 | 0.820904 (0.000175) | 0.817334 (0.000190) | 0.826704 (0.000407) | 0.821887 (0.000187) | 0.903915 (0.000143) |
| fedprox | dirichlet_0.5 | 5 | 0.718320 (0.005656) | 0.749959 (0.002134) | 0.665936 (0.052767) | 0.685284 (0.018263) | 0.813282 (0.004658) |
| fedprox | dirichlet_0.1 | 5 | 0.664248 (0.020728) | 0.723012 (0.018081) | 0.560416 (0.128457) | 0.558316 (0.109648) | 0.761187 (0.025086) |
| fedprox_huber | iid_stratified | 5 | 0.821744 (0.000037) | 0.814147 (0.000206) | 0.834384 (0.000148) | 0.823988 (0.000020) | 0.905129 (0.000031) |
| fedprox_huber | dirichlet_0.5 | 5 | 0.720088 (0.006952) | 0.755064 (0.002493) | 0.663424 (0.057311) | 0.680517 (0.029128) | 0.811290 (0.005093) |
| fedprox_huber | dirichlet_0.1 | 5 | 0.665968 (0.016873) | 0.752247 (0.019198) | 0.566464 (0.108716) | 0.576071 (0.083691) | 0.778233 (0.020108) |

## Main findings

- **FedAvg was strongest in every partition.** It reached 88.70% mean accuracy on
  IID data and retained 86.35% at `α=0.5`.
- **Heterogeneity was the dominant failure mode.** At `α=0.1`, FedAvg fell to
  77.38% accuracy and 71.52% F1; local-only fell to 57.66% accuracy.
- **The registered FedProx setting did not beat FedAvg.** Huber aggregation also
  did not improve benign test accuracy, so the project reports that negative result
  rather than tuning on the test set.
- Every federated cell exchanged **1.294 GB** of serialized Flower parameter
  payloads over 20 rounds. Training times are retained per cell, but the two added
  seeds ran concurrently and are not used for cross-strategy timing claims.
- The separate controlled-update benchmark shows why robust aggregation remains
  useful under attack: with one outlier client, aggregate L2 error was 71.06 for
  FedAvg, 4.23 for Huber, and 11.43 for median/trimmed mean. This synthetic result
  is not presented as sentiment-model accuracy.

## Paired descriptive comparisons

Each difference below pairs the same five seeds before averaging. Positive
values favor FedAvg; brackets contain two-sided 95% Student-t confidence
intervals in accuracy percentage points.

| Partition | FedAvg − local only | FedAvg − FedProx | FedAvg − FedProx + Huber |
| --- | ---: | ---: | ---: |
| IID | +3.42 [+3.25, +3.60] | +6.61 [+5.00, +8.22] | +6.52 [+5.77, +7.28] |
| Dirichlet `α=0.5` | +18.83 [+14.03, +23.64] | +14.52 [+6.93, +22.11] | +14.34 [+5.86, +22.82] |
| Dirichlet `α=0.1` | +19.72 [+9.80, +29.64] | +10.95 [+2.67, +19.24] | +10.78 [+4.30, +17.26] |

FedAvg exceeded every comparator's accuracy in all 15 paired
seed-and-partition comparisons. Every confidence interval above excludes zero.

The heterogeneity penalty below compares each non-IID cell with the same
strategy and seed under IID.

| Strategy | `α=0.5` − IID | `α=0.1` − IID |
| --- | ---: | ---: |
| Local only | −17.76 [−24.19, −11.32] | −27.62 [−34.81, −20.42] |
| FedAvg | −2.35 [−4.35, −0.35] | −11.32 [−26.58, +3.94] |
| FedProx | −10.26 [−19.37, −1.14] | −15.67 [−34.36, +3.03] |
| FedProx + Huber | −10.17 [−20.42, +0.09] | −15.58 [−31.81, +0.65] |

These comparisons support the narrow conclusion that FedAvg tolerated moderate
heterogeneity best in this campaign. Severe-partition intervals remain wide and
often include zero; the five-seed extension does not remove that uncertainty.

## Reproduce

Prepare the pinned IMDB artifacts, then run:

```bash
uv run --env-file .env.protocol python -m src.experiment_matrix \
  --output-dir artifacts/portfolio-results \
  --strategies local_only fedavg fedprox fedprox_huber \
  --partitions iid_stratified dirichlet_0.5 dirichlet_0.1 \
  --seeds 67 101 211 307 401
```

The published JSON contains every per-seed metric, confusion matrix, system metric,
`results.json` checksum, and `provenance.json` checksum. All 60 source cells were
clean executions of commit `e903f196f14f7ffdd7b75e1871de702fd1a24397`; all
provenance sidecars were rehashed before publication.

## Interpretation

This is a 60-cell subset, not the frozen protocol's maximum 290-cell
campaign. Five seeds support the registered paired confidence intervals but not
a broad population-level claim. Communication counts cover serialized Flower Parameters
protobufs and exclude metadata, TLS, and transport framing. Full machine-readable
evidence is in [`portfolio-matrix.json`](portfolio-matrix.json).
The project-wide experiment inventory and recommended extensions are in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md).
