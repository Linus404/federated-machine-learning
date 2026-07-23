# Portfolio experiment matrix

Status: **complete**

This compact, preregistered portfolio campaign ran four requested strategies over
IID, moderate non-IID (`Dirichlet α=0.5`), and severe non-IID
(`Dirichlet α=0.1`) partitions with seeds 67, 101, and 211. Every cell used 20
epochs/rounds and the same untouched 25,000-review test set. Metric cells show mean
(sample variance).

| Strategy | Partition | Seeds | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| local_only | iid_stratified | 3 | 0.853397 (0.000001) | 0.844267 (0.000018) | 0.867113 (0.000058) | 0.855351 (0.000003) | 0.931307 (0.000001) |
| local_only | dirichlet_0.5 | 3 | 0.672383 (0.001691) | 0.682215 (0.014354) | 0.720813 (0.032451) | 0.636533 (0.020883) | 0.844667 (0.001779) |
| local_only | dirichlet_0.1 | 3 | 0.566103 (0.006228) | 0.389784 (0.000196) | 0.642207 (0.033694) | 0.476352 (0.003717) | 0.607303 (0.008762) |
| fedavg | iid_stratified | 3 | 0.887573 (0.000001) | 0.888402 (0.000044) | 0.886613 (0.000102) | 0.887455 (0.000003) | 0.955738 (0.000000) |
| fedavg | dirichlet_0.5 | 3 | 0.863933 (0.000425) | 0.875461 (0.001009) | 0.852293 (0.007451) | 0.860955 (0.000972) | 0.944776 (0.000038) |
| fedavg | dirichlet_0.1 | 3 | 0.709360 (0.014169) | 0.858197 (0.013192) | 0.489307 (0.046511) | 0.611816 (0.036771) | 0.838768 (0.019430) |
| fedprox | iid_stratified | 3 | 0.822280 (0.000237) | 0.818076 (0.000161) | 0.828960 (0.000787) | 0.823336 (0.000286) | 0.904823 (0.000194) |
| fedprox | dirichlet_0.5 | 3 | 0.720360 (0.006207) | 0.738545 (0.002621) | 0.704827 (0.075874) | 0.695118 (0.023888) | 0.817915 (0.004635) |
| fedprox | dirichlet_0.1 | 3 | 0.570187 (0.008122) | 0.676067 (0.025183) | 0.374907 (0.124141) | 0.389509 (0.112436) | 0.671307 (0.019802) |
| fedprox_huber | iid_stratified | 3 | 0.822520 (0.000022) | 0.820401 (0.000131) | 0.826080 (0.000035) | 0.823169 (0.000008) | 0.905987 (0.000022) |
| fedprox_huber | dirichlet_0.5 | 3 | 0.708987 (0.011094) | 0.754719 (0.002164) | 0.638907 (0.110167) | 0.649639 (0.052194) | 0.806021 (0.006971) |
| fedprox_huber | dirichlet_0.1 | 3 | 0.586133 (0.009837) | 0.716525 (0.027485) | 0.421200 (0.123794) | 0.438482 (0.095811) | 0.701573 (0.018052) |

## Main findings

- **FedAvg was strongest in every partition.** It reached 88.76% mean accuracy on
  IID data and retained 86.39% at `α=0.5`.
- **Heterogeneity was the dominant failure mode.** At `α=0.1`, FedAvg fell to
  70.94% accuracy and 61.18% F1; local-only fell to 56.61% accuracy.
- **The registered FedProx setting did not beat FedAvg.** Huber aggregation also
  did not improve benign test accuracy, so the project reports that negative result
  rather than tuning on the test set.
- Every federated cell exchanged **1.294 GB** of serialized Flower parameter
  payloads over 20 rounds. Mean summed client training time was about 450–458 s for
  FedAvg and 614–711 s for the proximal strategies on this campaign host.
- The separate controlled-update benchmark shows why robust aggregation remains
  useful under attack: with one outlier client, aggregate L2 error was 71.06 for
  FedAvg, 4.23 for Huber, and 11.43 for median/trimmed mean. This synthetic result
  is not presented as sentiment-model accuracy.

## Paired descriptive comparisons

Each difference below pairs the same three seeds before averaging. Positive
values favor FedAvg; values are accuracy percentage points.

| Partition | FedAvg − local only | FedAvg − FedProx | FedAvg − FedProx + Huber |
| --- | ---: | ---: | ---: |
| IID | +3.42 | +6.53 | +6.51 |
| Dirichlet `α=0.5` | +19.16 | +14.36 | +15.49 |
| Dirichlet `α=0.1` | +14.33 | +13.92 | +12.32 |

FedAvg exceeded every comparator's accuracy in each of the nine paired
seed-and-partition comparisons. The severe non-IID F1 differences were less
stable: one seed favored local-only and both proximal strategies over FedAvg.
Three seeds are enough for descriptive paired differences, not a precise
confidence interval.

The heterogeneity penalty below compares each non-IID cell with the same
strategy and seed under IID.

| Strategy | `α=0.5` − IID | `α=0.1` − IID |
| --- | ---: | ---: |
| Local only | −18.10 | −28.73 |
| FedAvg | −2.36 | −17.82 |
| FedProx | −10.19 | −25.21 |
| FedProx + Huber | −11.35 | −23.64 |

These comparisons support the narrow conclusion that FedAvg tolerated moderate
heterogeneity best in this campaign. They do not establish a general ranking of
federated optimizers or robust aggregators.

## Reproduce

Prepare the pinned IMDB artifacts, then run:

```bash
uv run --env-file .env.protocol python -m src.experiment_matrix \
  --output-dir artifacts/portfolio-results \
  --strategies local_only fedavg fedprox fedprox_huber \
  --partitions iid_stratified dirichlet_0.5 dirichlet_0.1 \
  --seeds 67 101 211
```

The published JSON contains every per-seed metric, confusion matrix, system metric,
`results.json` checksum, and `provenance.json` checksum. All 36 source cells were
clean executions of commit `e903f196f14f7ffdd7b75e1871de702fd1a24397`; all
provenance sidecars were rehashed before publication.

## Interpretation

This is a compact 36-cell subset, not the frozen protocol's maximum 290-cell
campaign. Three seeds support a mean and sample variance but not a strong
population-level claim. Communication counts cover serialized Flower Parameters
protobufs and exclude metadata, TLS, and transport framing. Full machine-readable
evidence is in [`portfolio-matrix.json`](portfolio-matrix.json).
The project-wide experiment inventory and recommended extensions are in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md).
