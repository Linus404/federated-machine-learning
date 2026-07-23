# Experiment Catalogue

This catalogue separates published scientific evidence from engineering
validation and registered work that has not been executed. The frozen protocol
defines the admissible configurations; checked-in result files define what was
actually run.

## Published experiments

| Experiment | Design | Replication | Evidence | Main result |
| --- | --- | --- | --- | --- |
| Sentiment strategy and heterogeneity matrix | Local only, FedAvg, FedProx, and FedProx + Huber across IID, Dirichlet `α=0.5`, and Dirichlet `α=0.1` partitions | Seeds 67, 101, and 211; 36 cells; 20 epochs/rounds per cell | [`results/portfolio-matrix.md`](results/portfolio-matrix.md), [`results/portfolio-matrix.json`](results/portfolio-matrix.json) | FedAvg had the highest mean accuracy in every partition; severe heterogeneity reduced every strategy's accuracy and increased seed sensitivity. |
| Aggregation scaling microbenchmark | FedAvg, Huber, coordinate median, and trimmed mean over deterministic 100,000-dimensional updates from 4, 16, and 64 clients | Five timing repetitions per strategy and client count | [`docs/PERFORMANCE_AND_ROBUSTNESS.md`](docs/PERFORMANCE_AND_ROBUSTNESS.md), [`results/aggregation-benchmark.json`](results/aggregation-benchmark.json) | FedAvg was fastest; Huber was substantially slower, particularly at 64 clients. |
| Controlled-update robustness microbenchmark | Outlier and sign-flip transforms applied to 25% and 50% of four-client synthetic updates | Seed 67; one deterministic update set per attack condition | [`docs/PERFORMANCE_AND_ROBUSTNESS.md`](docs/PERFORMANCE_AND_ROBUSTNESS.md), [`results/aggregation-benchmark.json`](results/aggregation-benchmark.json) | Huber produced the lowest aggregate L2 error in all four attack conditions. |

The training matrix uses one pinned dataset revision, one frozen model and
preprocessing contract, the same untouched 25,000-review test set, and paired
seeds across strategies. Results are therefore directly comparable within the
published matrix. They are not estimates for other datasets, client populations,
architectures, or attack models.

## Engineering validation

The following checks validate implementation paths but are not comparative
scientific experiments:

- A two-client Flower end-to-end test completes a real training round with the
  application ClientApp and ServerApp code.
- The Compose smoke test starts four clients and all service roles, checks
  health, and completes one Flower round.
- Privacy evaluators are covered by exact golden vectors and input-validation
  tests. No membership-inference or model-update leakage result is published;
  the required candidate-level evidence was not retained by the 36-cell
  campaign.
- Corruption, compatibility, failure, checkpoint, and resume tests establish
  software behavior, not model quality.

## Registered protocol coverage

The frozen protocol defines a maximum 290-cell research program. The published
portfolio subset covers 36 training cells.

| Registered component | Published coverage |
| --- | --- |
| Primary four-client federated matrix | FedAvg, FedProx, and FedProx + Huber on three of four partitions and three of five seeds |
| Local-only matrix | Three of four partitions and three of five seeds |
| Centralized baseline | Training path implemented; no checked-in comparative result |
| FedMedian and FedTrimmedAvg training | Runner implemented; no checked-in end-to-end result |
| 16- and 64-client training | Aggregation microbenchmark only; no end-to-end accuracy result |
| End-to-end malicious-client matrix | Synthetic aggregation benchmark only |
| Membership inference and update leakage | Evaluators implemented; no checked-in attack result |

The scope and implementation-status strings inside
[`docs/scientific-protocol-v1.toml`](docs/scientific-protocol-v1.toml) record the
preregistration state when protocol version 1 was frozen. They remain unchanged
because every published cell records the protocol file's SHA-256 checksum.

## Recommended academic extensions

These extensions are ranked by the additional evidence they provide, not by
production value.

| Priority | Extension | Additional scope | Research question |
| ---: | --- | ---: | --- |
| 1 | Complete the five-seed version of the published matrix | 24 training cells for seeds 307 and 401 | Are the observed rankings stable enough for paired uncertainty estimates? |
| 2 | Add the centralized baseline and Dirichlet `α=1.0` | 5 centralized cells and 20 `α=1.0` cells after priority 1 | What is the cost of federation, and how smoothly does quality change with heterogeneity? |
| 3 | Add benign FedMedian and FedTrimmedAvg cells | 40 training cells over four partitions and five seeds | Do robust aggregators change clean-data quality without the FedProx objective as a confounder? |
| 4 | Execute the registered end-to-end robustness matrix | 80 cells: four strategies × two attacks × two malicious fractions × five seeds | Do synthetic aggregate-error gains transfer to test accuracy and F1? |
| 5 | Execute the registered 16- and 64-client scale matrix | Up to 70 cells | How do client count, communication, convergence, and accuracy interact? |
| 6 | Produce the registered privacy evidence | 10 membership and 5 update-leakage analyses, reusing matching trained models | How effective are the specified attacks on retained candidate-level evidence? |

Priority 1 is the smallest defensible addition. Priorities 2 and 3 complete the
clean-data academic comparison. The larger attack, scale, and privacy campaigns
are valuable follow-up studies but are not required to support the current
project's limited claims.
