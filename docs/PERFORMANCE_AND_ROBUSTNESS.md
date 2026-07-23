# Aggregation performance and robustness benchmark

`src.aggregation_benchmark` exercises the registered aggregation implementations
with deterministic synthetic `float32` model updates. It profiles 4, 16, and 64
simulated clients and applies the frozen outlier (`10 × update`) and sign-flip
(`-10 × update`) transforms to 25% and 50% of four-client updates.

```bash
uv run --env-file .env.protocol python -m src.aggregation_benchmark \
  --output results/aggregation-benchmark.json
```

The report includes raw aggregation durations, their mean, exact serialized Flower
parameter payload bytes for fit request/response tensors, and L2 error from the
honest FedAvg aggregate under each attack. The strategy runner serializes one
server request payload per round and multiplies its size by recipients; it serializes
each distinct client response once. This avoids redundant request serialization
without changing model precision or the frozen communication boundary.

## Published results

Mean aggregation time in milliseconds:

| Clients | FedAvg | Huber | Median | Trimmed mean |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.51 | 193.47 | 9.96 | 5.54 |
| 16 | 1.83 | 442.64 | 20.47 | 20.22 |
| 64 | 15.50 | 2,855.56 | 75.55 | 92.12 |

Aggregate L2 error from the honest FedAvg reference:

| Attack | Malicious clients | FedAvg | Huber | Median | Trimmed mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Outlier | 25% | 71.06 | **4.23** | 11.43 | 11.43 |
| Outlier | 50% | 100.47 | **13.02** | 63.68 | 63.68 |
| Sign flip | 25% | 86.85 | **17.46** | 21.59 | 21.59 |
| Sign flip | 50% | 122.79 | **32.79** | 82.34 | 82.34 |

Huber had the lowest error in every synthetic attack condition, but it was much
slower than the other methods. In the clean end-to-end sentiment matrix, FedProx
plus Huber did not improve test accuracy over FedAvg. Together, the experiments
show a robustness/performance trade-off rather than a universally superior
aggregator.

This is an aggregation microbenchmark, not a claim about 64-client training
accuracy, network throughput, Byzantine security, or production capacity.
Measurements depend on CPU, memory bandwidth, Python, NumPy, and Flower versions;
publish environment metadata beside any quoted timing.
