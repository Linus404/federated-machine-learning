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

This is an aggregation microbenchmark, not a claim about 64-client training
accuracy, network throughput, Byzantine security, or production capacity.
Measurements depend on CPU, memory bandwidth, Python, NumPy, and Flower versions;
publish environment metadata beside any quoted timing.
