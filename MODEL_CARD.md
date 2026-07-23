# Model Card: Federated IMDB Sentiment Classifier

## Model details

The project trains a binary English-language sentiment classifier with a
trainable embedding, one-dimensional convolution, global max pooling, dropout,
and a sigmoid output. The frozen architecture and training contract are defined
in [`docs/scientific-protocol-v1.toml`](docs/scientific-protocol-v1.toml).

Model artifacts are produced independently by centralized, local-only, and
federated strategies. No pretrained weights or endorsed release model are
distributed with this repository.

## Intended use

- Reproduce a local federated-learning demonstration.
- Compare registered aggregation strategies on the pinned IMDB dataset.
- Study experiment provenance, artifact compatibility, and robustness limits.

The model is not intended for moderation, reputation, employment, credit,
health, legal, or other decisions about people. It is not a general-purpose
sentiment model and should not be applied to languages or domains it was not
evaluated on.

## Evaluation

The registered evaluator reports accuracy, precision, recall, F1, ROC-AUC, and
a confusion matrix at a fixed `0.5` threshold. The official IMDB test split is
reserved as an untouched final test set.

Comparative performance values are not published yet. A generated model should
not be described as validated until the registered multi-seed IID/non-IID
matrix has been completed and its provenance artifacts have been reviewed.

## Limitations and risks

- Training shards are centrally prepared for a simulation; they do not
  represent independent data owners.
- IMDB reviews are an old, domain-specific proxy for binary sentiment.
- Binary labels cannot represent mixed, neutral, contextual, or sarcastic
  sentiment reliably.
- Model parameters, outputs, metrics, and the shared vocabulary can leak
  information about training data.
- The deployment demo has no TLS, component authentication, secure aggregation,
  formal differential privacy, or dashboard authentication.
- Robust aggregators reduce selected outlier effects under registered
  assumptions; they do not establish Byzantine security.

See [`THREAT_MODEL.md`](THREAT_MODEL.md) for the complete security and privacy
boundary.

## Reproducibility

Each completed run records its configuration, environment, code revision when
available, public-data identity, artifact schema version, and SHA-256 checksums.
The commands in [`README.md`](README.md) create new immutable output
directories rather than overwriting a selected model.
