# Dataset Card: Pinned IMDB Reviews

## Dataset details

The project uses `stanfordnlp/imdb` through Hugging Face Datasets at immutable
revision:

```text
e6281661ce1c48d982bc483cf8a173c1bbeb5d31
```

The dataset contains 25,000 labeled training reviews, 25,000 labeled test
reviews, and 50,000 unlabeled reviews. Labels are balanced between negative and
positive in both labeled splits. The unlabeled split is excluded.

The repository does not redistribute the review text. Users download the data
from the upstream source and are responsible for following its terms.

## Project use

- The official training split supplies fitted training rows and deterministic
  validation rows.
- The official test split is an untouched global test set used only for final
  registered evaluation and registered privacy analyses.
- A vocabulary of at most 20,000 tokens is adapted once on the complete
  training split and then frozen for all clients and strategies.
- The local demonstration centrally partitions training rows into simulated
  client shards. This is not evidence of independently owned or privately
  collected client data.

Exact split identities, row counts, SHA-256 checksums, preprocessing rules,
partition algorithms, and seeds are frozen in
[`docs/scientific-protocol-v1.toml`](docs/scientific-protocol-v1.toml).

## Sensitive information and responsible use

Reviews may contain names, quoted text, or other personal information supplied
by their original authors. Do not publish prepared shards or raw examples from
local artifacts. Do not treat a sentiment label as a fact about an author or
use the dataset to make decisions about people.

## Limitations

- The reviews cover one entertainment domain and are not representative of
  general language.
- The binary labels omit neutral and mixed sentiment.
- Review age, collection methods, and platform demographics can introduce
  temporal and population bias.
- Artificial IID and Dirichlet partitions model statistical heterogeneity, not
  real institutional, geographic, or demographic client boundaries.
- A vocabulary adapted on the full training split is public to every simulated
  participant and can reveal corpus-level properties.
