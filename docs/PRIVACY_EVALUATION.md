# Empirical privacy evaluation

`src.privacy_evaluation` implements the frozen protocol's two empirical attacks:

- **Black-box membership inference** scores each supplied candidate with the negative per-example binary-cross-entropy loss. A larger score means the trained model fits that labeled record more confidently.
- **Model-update leakage** scores each supplied candidate's exact negative loss gradient by cosine similarity with one target client's post-fit-minus-pre-fit update. A larger score means greater alignment with that update.

Both commands report ROC-AUC, maximum `TPR - FPR`, and interpolated TPR at 1% FPR using the frozen tie rules. Inputs are immutable `.npy` files with exact protocol dtypes; outputs are new JSON files and are never overwritten.

```bash
uv run python -m src.privacy_evaluation membership \
  --class-labels class-labels.npy \
  --probabilities candidate-probabilities.npy \
  --membership-labels membership-labels.npy \
  --output membership-results.json

uv run python -m src.privacy_evaluation update-leakage \
  --negative-loss-gradients negative-loss-gradients.npy \
  --client-update round-0-client-update.npy \
  --membership-labels membership-labels.npy \
  --output update-leakage-results.json
```

## Required upstream evidence

The evaluator deliberately does not infer private evidence from a final model. The experiment runner must first produce the protocol-selected balanced candidate records and their membership labels. Membership evaluation additionally requires sigmoid probabilities for those candidates. Update evaluation requires the selected client's individual round-0 update and the exact per-candidate negative gradients from the matching pre-update model, flattened in trainable-variable order. A final aggregate model or final test predictions cannot reconstruct these artifacts.

## Interpretation limits

These metrics measure the success of the specified attacks on the supplied candidate sample. They do **not** prove that a record was used for training, establish causality, bound every possible attack, measure legal identifiability, or provide a differential-privacy guarantee. Results are comparable only when candidate construction, model checkpoint, client update, seed, and protocol version match. An ROC-AUC near 0.5 means this attack did not rank the balanced sample better than chance; it does not prove absence of leakage. TPR at 1% FPR is interpolated and is not necessarily an attainable operating threshold.
