# Implementation checks

Validated on 2026-10-09 with Python 3.11.4, PyTorch 2.4.0, NumPy 1.26.4,
and scikit-learn 1.3.0. Tests use synthetic inputs, not clinical outcomes.

## Architecture compatibility

The extracted main path was compared locally with its source implementation:

- Identical parameter names, shapes, and seeded initialization.
- Identical response scores, probabilities, patient embeddings, and slide attention
  in both evaluation and training mode under matched random seeds (maximum difference 0).
- Ordinal loss and gradients agree within 2e−6 absolute tolerance.
- MPR-enriched pair subsampling and ranking loss agree under matched NumPy seeds.
- A local trained source checkpoint loads strictly; response scores and probabilities
  on synthetic inputs match exactly. No checkpoint is included in this repository.

Source file SHA256 fingerprints:

| Source component | SHA256 |
|---|---|
| Network implementation | `34646b314374e86a4973dd267c8b97df2773310954845bc09c31583e762d9a2c` |
| Training implementation | `6cb7467c0c4947ee080903da057f5bbb442bf2546e61df7b427920bcb30a4b2f` |

Compatibility checks concern the selected network and objective, not identical training
trajectories after changing the data partition. Cutpoint calibration explicitly uses
a consistent lexicographic tie-break key; it is tested as part of the five-fold workflow,
not claimed to reproduce every historical tied grid-search choice.

## Reproducible tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite verifies ordered cutpoints and probability sums, patch/slide attention
normalization, clinical coding and missingness, patient-disjoint stratified folds,
all-five-fold training and OOF coverage, train-only cutpoint calibration execution,
fixed-checkpoint external inference, rejection of internal/external overlap, and
refusal to overwrite existing fold outputs. Only temporary synthetic records are used.

This validates software execution and isolation rules, not clinical performance.
