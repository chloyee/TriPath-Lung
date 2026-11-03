# Training settings

`configs/main.json` controls training and `configs/selection.json` controls semantic patch selection. The tables below summarize the model architecture and default training settings.

| Setting | Main configuration |
|---|---|
| Outcome order | non-MPR, MPR-only, pCR |
| Development split | Patient-level train / validation lists in split JSON |
| Frozen CONCH visual features | 512 dimensions |
| Patch budget | 512 per slide; 60/20/20 selection |
| Matching temperature | 0.10 |
| Feature representation | 128 dimensions |
| Transformer depth | 1 |
| Transformer heads / FFN width | 4 / 256 |
| Transformer dropout / activation / normalization | 0.30 / GELU / pre-norm |
| Patch / slide dropout | 0.20 / 0.15 |
| Feature / gate / attention dropout | 0.25 / 0.30 / 0.30 |
| Clinical / interaction dropout | 0.40 / 0.40 |
| Clinical category embeddings | 8 dimensions |
| Clinical representation | 128 dimensions, two-layer MLP |
| Clinical / interaction coefficient initialization | −2.2 / −3.0 before sigmoid |
| Interaction output initialization | Zero |
| Ordinal cutpoint gap | softplus(gap) + 0.001 |
| First cutpoint / raw gap initialization | 0 / 0.5 |
| Optimizer | AdamW, LR 5e-5, weight decay 1e-3 |
| AdamW betas / epsilon | (0.9, 0.999) / 1e-8 |
| Learning-rate schedule | Constant |
| Gradient clipping | Norm 5 |
| Epoch limit / patience / minimum improvement | 60 / 8 / 0.001 |
| Patient batch size / precision | 1 / float32 |
| Random seed | 55 |
| MPR-only sampling | Twice per epoch |
| Ordinal BCE reduction | Sum over two thresholds; mean over patients |
| Class NLL / ranking weights | 0.5 / 0.05 |
| Ranking implementation | Softplus, current patient versus detached prior scores within the epoch |
| Maximum ranking pairs / MPR pair sampling factor | 64 / 4 |
| Binding weight / warmup / margins | 0.05 / 5 epochs / 0.01 and 0.01 |
| Validation objective | Ordinal BCE + 0.5 × class NLL |
| Main-model feature input | Direct projection of frozen features |
| Prediction cutpoints | Learned jointly with the model |

Class and threshold weights are computed from the training patients before oversampling. Ranking excludes same-label pairs and increases the selection probability of pairs involving MPR-only when sampling more than 64 candidate pairs. All four channel-binding passes preserve paired dropout masks.

## Configuration and model selection

`use_feature_adapter=false` applies the visual projection directly to frozen features. `calibrate_cumulative=false` uses the jointly learned ordinal cutpoints for validation and testing.

The split JSON assigns patients to training and validation. Clinical normalization and loss weights are fitted on training patients. The random seed controls initialization, sampling and dropout. The validation objective selects the checkpoint when its loss decreases by more than `min_delta`; `patience` controls early stopping. The checkpoint includes the selected epoch, model parameters, text prototypes and clinical preprocessing state.

## Evaluation

Evaluation restores the checkpoint and processes the test manifest with fixed model parameters and preprocessing statistics. Predictions contain three class probabilities and an ordinal score for each patient. See [Training and evaluation](../README.md) for commands and [Data preparation](data_preparation.md) for input formats.
