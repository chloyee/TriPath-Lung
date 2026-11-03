# Training settings

The main entry point implements the patient-level development protocol in Supplementary Methods 1.1–1.2. `configs/main.json` records run parameters; `configs/selection.json` records bag selection.

| Setting | Main configuration | Basis |
|---|---|---|
| Outcome order | non-MPR, MPR-only, pCR | Manuscript |
| Development split | User-provided patient-level train / validation allocation | User dataset |
| Frozen CONCH visual features | 512 dimensions | Manuscript |
| Patch budget | 512 per slide; 60/20/20 selection | Manuscript |
| Matching temperature | 0.10 | Manuscript |
| Feature representation | 128 dimensions | Manuscript |
| Transformer depth | 1 | Manuscript |
| Transformer heads / FFN width | 4 / 256 | Source implementation |
| Transformer dropout / activation / normalization | 0.30 / GELU / pre-norm | Source implementation |
| Patch / slide dropout | 0.20 / 0.15 | Source implementation |
| Feature / gate / attention dropout | 0.25 / 0.30 / 0.30 | Source implementation |
| Clinical / interaction dropout | 0.40 / 0.40 | Source implementation |
| Clinical category embeddings | 8 dimensions | Manuscript |
| Clinical representation | 128 dimensions, two-layer MLP | Manuscript |
| Clinical / interaction coefficient initialization | −2.2 / −3.0 before sigmoid | Manuscript |
| Interaction output initialization | Zero | Manuscript |
| Ordinal cutpoint gap | softplus(gap) + 0.001 | Manuscript |
| First cutpoint / raw gap initialization | 0 / 0.5 | Source implementation |
| Optimizer | AdamW, LR 5e-5, weight decay 1e-3 | Manuscript |
| AdamW betas / epsilon | (0.9, 0.999) / 1e-8 | PyTorch optimizer defaults used by the source |
| Learning-rate schedule | Constant | Source implementation |
| Gradient clipping | Norm 5 | Manuscript |
| Epoch limit / patience / minimum improvement | 60 / 8 / 0.001 | Manuscript |
| Patient batch size / precision | 1 / float32 | Source implementation |
| Random seed | 55 | Source configuration |
| MPR-only sampling | Twice per epoch | Manuscript |
| Ordinal BCE reduction | Sum over two thresholds; mean over patients | Supplementary equation (11) |
| Class NLL / ranking weights | 0.5 / 0.05 | Manuscript |
| Ranking implementation | Softplus, current patient versus detached prior scores within the epoch | Source implementation |
| Maximum ranking pairs / MPR pair sampling factor | 64 / 4 | Source implementation |
| Binding weight / warmup / margins | 0.05 / 5 epochs / 0.01 and 0.01 | Manuscript |
| Validation objective | Ordinal BCE + 0.5 × class NLL | Source implementation |
| Main-model feature input | Direct projection of frozen features | Supplementary equations (5)–(6) |
| Prediction cutpoints | Learned jointly with the model | Supplementary equation (10) |

Class and threshold weights are computed from the training patients before oversampling. Ranking excludes same-label pairs and increases the selection probability of pairs involving MPR-only when sampling more than 64 candidate pairs. All four channel-binding passes preserve paired dropout masks.

The source implementation retains optional residual feature-adapter parameters for code compatibility. `use_feature_adapter` is false in the main configuration. `calibrate_cumulative` is false; the manuscript entry point uses the jointly learned cutpoints throughout checkpoint selection and prediction.

The split file identifies the user-selected training and validation patients; the seed governs training randomness within that allocation. Cohort size, center names and text descriptions are supplied by the user. The eight concept definitions, model architecture, selection rule and training objective follow the manuscript. Best-checkpoint selection uses a loss decrease greater than the configured minimum. Record the checkpoint epoch alongside the exported predictions.
