# VLR-MIL

Concept-guided, image–clinical multiple-instance learning for ordered pathological
response prediction from pretreatment H&E biopsies and baseline clinical variables.

VLR-MIL combines frozen visual features, tissue concepts, spatial context, and
baseline clinical information through hierarchical multiple-instance learning.
An ordinal prediction head models Non-MPR, MPR-only, and pCR as ordered outcomes.

## Environment

Tested with Python 3.11, PyTorch 2.4.0, NumPy 1.26.4, and scikit-learn 1.3.0.
CPU training is supported; a CUDA-capable GPU accelerates Transformer processing.
Create a dedicated environment in this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Training consumes cached CONCH patch features and matching concept embeddings.

## Architecture

```text
Frozen patch features ── Residual feature adapter ─────────────┐
Raw concept scores ──── Best-matching concept text embedding ─┤
                                                             └─ Gated image–text fusion
Concept percentiles + normalized XY coordinates ─────────────────────┤
                                                                    v
                                               Spatial Transformer (128 channels)
                                                                    v
                                          Gated patch / chunk / slide MIL pooling
                                                                    v
                                                       Image response score
Baseline clinical variables ── Clinical MLP ── Small additive clinical residual
Image × clinical features ── Interaction MLP ── Small additive interaction residual
                                                                    v
                                                   Shared ordinal response score
                                                                    v
                                          Two strictly ordered cumulative cutpoints
                                                                    v
                                                  Non-MPR / MPR-only / pCR
```

The feature adapter has a 64-dimensional bottleneck and an initially zero residual.
Each selected patch is matched to the text embedding with the highest **raw** concept
score. A learned, feature-wise gate combines visual and matched text representations.
The numeric concept branch uses stable within-bag percentile ranks; tied scores retain
source-order ranks rather than averaged ranks. Spatial coordinates are centered and
scaled per bag. A one-layer, four-head Transformer supplies spatial context.

Three gated attention channels pool patches and slides; each channel is normalized
over its instances. Their concatenation is projected before ordinal prediction, so
the channels are not independently supervised class probabilities. With the 512-patch
budget, each slide has one chunk; a retained chunk pool supports the model interface.

Clinical fusion is image-dominant:

`s = s_image + sigmoid(a) * s_clinical + sigmoid(b) * mean(interaction_logits)`.

The scalar coefficients start at sigmoid(−2.2) and sigmoid(−3.0). They are learned
global scales, not patient-specific modality importance scores. Higher `s` favors pCR.
With `c2 = c1 + softplus(gap) + 0.001`, the cumulative probabilities satisfy:

```text
q1 = sigmoid(s − c1)       P(MPR-only or pCR)
q2 = sigmoid(s − c2)       P(pCR)
p  = [1 − q1, q1 − q2, q2]
```

## Data preparation

Supply one CSV row per slide. Multiple slides of a patient must have the same
clinical fields and outcome. Patient identifiers must be stable across manifests.

| Field | Encoding |
|---|---|
| `patient_id`, `slide_id` | Deidentified stable identifiers |
| `center` | Development cohort identifier; default `GDPH` |
| `label` | `0` Non-MPR, `1` MPR-only, `2` pCR |
| `pathology_type` | `LUAD`, `LUSC`, `Other`; blank for unknown |
| `T_stage` | Baseline T0–T4; suffixes such as T2a collapse to T2 |
| `age` | Age in years; blank for missing |
| `pdl1_tps` | Fraction in [0,1]; 0.01 means 1%; blank for missing |
| `pdl1_unit` | Must be `fraction`, including rows with missing TPS |
| `smoking_status` | `0` never; `1` former >10 years; `2` current or cessation within 10 years; blank unknown |
| `eligible_for_modeling` | Optional; only `1` rows are included when present |

PD-L1 values are converted to a consistent fraction representation before training:
1% corresponds to 0.01 and 100% to 1. A source value `<1%` is represented as 0.005
under the interval-midpoint convention. Missing observations remain blank.

Age and `log1p(100 * TPS)` are standardized using observed **training-fold** patients.
Missing continuous inputs become zero after standardization; missing categories use
index zero. Five missingness indicators distinguish missing values from observed zeros.
TPS ≥1% and TPS ≥50% supply two additional binary inputs. Histology, T stage and
smoking use category embeddings.

Cached files must have the layout `features/<center>/<slide_id>.npz`:

| Key | Shape / meaning |
|---|---|
| `features` | N × 512 frozen patch features |
| `concept_scores` | N × 8 raw concept scores, before visualization normalization |
| `level0_xy` | N × 2 cached spatial coordinates, in a consistent slide coordinate system |
| `selected` | N boolean mask; 1–512 selected patches |
| `concept_names` | Optional ordered string array, checked when present |

All arrays use the same patch order. The `selected` mask specifies the patches in
each slide bag and remains fixed during validation. Training applies patch dropout
to this bag as a regularizer.
Eight concepts, in order: viable tumor, artifact, immune-inflamed, TLS-like,
immune-excluded, immune-desert, necrosis, fibrotic stroma. The concept embedding NPZ
must contain `embeddings` of shape 8 × 512, in the same order and visual–text space.
The feature cache and embedding bank use the same CONCH version, prompt definitions,
and concept ordering. Feature extraction and concept scoring precede MIL training.

## Five-fold main experiment

```bash
bash scripts/run_fivefold.sh \
  --manifest data/GDPH_internal_slides.csv \
  --features-root features/conch_v1 \
  --concept-embeddings features/conch_v1/concept_embeddings.npz \
  --output outputs/main_seed55 \
  --device cuda:0
```

Use `--device cpu --threads 4` for CPU execution. Add `--fold 0` to run one fold;
omit `--fold` for all five folds. To change the seed or training hyperparameters,
edit a copy of [configs/main.json](configs/main.json) and pass `--config`.
Each run uses a fresh output directory.

Patients are sorted and stratified by outcome with `StratifiedKFold(5, shuffle=True,
random_state=55)`. All slides from a patient remain together. Training accepts only
the configured development center. Each fold uses four patient partitions for
optimization and clinical preprocessing, and the remaining partition for validation
and early stopping. This is validation-based five-fold cross-validation rather than
nested cross-validation.

| Setting | Default |
|---|---|
| Optimizer | AdamW, LR 5e−5, weight decay 1e−3 |
| Epochs / early stopping | 60 / patience 8, validation loss, minimum improvement 0.001 |
| Patch / slide dropout | 0.20 / 0.15 |
| Adapter / Transformer dropout | 0.10 / 0.30 |
| Clinical dropout | 0.40 |
| Patient batch size | 1 |
| MPR patient sampling | 2× per epoch |
| Ranking loss weight / maximum pairs | 0.05 / 64 |
| MPR pair sampling factor | 4× when subsampling more than 64 candidate pairs |
| Gradient clipping | Norm 5 |

The loss combines positive-weighted cumulative binary losses, 0.5× class-weighted
categorical NLL, and 0.05× pairwise softplus ranking. All class weights are estimated
from the training fold. Ranking compares the current patient against detached scores
from earlier training patients in the same epoch; same-label pairs are excluded.
MPR pair enrichment applies during pair subsampling, not as an unconditional 4× loss multiplier.

After checkpoint selection, optional cutpoint calibration uses **training-fold** scores
and labels only, maximizing macro-F1, then balanced accuracy, MPR recall and QWK.
Set `calibrate_cumulative` to false to use only the cutpoints learned by gradient
descent. The selected cutpoints and clinical preprocessing parameters are saved
with the model checkpoint.

## Training outputs

Each fold stores `best.pt`, `split.json`, `config.json`, `history.jsonl`,
`validation_predictions.csv` and `metrics.json`. Running all five folds also produces
one prediction per internal patient in `oof_predictions.csv` and pooled OOF metrics.
Input hashes and code hashes are recorded in `input_provenance.json`.

Metrics include macro-AUROC, ACC, macro-F1, balanced accuracy, QWK, Brier, ECE,
NLL, and per-class AUROC, OVR-ACC, precision, sensitivity, specificity and F1.
Each prediction row contains the patient identifier, true and predicted classes,
three class probabilities, and the shared ordinal score.

## Repository layout

```text
configs/       Main training hyperparameters
src/vlr_mil/   Network, losses, clinical preprocessing, data loading and training
scripts/       One-command five-fold launcher
```

## License

Source code is distributed under the [MIT License](LICENSE). Upstream foundation
models and datasets are subject to their respective licenses and access requirements.
