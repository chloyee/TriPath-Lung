# TriPath-Lung

TriPath-Lung integrates semantically selected regions from pretreatment H&E biopsies with clinical information to predict three ordered pathological response states: **non-MPR, MPR without pCR (MPR-only), and pCR**. Frozen CONCH features guide tissue selection, hierarchical attention combines regions and slides, and a shared ordinal head integrates tissue and clinical evidence.

PyTorch implementation of TriPath-Lung with semantic patch selection, multimodal training and patient-level evaluation.

## Installation

Requirements: Python 3.11+, PyTorch 2.4.0, NumPy 1.26.4 and scikit-learn 1.3.0. Training and testing run on CPU or a CUDA device. Feature extraction uses [CONCH](https://github.com/mahmoodlab/CONCH), installed separately.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For a Linux CUDA 12.1 environment, install the corresponding PyTorch wheel before the editable package:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

The trainer processes one patient per optimization step on the selected device. Use `--device cpu` for CPU execution or `--device cuda:0` for a CUDA device.

## Data preparation

Training requires:

1. A slide manifest containing patient IDs, response labels and clinical variables.
2. A JSON file listing training and validation patient IDs.
3. Frozen CONCH visual features, concept similarities, patch coordinates and text prototypes.

[Data preparation](docs/data_preparation.md) defines the file formats, clinical encoding and upstream image-processing requirements. [Training settings](docs/training_settings.md) records the architecture and optimization settings.

### Patient allocation

The split JSON has two keys, `train` and `validation`, each containing an array of patient ID strings matching the manifest. All slides of each patient follow that patient's allocation. The trainer checks unique IDs, disjoint partitions, complete manifest coverage and representation of all three outcomes in each partition before fitting clinical preprocessing.

`internal_center` optionally restricts the manifest to one center; `expected_class_counts` optionally validates the class counts in each partition. Both default to `null`. Center identifiers and partition sizes are read from the manifest and split file.

Text descriptions define eight tissue concepts. Encode these descriptions with CONCH to build the text prototypes and calculate patch–concept similarities. [Data preparation](docs/data_preparation.md) specifies the concept order and cache format.

### Semantic patch selection

For raw CONCH cosine similarities `c`, the patch-selection score is:

```text
support = (c_immune_inflamed + c_tls_like) / 2
        - (c_immune_excluded + c_immune_desert) / 2
        + 0.5 * c_viable_tumor - 0.5 * c_artifact
```

The selector retains up to 512 patches per slide: 60% by support, 20% by the entropy of `softmax(c / 0.10)` among unselected patches, and the remaining budget by random sampling without replacement. At 512 patches the quotas are 307, 102 and 103. Smaller bags retain all available patches. Ties preserve cache order; random selection uses a deterministic seed derived from the slide ID.

```bash
python -m vlr_mil.selection \
  --manifest data/development_slides.csv \
  --features-root features/conch_candidates \
  --output features/conch_selected \
  --config configs/selection.json
```

The output directory is created for this selection run. Original array order is preserved; the selector adds `selected`, `selection_group` and `response_support_score`. `selection_manifest.json` records counts, settings and file hashes. Raw float32 scores preserve extraction-time ranking precision. An existing verified selection mask can be supplied directly to training.

## Training

```bash
bash scripts/run_train.sh \
  --manifest data/development_slides.csv \
  --split data/patient_split.json \
  --features-root features/conch_selected \
  --concept-embeddings features/concept_embeddings.npz \
  --output outputs/tripath_main_seed55 \
  --device cuda:0
```

To run selection and training together, use the candidate cache for `--features-root` and add `--selection-output features/conch_selected_run1`. Each run uses a fresh output directory. `tripath-train` and `python -m vlr_mil.train` expose the same arguments. The internal Python package is named `vlr_mil`.

Clinical preprocessing is fitted on training patients. Internal-validation loss selects the checkpoint. The main configuration uses AdamW with learning rate `5e-5`, weight decay `1e-3`, gradient-norm clipping at `5`, up to `60` epochs and patience `8` with minimum loss improvement `0.001`.

## Evaluation

Prepare a labeled test manifest and selected feature caches using the same CONCH encoder and text prototypes as training. Assign stable patient identifiers across manifests so the test cohort is separate from both training and model-selection patients.

```bash
bash scripts/run_test.sh \
  --checkpoint outputs/tripath_main_seed55/main/best.pt \
  --manifest data/test_slides.csv \
  --features-root features/conch_test_selected \
  --output outputs/tripath_test \
  --device cuda:0
```

`tripath-test` and `python -m vlr_mil.evaluate` expose the same arguments. The test entry point restores the model, concept prototypes, ordinal cutpoints and clinical preprocessing directly from the checkpoint. Test labels are used to calculate evaluation metrics. Inference uses evaluation mode with fixed parameters and training-derived clinical normalization.

The output directory contains `test_predictions.csv` (patient-level labels, class probabilities and ordinal scores), `metrics.json` (AUROC, accuracy, F1, sensitivity, specificity and calibration metrics), and `test_provenance.json` (checkpoint/manifest hashes, selected epoch and preprocessing state). Each test run uses a fresh output directory.

## Model and objective

Visual features, the best-matching text prototype and eight concept percentile features are projected to 128 dimensions. An elementwise gate combines image and text representations using their projections and the maximum raw concept similarity. Concept and normalized positional features are added before a single Transformer layer. Three outcome-directed patch-attention channels form a slide representation; multi-channel attention then combines the patient's slides.

Histology, pretreatment clinical T category, age, PD-L1 TPS and smoking status form the clinical representation. The response score is:

```text
r = image_score + sigmoid(a) * clinical_score + sigmoid(b) * interaction_score
```

The coefficients start at `a = -2.2` and `b = -3.0`. Two jointly learned, ordered cutpoints yield:

```text
theta2 = theta1 + softplus(gap) + 0.001
q1 = sigmoid(r - theta1)       P(MPR-only or pCR)
q2 = sigmoid(r - theta2)       P(pCR)
p  = [1 - q1, q1 - q2, q2]
```

The objective combines weighted ordinal BCE summed over the two thresholds, `0.5 ×` class-weighted NLL, `0.05 ×` pairwise ranking, and outcome-directed channel binding. Patient losses are averaged for reporting. MPR-only patients are sampled twice per epoch. Binding is active from epoch 6 with weight `0.05` and margins `0.01`. [Channel binding](docs/channel_binding.md) describes the matched-mask ablation calculation and its training diagnostics.

## Outputs

Each run creates `input_provenance.json` and a `main/` directory containing:

| File | Contents |
|---|---|
| `best.pt` | Model parameters, learned cutpoints, clinical preprocessing, split, configuration and selected epoch |
| `split.json` | Patient allocation used in this run |
| `config.json` | Training settings |
| `history.jsonl` | Epoch losses, validation metrics and channel-binding diagnostics |
| `validation_predictions.csv` | Patient labels, three probabilities, predicted class and ordinal score |
| `metrics.json` | Patient-level discrimination, classification and probability metrics |

The checkpoint stores the same ordinal head used for validation predictions. Model selection monitors ordinal BCE plus weighted class NLL; ranking and binding supervise training. The selected epoch records which stages of training the checkpoint completed.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/check_secrets.py
```

Tests cover semantic selection, patient separation, training-only clinical preprocessing, the ordinal loss formula, raw-score gating, matched stochastic masks, auxiliary gradients, a six-epoch synthetic training run, held-out test execution, development/test separation and checkpoint-to-prediction consistency.

## License

Source code is distributed under the [MIT License](LICENSE). Foundation-model weights and clinical data follow their respective access and licensing terms.
