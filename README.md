# TriPath-Lung

TriPath-Lung integrates semantically selected regions from pretreatment H&E biopsies with clinical information to predict three ordered pathological response states: **non-MPR, MPR without pCR (MPR-only), and pCR**. Frozen CONCH features guide tissue selection, hierarchical attention combines regions and slides, and a shared ordinal head integrates tissue and clinical evidence.

This repository implements the manuscript's main-model training logic for user-provided datasets and patient allocations. The workflow starts from cached CONCH features: semantic bag selection, patient-level data loading, multimodal ordinal modeling, internal validation, and checkpoint export.

## Environment

The manuscript reports Python 3.11.4, PyTorch 2.4.0, torchvision 0.19.0 and CUDA 12.1 on two NVIDIA RTX 4090 GPUs. Cached-feature training requires NumPy, scikit-learn and PyTorch; torchvision and CONCH are used upstream when preparing image features.

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

## Prepare the training inputs

Provide these three resources:

1. A deidentified slide manifest for your development cohort.
2. Your patient-level training and validation allocation.
3. Cached frozen CONCH visual features, raw concept similarities, patch coordinates and the matching text prototype bank.

[Data preparation](docs/data_preparation.md) defines the file formats, clinical encoding and upstream image-processing requirements. [Training settings](docs/training_settings.md) records the architecture and optimization settings.

### Patient allocation

The split JSON has two keys, `train` and `validation`, each containing an array of patient ID strings matching the manifest. Choose the allocation and proportions for your cohort. All slides of each patient follow that patient's allocation. The trainer checks unique IDs, disjoint partitions, complete manifest coverage and representation of all three outcomes in each partition before fitting clinical preprocessing.

The default configuration accepts user-defined center names and cohort sizes. `internal_center` optionally restricts the manifest to one center; `expected_class_counts` optionally checks a user-specified allocation. Both default to `null`. Study patient data and allocations remain private; users prepare inputs from their own datasets.

CONCH is obtained from its [official repository](https://github.com/mahmoodlab/CONCH). Users define the text descriptions for the eight tissue concepts and generate the matching prototype bank and concept-score caches, following [Data preparation](docs/data_preparation.md).

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

## Train the main model

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

Tests cover semantic selection, patient separation, training-only clinical preprocessing, the manuscript loss formula, raw-score gating, matched stochastic masks, auxiliary gradients, a six-epoch synthetic training run, and checkpoint-to-prediction consistency. The synthetic run validates the training workflow on a user-defined cohort and allocation.

## Publication and license

Associated manuscript: *TriPath-Lung: semantically guided and interpretable prediction of pathological response to neoadjuvant immunotherapy*.

Source code is distributed under the [MIT License](LICENSE). Foundation-model weights and clinical data follow their respective access and licensing terms. Training runs locally from cached features; repository configuration contains no service credentials.
