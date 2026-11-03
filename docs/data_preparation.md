# Data preparation

Training uses frozen CONCH feature caches and a patient-level clinical manifest. Prepare pretreatment biopsy WSIs with tissue filtering, Macenko stain normalization and 256 × 256-pixel patch extraction. Record the extraction resolution, stride, tissue thresholds, stain reference, CONCH checkpoint version and image transforms with the feature cache.

## CONCH and tissue concepts

Use the same CONCH version for image features and text prototypes. Save the 512-dimensional visual features used for image analysis. Compute raw cosine similarities using normalized projected image embeddings and normalized text prototypes. Each prototype averages the synonymous descriptions for its concept and is normalized after averaging.

The canonical concept order is:

```text
viable_tumor, artifact, immune_inflamed, tls_like,
immune_excluded, immune_desert, necrosis, fibrotic_stroma
```

Retain the exact text descriptions and encoder preprocessing settings used to produce the cache. The TLS-like concept describes lymphoid morphology; tissue annotation establishes TLS maturation and function.

## Slide manifest

Provide one CSV row per slide. A patient's rows share the same clinical variables, center and response label. Stable deidentified identifiers join the manifest, feature files and split JSON.

| Field | Encoding |
|---|---|
| `patient_id`, `slide_id` | Stable deidentified identifiers |
| `center` | `GDPH` for development |
| `label` | `0` non-MPR; `1` MPR-only; `2` pCR |
| `pathology_type` | `LUAD`, `LUSC`, `Other`; blank for missing |
| `T_stage` | Pretreatment `T0`–`T4`; substages such as `T2a` map to `T2` |
| `age` | Years; blank for missing |
| `pdl1_tps` | Fraction from 0 to 1; 0.01 represents 1%; blank for missing |
| `pdl1_unit` | `fraction` on every row |
| `smoking_status` | `0` never; `1` former >10 years; `2` current or cessation within 10 years; blank for missing |
| `eligible_for_modeling` | Optional: rows with `1` are included |

The model standardizes age and `log1p(100 * pdl1_tps)` using observed training patients, once per patient. Missing continuous values map to zero after standardization. Missing categories map to index zero. Five missingness flags accompany the three category embeddings and the age/TPS features. TPS ≥1% and ≥50% provide additional binary features. If a source records `<1%`, document the conversion convention; the source repository used the interval midpoint, 0.005.

## Feature files

Each slide is stored at `features/<center>/<slide_id>.npz`.

| Array | Shape and contents |
|---|---|
| `features` | `N × 512`, frozen visual features |
| `concept_scores` | `N × 8`, raw cosine similarities in canonical order |
| `level0_xy` | `N × 2`, coordinates in a common level-zero slide coordinate system |
| `concept_names` | Eight canonical names in order |
| `selected` | Boolean vector of length `N`, selecting 1–512 patches after bag preparation |

Candidate caches supply the first four arrays. The selector generates `selected` and provenance. Prepared caches supply all five. Every array uses the same patch order. Text prototypes are stored in a separate NPZ with `embeddings` of shape `8 × 512` and `concept_names` in canonical order. Values must be finite. The learned text projection consumes this prototype bank; patch matching uses raw concept scores.

## Allocation file

The split JSON contains `train` and `validation`, each holding patient ID strings. Together the lists cover the eligible development manifest exactly once. Use the original study allocation. The manuscript configuration checks both total counts and class counts. For another cohort, set `expected_class_counts` in a separate configuration to that cohort's allocation.

## Resources accompanying a training run

Keep the deidentified allocation, manifest, feature caches, prototype bank and extraction record together. Source-model weights are obtained under CONCH's access terms. The cached-feature trainer operates without an API service. Store any upstream download credentials in the user's environment, separately from source files and experiment artifacts.
