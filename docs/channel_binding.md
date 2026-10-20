# Outcome-directed patch attention without an additional classifier

The default training configuration encourages patch-pooling channel 0/1/2 to support
Non-MPR/MPR-only/pCR through a functional dependence loss. It adds no parameters and
reuses the existing ordinal head. Setting `channel_binding_weight` to zero disables
this term; older configurations without the key retain their original behavior.

For every patient with true label y, compute complete probabilities p and three
counterfactual probabilities p(-k). In pass k, zero the kth patch-pool summary vector
after attention-weighted aggregation and before concatenation/projection, in every
chunk of every retained slide. The other vectors, projection bias, and clinical
input remain unchanged. Chunk/slide pooling channels are NOT assigned class labels.

Define D[k,y] = p[y] - p(-k)[y]. Then

```text
L_bind = relu(m_support - D[y,y])
       + mean_{k != y} relu(m_specificity + D[k,y] - D[y,y])
L_total = L_original + lambda * L_bind
```

The first term encourages the matching channel to support the true outcome. The
second encourages its removal to reduce the true-class probability more than removal
of another channel. Gradients flow through both complete and ablated predictions.
MPR uses P(MPR-only), not a monotonically increasing risk-score target. No validation
or external labels are used for this training term.

For example, P(MPR)=.50 and probabilities after removing channels 0/1/2 of
[.49,.40,.49] give D=[.01,.10,.01], satisfying both .01 margins. Removing channel 1
has the strongest effect on MPR probability. This is a mathematical illustration,
not a patient result.

All four passes replay the same CPU/CUDA RNG states so that clinical, slide,
Transformer and feature dropout masks are matched. The outer RNG advances as for
one ordinary forward. Patch dropout is sampled once by the data loader before the
four passes. Auxiliary gradients are not detached. This straightforward implementation
requires four forward graphs during active training and can substantially increase
runtime/memory. Ordinary inference still uses one forward pass.

## Experiment

`configs/main.json` sets lambda=.05, both margins=.01 probability units,
and five warmup epochs (auxiliary loss starts in epoch 6). These are pilot values,
not validated best hyperparameters. Standard five-fold training enables this schedule.

```bash
bash scripts/run_fivefold.sh --config configs/main.json \
  --manifest data/GDPH_internal_slides.csv \
  --features-root features/conch_v1 \
  --concept-embeddings features/conch_v1/concept_embeddings.npz \
  --output outputs/channel_binding_seed55_fold0 --fold 0 --device cpu
```

Training history includes the unweighted binding loss and a 3x3 mean delta matrix:
rows are true outcomes, columns are ablated patch channels. These training summaries
include patient resampling and dropout; they are not independent validation estimates.
Validation and early stopping retain the original ordinal loss, without auxiliary
loss or ablation. Learned ordinal cutpoints are used in this auxiliary term; optional
post-training cutpoint calibration is a separate step.

## Interpretation limits

This loss encourages class-linked functional reliance, not a guarantee of correct
localization. The network may exploit feature scale or compensate in downstream
weights. It does not guarantee diverse maps, positive contribution for every high
attention patch, or improved classification. A trained variant needs held-out
channel-ablation and patch-deletion tests, attention-overlap checks, and comparison
with the unchanged baseline. Existing checkpoints are compatible but have not learned
this new constraint; they must not be relabeled as class-bound attention models.

Tests: `python -m unittest discover -s tests -v` checks the loss direction, shared
stochastic masks, RNG preservation, probability normalization, and gradients reaching
the patch attention parameters. These are implementation tests, not efficacy evidence.
