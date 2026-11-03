# Outcome-directed patch attention

TriPath-Lung directs patch-pooling channels 0, 1 and 2 toward non-MPR, MPR-only and pCR through their effect on the shared ordinal prediction. The auxiliary objective acts through the shared ordinal prediction head.

For a patient with label `y`, compute complete probabilities `p` and three channel-ablated predictions `p(-k)`. Each ablation sets the weighted patch summary for channel `k` to zero before concatenation and projection in every slide. The remaining network recomputes the three response probabilities.

```text
D[k,y] = p[y] - p(-k)[y]
L_bind = relu(0.01 - D[y,y])
       + mean_{k != y} relu(0.01 + D[k,y] - D[y,y])
L = L_ord + 0.5 * L_cls + 0.05 * L_rank + lambda_bind * L_bind
```

The support term encourages the outcome-matched channel to increase the observed-class probability. The specificity term encourages a larger probability change for that channel than for either other channel. MPR-only supervision uses its interval probability between the two cumulative thresholds.

## Training schedule

`configs/main.json` sets `channel_binding_warmup_epochs=5` and `channel_binding_weight=0.05`. The binding weight is zero for epochs 1–5 and 0.05 from epoch 6. The two margins are `0.01` probability units. Complete and ablated passes share the selected patches and replay the same CPU/CUDA random states, matching feature, Transformer, clinical and slide dropout. Patch dropout is sampled once by the loader. Gradients propagate through all four predictions. Standard inference uses one complete forward pass.

The full training command is in the [README](../README.md). Internal validation uses the complete model and monitors the weighted ordinal and class losses. Checkpoint metadata records the selected epoch.

## Reading the outputs

`history.jsonl` records the binding loss and a 3 × 3 mean probability-drop matrix: rows are observed outcomes and columns are ablated patch channels. These are training diagnostics with the configured sampling and dropout. Patch attention describes relative pooling within each channel; channel ablation measures dependence of the shared prediction on a pooled representation. Patient-level slide aggregation channels are jointly optimized.

Run `python -m unittest discover -s tests -p test_channel_binding.py -v` to check loss direction, stochastic-mask reuse, probability normalization and gradient propagation.
