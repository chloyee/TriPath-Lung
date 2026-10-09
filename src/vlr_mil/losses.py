"""Class-weighted ordinal likelihood and MPR-enriched online pair ranking."""
import numpy as np
import torch
from torch.nn import functional as F


def ordinal_loss(output, label, class_weights, threshold_positive_weights):
    target = F.one_hot(label.long(), 3).float()
    ordinal = output['risk_score'].new_zeros(())
    for j, (q, t) in enumerate(((output['q1'], target[1]+target[2]), (output['q2'], target[2]))):
        q = q.clamp(1e-7, 1-1e-7)
        ordinal = ordinal - (t*q.log() + (1-t)*(1-q).log()) * (1+(threshold_positive_weights[j]-1)*t)
    # sum preserves the patient's class weight with one patient per step.
    nll = F.nll_loss(output['probabilities'].clamp_min(1e-8).log().reshape(1, 3),
                     label.reshape(1), weight=class_weights, reduction='sum')
    return ordinal + .5 * nll


def ranking_loss(score, label, memory, max_pairs=64, mpr_factor=4.):
    candidates = [(y, s) for y, s in memory if y != label]
    if not candidates:
        return None
    if len(candidates) > max_pairs:
        weights = np.array([mpr_factor if label == 1 or y == 1 else 1. for y, _ in candidates])
        selected = np.random.choice(len(candidates), max_pairs, replace=False, p=weights/weights.sum())
        candidates = [candidates[int(i)] for i in selected]
    return torch.stack([F.softplus(-(1. if label > y else -1.) * (score.reshape(())-s.reshape(())))
                        for y, s in candidates]).mean()
