"""Patient-level metrics and train-only ordered-cutpoint calibration."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score, roc_auc_score, log_loss)


def probabilities(scores, cutpoints):
    logits = np.clip(np.asarray(scores)[:, None]-np.asarray(cutpoints)[None, :], -700, 700)
    q = 1/(1+np.exp(-logits))
    return np.column_stack((1-q[:, 0], q[:, 0]-q[:, 1], q[:, 1]))


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = p.argmax(1)
    aucs = [float(roc_auc_score(y == k, p[:, k])) if len(np.unique(y == k)) == 2 else None for k in range(3)]
    result = dict(accuracy=float(accuracy_score(y, pred)), macro_auroc=float(np.mean(aucs)) if None not in aucs else None,
                  macro_f1=float(f1_score(y, pred, labels=[0,1,2], average='macro', zero_division=0)),
                  balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                  qwk=float(cohen_kappa_score(y, pred, weights='quadratic')),
                  brier=float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))),
                  nll=float(log_loss(y, p, labels=[0,1,2])))
    confidence = p.max(1); bins = np.minimum((confidence*10).astype(int), 9)
    result['ece_10bin'] = float(sum(np.mean(bins == b)*abs(confidence[bins == b].mean()-(pred[bins == b] == y[bins == b]).mean()) for b in range(10) if (bins == b).any()))
    precision = precision_score(y, pred, labels=[0,1,2], average=None, zero_division=0)
    recall = recall_score(y, pred, labels=[0,1,2], average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=[0,1,2], average=None, zero_division=0)
    result['per_class'] = [dict(label=k, support=int((y == k).sum()), auc=aucs[k], precision=float(precision[k]),
                              sensitivity=float(recall[k]), f1=float(f1[k]),
                              specificity=float((pred[y != k] != k).mean()) if (y != k).any() else None,
                              ovr_accuracy=float(((y == k) == (pred == k)).mean())) for k in range(3)]
    return result


def fit_cutpoints(y, scores):
    """Grid objective: macro-F1, balanced accuracy, MPR recall, then QWK.

    Only training patients may be passed. Ties prefer smaller |first cutpoint|
    and a gap nearer 0.5. No validation/external labels enter this search.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    lo, hi = np.percentile(scores, [2, 98])
    best, cuts = None, None
    for center in np.linspace(lo-1, hi+1, 81):
        for gap in np.linspace(.05, 2., 40):
            pred = probabilities(scores, [center, center+gap]).argmax(1)
            cm = np.bincount(3*y+pred, minlength=9).reshape(3, 3)
            support, predicted = cm.sum(1), cm.sum(0)
            recall = np.diag(cm)/np.maximum(support, 1)
            f1 = 2*np.diag(cm)/np.maximum(support+predicted, 1)
            weights = (np.arange(3)[:, None]-np.arange(3)[None, :])**2
            expected = np.outer(support, predicted)/len(y)
            qwk = 1-(weights*cm).sum()/max((weights*expected).sum(), 1e-12)
            key = (f1.mean(), recall.mean(), recall[1], qwk, -abs(center), -abs(gap-.5))
            if best is None or key > best:
                best, cuts = key, [float(center), float(center+gap)]
    return cuts
