"""Manifest validation, fold-local clinical encoding, and immutable cached bags."""
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

CONCEPTS = ['viable_tumor', 'artifact', 'immune_inflamed', 'tls_like',
            'immune_excluded', 'immune_desert', 'necrosis', 'fibrotic_stroma']
CLINICAL = ['pathology_type', 'T_stage', 'age', 'pdl1_tps', 'pdl1_unit', 'smoking_status']


def read_manifest(path, internal_center=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'patient_id', 'slide_id', 'center', 'label', *CLINICAL}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing manifest columns: {required-set(reader.fieldnames or [])}')
        rows = [r for r in reader if r.get('eligible_for_modeling', '1') == '1']
    groups, slides = {}, set()
    for r in rows:
        if internal_center is not None and r['center'] != internal_center:
            raise ValueError('Training manifest contains a non-internal center')
        for key in ['patient_id', 'center', 'slide_id']:
            if not r[key] or '/' in r[key] or '\\' in r[key] or r[key] in {'.', '..'}:
                raise ValueError(f'Invalid identifier in {key}')
        if r['label'] not in {'0', '1', '2'}:
            raise ValueError('Labels must be 0, 1, or 2')
        if r['pdl1_unit'] != 'fraction':
            raise ValueError('PD-L1 must be explicitly marked fraction; no unit guessing')
        if r['pdl1_tps'] and not 0 <= float(r['pdl1_tps']) <= 1:
            raise ValueError('PD-L1 TPS must lie in [0,1]')
        if r['age'] and not np.isfinite(float(r['age'])):
            raise ValueError('Nonfinite age; use blank for missing')
        key = (r['center'], r['slide_id'])
        if key in slides:
            raise ValueError('Duplicate slide key')
        slides.add(key)
        group = groups.setdefault(r['patient_id'], [])
        if group and any(group[0][k] != r[k] for k in ['center', 'label', *CLINICAL]):
            raise ValueError('Conflicting clinical values, center, or labels across patient slides')
        group.append(r)
    if not groups:
        raise ValueError('No eligible patients')
    return groups


def five_folds(groups, seed):
    ids = np.array(sorted(groups))
    y = np.array([int(groups[p][0]['label']) for p in ids])
    if np.bincount(y, minlength=3).min() < 5:
        raise ValueError('Each class needs at least five patients for stratified five-fold CV')
    return [dict(train=ids[tr].tolist(), validation=ids[va].tolist())
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)]


def stage(value):
    for c in value:
        if c.isdigit():
            return int(c)+1 if int(c) <= 4 else 0
    return 0


class ClinicalTransform:
    def __init__(self, rows=None, state=None):
        if state is not None:
            self.state = state
            return
        ages = [float(r['age']) for r in rows if r['age']]
        if not ages:
            raise ValueError('No observed ages in the training fold')
        tps = [np.log1p(100*float(r['pdl1_tps'])) for r in rows if r['pdl1_tps']]
        self.state = dict(age_mean=float(np.mean(ages)), age_std=float(np.std(ages)) or 1.,
                          pdl1_mean=float(np.mean(tps)) if tps else 0.,
                          pdl1_std=(float(np.std(tps)) or 1.) if tps else 1.)

    def __call__(self, row, device):
        pathology = {'LUAD': 1, 'LUSC': 2, 'Other': 3}.get(row['pathology_type'], 0)
        t = stage(row['T_stage'])
        smoking = 0
        if row['smoking_status']:
            try:
                code = int(float(row['smoking_status']))
                smoking = code+1 if code in {0, 1, 2} else 0
            except ValueError:
                pass
        s = self.state
        age = (float(row['age'])-s['age_mean'])/s['age_std'] if row['age'] else 0.
        tps = 100*float(row['pdl1_tps']) if row['pdl1_tps'] else 0.
        scaled = (np.log1p(tps)-s['pdl1_mean'])/s['pdl1_std'] if row['pdl1_tps'] else 0.
        return (torch.tensor([pathology, t, smoking], device=device),
                torch.tensor([age, scaled, float(tps >= 1), float(tps >= 50)], dtype=torch.float32, device=device),
                torch.tensor([pathology == 0, t == 0, not row['age'], not row['pdl1_tps'], smoking == 0], device=device))


def load_patient(rows, root, device, training=False, patch_dropout=.2):
    slides = []
    for r in rows:
        with np.load(Path(root)/r['center']/f"{r['slide_id']}.npz", allow_pickle=False) as cache:
            features, scores, coords = cache['features'], cache['concept_scores'], cache['level0_xy']
            selected = cache['selected']
            if features.ndim != 2 or features.shape[1] != 512 or scores.shape != (len(features), 8) or coords.shape != (len(features), 2) or selected.shape != (len(features),):
                raise ValueError('Invalid cache tensor shapes')
            if not np.isin(selected, [0, 1]).all():
                raise ValueError('selected must be a boolean mask')
            if 'concept_names' in cache and cache['concept_names'].tolist() != CONCEPTS:
                raise ValueError('Concept order differs from the configured eight concepts')
            indices = np.flatnonzero(selected.astype(bool))
            if not 1 <= len(indices) <= 512:
                raise ValueError('Cached selection must contain 1 to 512 patches; selection is not recomputed')
            if training and patch_dropout > 0 and len(indices) > 1:
                keep = np.random.random(len(indices)) >= patch_dropout
                minimum = min(16, len(indices))
                if keep.sum() < minimum:
                    keep[np.random.choice(len(indices), minimum, replace=False)] = True
                indices = indices[keep]
            arrays = dict(features=features[indices], concept_scores=scores[indices], coordinates=coords[indices])
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError('Nonfinite cached inputs')
            slides.append({k: torch.from_numpy(v.astype(np.float32)).to(device) for k, v in arrays.items()})
    return slides
