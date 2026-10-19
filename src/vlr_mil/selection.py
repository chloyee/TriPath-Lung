"""Deterministic concept-guided bag construction from raw concept scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import CONCEPTS, read_manifest


def validate_settings(settings):
    minimum, maximum = settings['min_patches_per_slide'], settings['max_patches_per_slide']
    if not 0 <= minimum <= maximum <= 512 or int(minimum) != minimum or int(maximum) != maximum or maximum < 1:
        raise ValueError('Require integer budgets 0 <= minimum <= maximum <= 512')
    fractions = [settings[k] for k in ('top_fraction', 'uncertainty_fraction', 'random_fraction')]
    if not all(np.isfinite(f) and 0 <= f <= 1 for f in fractions) or not np.isclose(sum(fractions), 1, atol=1e-12, rtol=0):
        raise ValueError('Selection fractions must be finite, nonnegative and sum to one')
    if not np.isfinite(settings['softmax_temperature']) or settings['softmax_temperature'] <= 0:
        raise ValueError('softmax_temperature must be positive and finite')
    if not all(np.isfinite(settings[k]) for k in ('alpha_tissue', 'beta_artifact')):
        raise ValueError('Support coefficients must be finite')


def select_patches(slide_id, scores, names, settings):
    """Return (selected mask, source group, response-support score).

    Scores are raw cosine similarities, not percentile ranks or RGB values.
    Ties preserve source order. Randomness depends on slide ID, not CV seed.
    """
    validate_settings(settings)
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 8 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError('Expected a nonempty finite N x 8 concept-score matrix')
    if len(names) != 8 or set(names) != set(CONCEPTS):
        raise ValueError('Expected the eight unique tissue concepts')
    column = {name: index for index, name in enumerate(names)}
    positive = (scores[:, column['immune_inflamed']] + scores[:, column['tls_like']]) / 2
    negative = (scores[:, column['immune_excluded']] + scores[:, column['immune_desert']]) / 2
    support = (positive - negative
               + float(settings['alpha_tissue']) * scores[:, column['viable_tumor']]
               - float(settings['beta_artifact']) * scores[:, column['artifact']])
    logits = scores / float(settings['softmax_temperature'])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    uncertainty = -(probabilities * np.log(np.clip(probabilities, 1e-8, None))).sum(axis=1)
    count = len(scores)
    maximum = min(int(settings['max_patches_per_slide']), count)
    if count <= int(settings['min_patches_per_slide']):
        return np.ones(count, dtype=bool), np.full(count, 'all', dtype='U12'), support
    n_top = min(maximum, round(maximum * float(settings['top_fraction'])))
    n_uncertain = min(maximum - n_top, round(maximum * float(settings['uncertainty_fraction'])))
    selected = np.zeros(count, dtype=bool)
    groups = np.full(count, 'not_selected', dtype='U12')
    top = np.argsort(-support, kind='stable')[:n_top]
    selected[top] = True
    groups[top] = 'support'
    uncertain_order = np.argsort(-uncertainty, kind='stable')
    uncertain = uncertain_order[~selected[uncertain_order]][:n_uncertain]
    selected[uncertain] = True
    groups[uncertain] = 'uncertain'
    remaining = np.flatnonzero(~selected)
    needed = maximum - int(selected.sum())
    seed = int.from_bytes(hashlib.sha256(slide_id.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
    selected[random_indices] = True
    groups[random_indices] = 'random'
    return selected, groups, support


def prepare_bags(groups, features_root, output_root, settings):
    """Create a new cache tree; preserve row order and all non-selection arrays."""
    validate_settings(settings)
    features_root, output_root = Path(features_root).resolve(), Path(output_root).resolve()
    if output_root == features_root or output_root.is_relative_to(features_root):
        raise ValueError('Selection output must be separate from the source feature cache')
    output_root.mkdir(parents=True, exist_ok=False)
    records = []
    for rows in groups.values():
        for row in rows:
            source = features_root/row['center']/f"{row['slide_id']}.npz"
            with np.load(source, allow_pickle=False) as cache:
                arrays = {key: cache[key] for key in cache.files}
            if 'concept_names' not in arrays or arrays['concept_names'].tolist() != CONCEPTS:
                raise ValueError(f'{source}: explicit canonical concept_names are required')
            scores = arrays['concept_scores']
            n = len(scores)
            if arrays['features'].shape != (n, 512) or arrays['level0_xy'].shape != (n, 2):
                raise ValueError(f'{source}: inconsistent features/scores/coordinates')
            # Promote quantized caches for arithmetic; original float32 extraction
            # scores should be retained when exact original selection is required.
            selected, labels, support = select_patches(row['slide_id'], scores.astype(np.float32), CONCEPTS, settings)
            arrays.update(selected=selected, selection_group=labels, response_support_score=support)
            destination = output_root/row['center']/source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(destination, **arrays)
            records.append(dict(center=row['center'], slide_id=row['slide_id'], candidates=n,
                                selected=int(selected.sum()), source_score_dtype=str(scores.dtype),
                                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                                selection_counts={str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}))
    provenance = dict(settings=settings, source_root=str(features_root), slides=records,
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_root/'selection_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--features-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/selection.json'))
    args = parser.parse_args()
    result = prepare_bags(read_manifest(args.manifest), args.features_root, args.output,
                          json.loads(args.config.read_text()))
    print(f"Prepared {len(result['slides'])} slide bags in {args.output}")


if __name__ == '__main__':
    main()
