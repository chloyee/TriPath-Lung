"""Optional external evaluation of fixed fold checkpoints; never selects a model."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .data import ClinicalTransform, read_manifest
from .train import build_model, predict, write_predictions, summarize, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--features-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(); torch.set_num_threads(a.threads)
    groups = read_manifest(a.manifest)
    a.output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    for index, path in enumerate(a.checkpoint):
        # Only load checkpoints from a trusted source (PyTorch pickle format).
        checkpoint = torch.load(path, map_location=a.device, weights_only=False)
        if set(groups) & set(checkpoint['internal_patient_ids']):
            raise ValueError('External patients overlap the internal development cohort')
        if any(r['center'] == checkpoint['internal_center'] for rows in groups.values() for r in rows):
            raise ValueError('External evaluation cannot include the internal center')
        state = checkpoint['model']
        model = build_model(state['concept_embeddings'], checkpoint['config'], a.device)
        model.load_state_dict(state)
        transform = ClinicalTransform(state=checkpoint['clinical'])
        rows = predict(model, transform, groups, sorted(groups), a.features_root, a.device, checkpoint.get('calibrated_cutpoints'))
        write_predictions(a.output/f'fold_{index}_predictions.csv', rows)
        dump(a.output/f'fold_{index}_metrics.json', summarize(rows))
        all_rows.append(rows)
    ensemble = []
    for i, pid in enumerate(sorted(groups)):
        row = dict(all_rows[0][i])
        for key in ['prob_non_mpr', 'prob_mpr_only', 'prob_pcr', 'shared_ordinal_score']:
            row[key] = float(np.mean([r[i][key] for r in all_rows]))
        row['predicted_label'] = int(np.argmax([row[k] for k in ['prob_non_mpr', 'prob_mpr_only', 'prob_pcr']]))
        ensemble.append(row)
    write_predictions(a.output/'ensemble_predictions.csv', ensemble)
    dump(a.output/'ensemble_metrics.json', summarize(ensemble))
    dump(a.output/'protocol.json', dict(checkpoints=[str(x) for x in a.checkpoint],
        aggregation='Unweighted probability mean; mean raw score is descriptive, not a calibrated shared scale',
        selection='No selection, fitting, or cutpoint optimization on external data'))


if __name__ == '__main__':
    main()
