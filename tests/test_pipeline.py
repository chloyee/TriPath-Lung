"""Synthetic-only tests: no research manifests, weights, or patient data."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vlr_mil.data import ClinicalTransform, CONCEPTS, five_folds, read_manifest
from vlr_mil.metrics import fit_cutpoints, probabilities
from vlr_mil.model import VLRMIL, percentile_rank_columns

ROOT = Path(__file__).resolve().parents[1]


def synthetic_data(root, center='GDPH', count=15):
    rng = np.random.default_rng(123)
    folder = root/'features'/center
    folder.mkdir(parents=True)
    rows = []
    for i in range(count):
        pid = f'synthetic_{center}_{i:03}'
        rows.append(dict(patient_id=pid, slide_id=pid+'_slide', center=center, label=str(i % 3),
            pathology_type=['LUAD', 'LUSC', 'Other'][i % 3], T_stage=f'T{i % 4 + 1}', age=str(40+i),
            pdl1_tps='' if i % 4 == 0 else str(i/100), pdl1_unit='fraction', smoking_status=str(i % 3)))
        np.savez(folder/f'{pid}_slide.npz', features=rng.normal(size=(18, 512)).astype('float32'),
                 concept_scores=rng.normal(size=(18, 8)).astype('float32'), concept_names=np.array(CONCEPTS),
                 level0_xy=rng.integers(0, 10000, (18, 2)), selected=np.ones(18, dtype=bool))
    manifest = root/f'{center}.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return manifest


class PipelineTests(unittest.TestCase):
    def test_ordinal_and_attention(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        model = VLRMIL(torch.randn(8, 512)).eval()
        slide = dict(features=torch.randn(18, 512), concept_scores=torch.randn(18, 8), coordinates=torch.randn(18, 2))
        output = model([slide], torch.tensor([1, 2, 3]), torch.zeros(4), torch.zeros(5, dtype=torch.bool))
        self.assertGreater(float(model.head.cutpoints()[1]-model.head.cutpoints()[0]), 0)
        self.assertTrue(torch.all(output['probabilities'] >= 0))
        self.assertAlmostEqual(float(output['probabilities'].sum()), 1., places=6)
        self.assertTrue(torch.allclose(output['patch_attention'][0][0].sum(-1), torch.ones(3)))
        self.assertTrue(torch.allclose(output['slide_attention'].sum(-1), torch.ones(3)))
        self.assertTrue(torch.equal(percentile_rank_columns(torch.ones(1, 8)), torch.full((1, 8), .5)))

    def test_clinical_missing(self):
        row = dict(pathology_type='LUAD', T_stage='T2b', smoking_status='0', age='60', pdl1_tps='0.01')
        transform = ClinicalTransform([row])
        cat, cont, missing = transform(row, 'cpu')
        self.assertEqual(cat.tolist(), [1, 3, 1])
        self.assertEqual(cont.tolist(), [0, 0, 1, 0])
        row.update(age='', pdl1_tps='', smoking_status='')
        cat, cont, missing = transform(row, 'cpu')
        self.assertEqual(cont.tolist(), [0, 0, 0, 0])
        self.assertEqual(missing.tolist(), [False, False, True, True, True])

    def test_fivefold_training_and_external_isolation(self):
        with tempfile.TemporaryDirectory(prefix='vlr-training-test-') as temp:
            root = Path(temp)
            manifest = synthetic_data(root)
            groups = read_manifest(manifest, 'GDPH')
            splits = five_folds(groups, 55)
            validation = []
            for split in splits:
                self.assertFalse(set(split['train']) & set(split['validation']))
                validation.extend(split['validation'])
            self.assertEqual(sorted(validation), sorted(groups))
            external = synthetic_data(root, 'External', count=6)
            with self.assertRaises(ValueError):
                read_manifest(external, 'GDPH')
            config = json.loads((ROOT/'configs/main.json').read_text())
            config.update(epochs=1, patience=1)
            (root/'config.json').write_text(json.dumps(config))
            np.savez(root/'concepts.npz', embeddings=np.random.default_rng(4).normal(size=(8, 512)).astype('float32'))
            command = [sys.executable, '-m', 'vlr_mil.train', '--config', str(root/'config.json'),
                       '--manifest', str(manifest), '--features-root', str(root/'features'),
                       '--concept-embeddings', str(root/'concepts.npz'), '--output', str(root/'output'),
                       '--device', 'cpu', '--threads', '1']
            run = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            with (root/'output/oof_predictions.csv').open() as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual(len(predictions), 15)
            checkpoints = sorted((root/'output').glob('seed55_fold*/best.pt'))
            self.assertEqual(len(checkpoints), 5)
            for path in checkpoints:
                ckpt = torch.load(path, weights_only=False, map_location='cpu')
                self.assertLess(*ckpt['calibrated_cutpoints'])
                self.assertEqual(set(ckpt['internal_patient_ids']), set(groups))
            evaluate = [sys.executable, '-m', 'vlr_mil.evaluate', '--checkpoint', *map(str, checkpoints),
                        '--manifest', str(external), '--features-root', str(root/'features'),
                        '--output', str(root/'external_output'), '--threads', '1']
            run = subprocess.run(evaluate, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            # Refuse an attempt to evaluate development patients as independent external data.
            evaluate[evaluate.index('--manifest')+1] = str(manifest)
            evaluate[evaluate.index('--output')+1] = str(root/'forbidden_output')
            run = subprocess.run(evaluate, text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('overlap', run.stderr)
            run = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('overwrite', run.stderr)


if __name__ == '__main__':
    unittest.main()
