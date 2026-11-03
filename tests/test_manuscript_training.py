import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from vlr_mil.data import CONCEPTS, ClinicalTransform, read_manifest, validate_split
from vlr_mil.losses import ordinal_loss
from vlr_mil.model import VLRMIL
from vlr_mil.train import build_model, predict

ROOT = Path(__file__).resolve().parents[1]


class ManuscriptTests(unittest.TestCase):
    def test_ordinal_loss_matches_equation_11(self):
        for y in range(3):
            score = torch.tensor(0.4, requires_grad=True)
            logits = score - torch.tensor([-0.2, 1.1])
            q = logits.sigmoid()
            p = torch.stack([1-q[0], q[0]-q[1], q[1]])
            weights = torch.tensor([0.8, 1.7, 1.2])
            positive = torch.tensor([0.6, 1.8])
            expected = F.binary_cross_entropy_with_logits(
                logits, torch.tensor([float(y >= 1), float(y >= 2)]),
                pos_weight=positive, reduction='sum')
            expected = expected - 0.5 * weights[y] * p[y].log()
            actual = ordinal_loss(dict(risk_score=score, q1=q[0], q2=q[1], probabilities=p),
                                  torch.tensor(y), weights, positive)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            self.assertTrue(torch.isfinite(score.grad))

    def test_gate_uses_raw_similarity(self):
        torch.set_num_threads(2)
        model = VLRMIL(torch.randn(8, 512)).eval()
        scores = torch.linspace(-0.7, 0.3, 40).reshape(5, 8)
        captured = []
        hook = model.slide_encoder.alignment_gate[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        with torch.no_grad():
            model.encode_slide(dict(features=torch.randn(5, 512), concept_scores=scores,
                                    coordinates=torch.randn(5, 2)))
        hook.remove()
        torch.testing.assert_close(captured[0][:, -1], scores.max(dim=1).values)
        self.assertIsInstance(model.feature_adapter, torch.nn.Identity)

    def test_patient_split_guards(self):
        groups = {str(i): [dict(label=str(i % 3))] for i in range(6)}
        good = dict(train=['0', '1', '2'], validation=['3', '4', '5'])
        self.assertEqual(validate_split(groups, good), good)
        bad_splits = [dict(train=['0', '1', '2'], validation=['2', '3', '4', '5']),
                      dict(train=['0', '1', '1', '2'], validation=['3', '4', '5']),
                      dict(train=['0', '1', '2'], validation=['3', '4']),
                      dict(train=['0', '1', '2'], validation=['3', '4', 'unknown'])]
        for split in bad_splits:
            with self.assertRaises(ValueError):
                validate_split(groups, split)
        with self.assertRaises(ValueError):
            validate_split(groups, good, {'train': [136, 63, 117], 'validation': [34, 16, 30]})

    def test_fixed_split_training_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root/'candidates'
            (features/'USER_CENTER').mkdir(parents=True)
            rng = np.random.default_rng(33)
            rows = []
            for i in range(9):
                # Patient 0 has two slides; both stay in the training partition.
                for j in range(2 if i == 0 else 1):
                    slide = f's{i}_{j}'
                    rows.append(dict(patient_id=f'p{i}', slide_id=slide, center='USER_CENTER',
                        label=str(i % 3), pathology_type='LUAD', T_stage='T2a',
                        age=str(40+i if i < 6 else 90), pdl1_tps='0.1', pdl1_unit='fraction', smoking_status='0'))
                    np.savez(features/'USER_CENTER'/f'{slide}.npz',
                        features=rng.normal(size=(5, 512)).astype('float32'),
                        concept_scores=rng.uniform(-.5, .5, size=(5, 8)).astype('float32'),
                        level0_xy=rng.integers(0, 1000, size=(5, 2)), concept_names=np.array(CONCEPTS))
            manifest = root/'manifest.csv'
            with manifest.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            split = dict(train=[f'p{i}' for i in range(6)], validation=[f'p{i}' for i in range(6, 9)])
            (root/'split.json').write_text(json.dumps(split))
            config = json.loads((ROOT/'configs/main.json').read_text())
            self.assertIsNone(config['internal_center'])
            self.assertIsNone(config['expected_class_counts'])
            config.update(epochs=6, patience=8)
            (root/'config.json').write_text(json.dumps(config))
            bank = rng.normal(size=(8, 512)).astype('float32')
            bank /= np.linalg.norm(bank, axis=1, keepdims=True)
            np.savez(root/'concepts.npz', embeddings=bank, concept_names=np.array(CONCEPTS))
            command = [sys.executable, '-m', 'vlr_mil.train', '--config', str(root/'config.json'),
                '--manifest', str(manifest), '--split', str(root/'split.json'),
                '--features-root', str(features), '--concept-embeddings', str(root/'concepts.npz'),
                '--output', str(root/'run'), '--selection-output', str(root/'selected'),
                '--device', 'cpu', '--threads', '2']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            dest = root/'run/main'
            history = [json.loads(line) for line in (dest/'history.jsonl').read_text().splitlines()]
            self.assertEqual([r['channel_binding_active'] for r in history], [False]*5 + [True])
            self.assertGreater(history[-1]['channel_binding_loss'], 0)
            checkpoint = torch.load(dest/'best.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['split'], split)
            self.assertAlmostEqual(checkpoint['clinical']['age_mean'], 42.5)
            self.assertNotIn('calibrated_cutpoints', checkpoint)
            model = build_model(torch.tensor(bank), checkpoint['config'], 'cpu')
            model.load_state_dict(checkpoint['model'])
            regenerated = predict(model, ClinicalTransform(state=checkpoint['clinical']),
                read_manifest(manifest), split['validation'], root/'selected', 'cpu')
            with (dest/'validation_predictions.csv').open() as f:
                saved = list(csv.DictReader(f))
            self.assertEqual([r['patient_id'] for r in saved], split['validation'])
            for first, second in zip(saved, regenerated):
                p = [float(first[k]) for k in ['prob_non_mpr', 'prob_mpr_only', 'prob_pcr']]
                self.assertAlmostEqual(sum(p), 1., places=6)
                self.assertTrue(all(0 <= x <= 1 for x in p))
                for key in ['prob_non_mpr', 'prob_mpr_only', 'prob_pcr']:
                    self.assertAlmostEqual(float(first[key]), second[key], places=6)
            # A second invocation must not overwrite the completed experiment.
            repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)


if __name__ == '__main__':
    unittest.main()
