import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vlr_mil.data import CONCEPTS, load_patient
from vlr_mil.selection import prepare_bags, select_patches

SETTINGS = json.loads((Path(__file__).resolve().parents[1]/'configs/selection.json').read_text())


class SelectionTests(unittest.TestCase):
    def test_budgets_and_repeatability(self):
        for n in [1, 128, 129, 350, 512, 900]:
            scores = np.random.default_rng(7).normal(size=(n,8)).astype(np.float32)
            first = select_patches('slide-a', scores, CONCEPTS, SETTINGS)
            second = select_patches('slide-a', scores, CONCEPTS, SETTINGS)
            for a,b in zip(first,second):
                np.testing.assert_array_equal(a,b)
            self.assertEqual(first[0].sum(), min(n,512))
            self.assertEqual((first[1] != 'not_selected').sum(), min(n,512))
            if n > 512:
                self.assertEqual((first[1]=='support').sum(),307)
                self.assertEqual((first[1]=='uncertain').sum(),102)
                self.assertEqual((first[1]=='random').sum(),103)

    def test_stable_ties(self):
        selected, groups, support = select_patches('ties', np.zeros((800,8),dtype=np.float32), CONCEPTS, SETTINGS)
        np.testing.assert_array_equal(groups[:307], np.full(307,'support'))
        np.testing.assert_array_equal(groups[307:409], np.full(102,'uncertain'))
        self.assertTrue(np.all(support == 0))

    def test_formula_and_entropy_order(self):
        scores = np.random.default_rng(3).normal(size=(1000,8)).astype(np.float32)
        selected, groups, support = select_patches('formula',scores,CONCEPTS,SETTINGS)
        expected=(scores[:,2]+scores[:,3])/2-(scores[:,4]+scores[:,5])/2+.5*scores[:,0]-.5*scores[:,1]
        np.testing.assert_array_equal(support,expected)
        np.testing.assert_array_equal(np.flatnonzero(groups=='support'),np.sort(np.argsort(-expected,kind='stable')[:307]))
        logits=scores/.1; logits-=logits.max(axis=1,keepdims=True)
        p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
        entropy=-(p*np.log(np.clip(p,1e-8,None))).sum(axis=1)
        remaining=np.argsort(-entropy,kind='stable')
        expected_uncertain=remaining[groups[remaining]!='support'][:102]
        np.testing.assert_array_equal(np.flatnonzero(groups=='uncertain'),np.sort(expected_uncertain))

    def test_invalid_input(self):
        for scores in [np.zeros((0,8)),np.full((10,8),np.nan),np.zeros((10,7))]:
            with self.assertRaises(ValueError):
                select_patches('bad',scores,CONCEPTS,SETTINGS)
        with self.assertRaises(ValueError):
            select_patches('bad',np.zeros((10,8)),CONCEPTS,dict(SETTINGS,random_fraction=.5))

    def test_cache_training_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'raw'; (source/'GDPH').mkdir(parents=True)
            rng=np.random.default_rng(4)
            features=rng.normal(size=(700,512)).astype(np.float32)
            scores=rng.normal(size=(700,8)).astype(np.float32)
            coords=np.arange(1400).reshape(700,2)
            path=source/'GDPH/s.npz'
            np.savez(path,features=features,concept_scores=scores,level0_xy=coords,concept_names=np.array(CONCEPTS))
            original=path.read_bytes()
            rows=[dict(patient_id='p',center='GDPH',slide_id='s')]
            output=root/'selected'
            prepare_bags({'p':rows},source,output,SETTINGS)
            self.assertEqual(path.read_bytes(),original)
            expected=select_patches('s',scores,CONCEPTS,SETTINGS)[0]
            bag=load_patient(rows,output,'cpu')[0]
            np.testing.assert_array_equal(bag['features'].numpy(),features[expected])
            np.testing.assert_array_equal(bag['coordinates'].numpy(),coords[expected])
            np.testing.assert_array_equal(bag['concept_scores'].numpy(),scores[expected])
            with self.assertRaises(FileExistsError):
                prepare_bags({'p':rows},source,output,SETTINGS)


if __name__=='__main__':
    unittest.main()
