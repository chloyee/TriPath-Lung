import unittest
import json
from pathlib import Path
import torch
from vlr_mil.model import VLRMIL
from vlr_mil.losses import channel_binding_loss


class BindingTests(unittest.TestCase):
    def test_default_config_enables_binding(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/main.json').read_text())
        self.assertEqual(config['channel_binding_weight'],.05)
        self.assertEqual(config['channel_binding_warmup_epochs'],5)
        self.assertEqual(config['channel_binding_support_margin'],.01)
        self.assertEqual(config['channel_binding_specificity_margin'],.01)

    def test_loss_targets_true_class(self):
        full = torch.tensor([.2,.5,.3], requires_grad=True)
        # For MPR, correct removal drops P(MPR) by .10; other drops are .01.
        removed = torch.tensor([[.2,.49,.31],[.25,.40,.35],[.21,.49,.30]], requires_grad=True)
        output = dict(probabilities=full, channel_ablated_probabilities=removed)
        self.assertEqual(channel_binding_loss(output,1).item(),0.)
        bad = dict(probabilities=full,channel_ablated_probabilities=removed[[1,0,2]])
        loss = channel_binding_loss(bad,1)
        self.assertAlmostEqual(loss.item(),.055,places=6)
        loss.backward()
        self.assertIsNotNone(removed.grad)

    def test_stochastic_masks_rng_and_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(55)
        model=VLRMIL(torch.randn(8,512));model.train()
        slides=[dict(features=torch.randn(18,512),concept_scores=torch.randn(18,8),
                     coordinates=torch.randn(18,2)) for _ in range(2)]
        clinical=(torch.tensor([1,3,1]),torch.randn(4),torch.zeros(5,dtype=torch.bool))
        before=torch.get_rng_state()
        ordinary=model(slides,*clinical)
        after=torch.get_rng_state()
        torch.set_rng_state(before)
        bound=model.forward_with_channel_ablation(slides,*clinical)
        torch.testing.assert_close(bound['probabilities'],ordinary['probabilities'],rtol=0,atol=0)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for k in range(3):
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(before)
                expected=model(slides,*clinical,ablate_patch_channel=k)['probabilities']
            torch.testing.assert_close(bound['channel_ablated_probabilities'][k],expected,rtol=0,atol=0)
        # Large margins activate the test gradient; experiment defaults remain .01.
        channel_binding_loss(bound,1,support_margin=1.,specificity_margin=1.).backward()
        grad=model.slide_encoder.pool.attention.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(),0.)
        torch.testing.assert_close(bound['channel_ablated_probabilities'].sum(-1),torch.ones(3))


if __name__=='__main__':
    unittest.main()
