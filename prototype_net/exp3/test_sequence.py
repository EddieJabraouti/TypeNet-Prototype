"""Inference and masking controls for the sequence experiment."""
import unittest
import numpy as np
import torch
from .sequence_model import baseline_reference, observed_sequence, TimingSequenceNet
from .relational import RelationalTimingNet


class SequenceControls(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(12)
        self.gallery=np.zeros((10,50,5),dtype='float32')
        self.gallery[:,:,0]=rng.uniform(.06,.2,(10,50))
        self.gallery[:,:,2]=rng.uniform(.1,.6,(10,50))
        self.gallery[:,:,4]=65/255
        self.ref,self.ctx=baseline_reference(self.gallery,np.full(10,50))

    def test_ignores_terminal_and_reconstructed_channels(self):
        q=self.gallery[0].copy(); original=observed_sequence(q,50,self.ref,self.ctx)
        q[:,1]=999; q[:,3]=-888; q[-1,:4]=12345
        changed=observed_sequence(q,50,self.ref,self.ctx)
        for a,b in zip(original,changed): np.testing.assert_array_equal(a,b)

    def test_padding_has_no_effect(self):
        q=self.gallery[0].copy(); changed=q.copy(); changed[20:]=999
        for a,b in zip(observed_sequence(q,20,self.ref,self.ctx),observed_sequence(changed,20,self.ref,self.ctx)):
            np.testing.assert_array_equal(a,b)

    def test_reference_independent_of_query(self):
        before=self.ref.copy()
        q=self.gallery[0].copy(); q[:,0]*=1.5
        observed_sequence(q,50,self.ref,self.ctx)
        np.testing.assert_array_equal(before,self.ref)

    def test_forward_backward_and_bundle_permutation(self):
        torch.manual_seed(5); torch.set_num_threads(2)
        data=[observed_sequence(q,50,self.ref,self.ctx) for q in self.gallery[:5]]
        values,keys,mask=[torch.from_numpy(np.stack([row[i] for row in data])[None]) for i in range(3)]
        ctx=torch.from_numpy(self.ctx[None])
        model=TimingSequenceNet(width=16).eval()
        s,b=model(values,keys,mask,ctx)
        sr,br=model(values.flip(1),keys.flip(1),mask.flip(1),ctx)
        torch.testing.assert_close(s,sr.flip(1)); torch.testing.assert_close(b,br)
        self.assertEqual(tuple(s.shape),(1,5,4)); self.assertEqual(tuple(b.shape),(1,4))
        (s.sum()+b.sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_individual_gallery_comparisons_are_order_invariant(self):
        torch.manual_seed(8); torch.set_num_threads(2)
        rows=[observed_sequence(q,50,self.ref,self.ctx) for q in self.gallery]
        rows+=rows[:5]
        v,k,m=[torch.from_numpy(np.stack([r[i] for r in rows])[None]) for i in range(3)]
        ctx=torch.from_numpy(self.ctx[None])
        model=RelationalTimingNet(width=16).eval()
        s,b=model(v,k,m,ctx)
        order=torch.tensor(list(reversed(range(10)))+list(range(10,15)))
        sr,br=model(v[:,order],k[:,order],m[:,order],ctx)
        torch.testing.assert_close(s,sr); torch.testing.assert_close(b,br)
        one,_=model(v[:,:11],k[:,:11],m[:,:11],ctx)
        torch.testing.assert_close(one[:,0],s[:,0])
        (s.sum()+b.sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__': unittest.main()
