"""Focused tests of leakage defenses and feature/inference consistency."""
import unittest
import numpy as np
from .features import channels, enrollment, session_vector, bundle_vector
from .run import calibrated, seed_for


class Controls(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        self.x=np.zeros((10,50,5),dtype='float32')
        self.x[:,:,0]=rng.uniform(.06,.18,(10,50))
        self.x[:,:,2]=rng.uniform(.1,.4,(10,50))
        self.x[:,:,4]=65/255
        self.ref=enrollment(self.x,np.full(10,50))

    def test_terminal_and_unused_channels_do_not_change_features(self):
        q=self.x[0].copy()
        expected=session_vector(q,50,self.ref)
        q[-1,1:4]=900
        q[:,1]=500
        q[:,3]=-500
        np.testing.assert_array_equal(expected,session_vector(q,50,self.ref))

    def test_precision_is_equalized(self):
        q=self.x[0].copy()
        q[:,0]=.1
        r=q.copy(); r[:,0]+=.00001
        np.testing.assert_array_equal(channels(q,50)[0],channels(r,50)[0])

    def test_enrollment_is_not_mutated(self):
        before=self.x.copy()
        session_vector(self.x[0],50,self.ref)
        np.testing.assert_array_equal(before,self.x)

    def test_padding_not_observed(self):
        q=self.x[0].copy(); r=q.copy(); r[20:]=999
        np.testing.assert_array_equal(session_vector(q,20,self.ref),session_vector(r,20,self.ref))

    def test_calibration_preserves_classes_and_normalizes(self):
        p=np.array([[.1,.5,.2,.2],[.7,.1,.1,.1]])
        c=calibrated(p,1.7)
        np.testing.assert_array_equal(p.argmax(1),c.argmax(1))
        np.testing.assert_allclose(c.sum(1),1)

    def test_seeds_separate_roles_people_and_classes(self):
        values=[seed_for(role,user,c) for role in ('train','test') for user in ('1','2') for c in range(4)]
        self.assertEqual(len(values),len(set(values)))

    def test_bundle_order_invariant(self):
        rows=np.stack([session_vector(q,50,self.ref) for q in self.x[:5]])
        np.testing.assert_allclose(bundle_vector(rows),bundle_vector(rows[::-1]),atol=1e-6)


if __name__=='__main__':
    unittest.main()
