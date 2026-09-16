import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .experiment import ordered_blocks, paired_difference, rng_for
from .predict import vector

class AccumulatedControls(unittest.TestCase):
    def test_chronology_deduplication_and_nonoverlapping_chunks(self):
        # Reverse file order and duplicate an event; neither can inflate the windows.
        rows=[]
        for sentence in range(4):
            for key in range(110):
                press=sentence*100000+key*200
                rows.append([sentence,press,press+80,65+sentence,sentence*1000+key])
        rows.append(rows[0]); rows.reverse()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'user.txt'
            with path.open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t'); w.writerow(['TEST_SECTION_ID','PRESS_TIME','RELEASE_TIME','KEYCODE','KEYSTROKE_ID']); w.writerows(rows)
            g,q=ordered_blocks(path)
        self.assertEqual([n for _,n,_ in g],[50,50,10,50,50,10])
        self.assertEqual(sum(n for _,n,_ in g+q),440)
        self.assertTrue(all(round(float(x[0,4])*255)<67 for x,_,_ in g))
        self.assertTrue(all(round(float(x[0,4])*255)>=67 for x,_,_ in q))

    def test_feature_padding_and_unused_channels_cannot_signal_label(self):
        rng=np.random.default_rng(21)
        g=np.zeros((5,50,5),dtype='float32'); g[:,:,0]=rng.uniform(.05,.2,(5,50)); g[:,:,2]=rng.uniform(.1,.5,(5,50)); g[:,:,4]=65/255
        q=g.copy(); lengths=np.full(5,20,dtype='int64')
        expected=vector(g,lengths,q,lengths)
        q[:,20:]=777; q[:,:,1]=111; q[:,:,3]=222; q[:,19,2]=333
        np.testing.assert_array_equal(expected,vector(g,lengths,q,lengths))

    def test_reject_unvalidated_observation_budget(self):
        with self.assertRaises(ValueError): vector(np.zeros((20,50,5)),np.full(20,50),np.zeros((20,50,5)),np.full(20,50))

    def test_paired_cluster_difference(self):
        y=np.tile(np.arange(4),3); p=np.eye(4)[y]; users=np.repeat(['a','b','c'],4)
        result=paired_difference(y,p,p,users)
        self.assertEqual(result['accuracy_difference'],0)
        self.assertEqual(result['participant_bootstrap_95ci'],[0,0])

    def test_separate_rng_roles_with_reproducibility(self):
        np.testing.assert_array_equal(rng_for('test','a',1).random(20),rng_for('test','a',1).random(20))
        self.assertFalse(np.array_equal(rng_for('test','a',1).random(20),rng_for('train','a',1).random(20)))

if __name__=='__main__': unittest.main()
