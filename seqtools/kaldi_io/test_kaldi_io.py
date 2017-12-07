import unittest
import collections
import os
import numpy as np
from . import kaldi_io

class kaldi_tests(unittest.TestCase):
    def test_read_write_ark(self):
        ark_file = 'temp.ark'
        for i in range(10):
            print(i)
            np.random.seed()
            test_mat = np.random.random((1000,333))
            test_mat2 = test_mat / (i+1)
            ref_dict = {'1': test_mat, '2': test_mat2}
            with open(ark_file, 'wb') as f:
                for key, mat in ref_dict.items():
                    kaldi_io.write_mat(f, mat, key=key)

            test_dict = {key: mat for key, mat in kaldi_io.read_mat_ark(ark_file)}

            for key in test_dict:
                self.assertEqual(np.all(ref_dict[key]==test_dict[key]), True)
            os.remove('temp.ark')