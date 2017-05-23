import unittest
import metrics

class metrics_tests(unittest.TestCase):

    def test_convert_from_ctc_to_easy_labels(self):
        easy_labels = metrics.convert_from_ctc_to_easy_labels(bY=[1,2,3,4,2,4,3],lenY=[3,4])

        self.assertEqual(easy_labels, [[1, 2, 3], [4, 2, 4, 3]])

    def test_get_single_decoding(self):
        single_decoding = metrics.get_single_decoding(guess_vec=[2, 4, 1, 0])
        self.assertEqual(single_decoding, '2-4-1-0')
