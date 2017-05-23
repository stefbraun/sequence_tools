import unittest
import progress
import numpy as np

def prepare_data1():
    error_rates = np.cos(np.arange(0,1,0.1))
    return error_rates

def prepare_data2():
    error_rates = np.sin(np.arange(0,1,0.1))
    return error_rates

class progress_tests(unittest.TestCase):
    def test_bool_best1(self):
        prog_track = progress.progress_tracker(wait_period=5, max_patience=5)
        prog_track.track = prepare_data1()
        self.assertEqual(prog_track.bool_best(), True)

    def test_bool_best2(self):
        prog_track = progress.progress_tracker(wait_period=5, max_patience=5)
        prog_track.track = prepare_data2()
        prog_track.bool_best()
        self.assertEqual(prog_track.bool_best(), False)

    def test_check_patience1(self):
        prog_track = progress.progress_tracker(wait_period=5, max_patience=3)
        for error_rate in range(10):
            prog_track.track.append(error_rate)
            prog_track.check_patience()
        self.assertEqual(prog_track.patience, 9)

    def test_check_patience2(self):
        prog_track = progress.progress_tracker(wait_period=5, max_patience=3)
        for error_rate in range(10,0,-1):
            prog_track.track.append(error_rate)
            prog_track.check_patience()
        self.assertEqual(prog_track.patience, 0)


