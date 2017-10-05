import unittest
import numpy as np
import matplotlib.pyplot as plt

from . import progress

def prepare_data1():
    error_rates = np.cos(np.arange(0,1,0.1))
    return error_rates

def prepare_data2():
    error_rates = np.sin(np.arange(0,1,0.1))
    return error_rates

class progress_tests(unittest.TestCase):
    def test_bool_best1(self):
        # decreasing error rate
        prog_track = progress.progress_tracker(wait_period=5, max_patience=5)
        prog_track.track = prepare_data1()
        self.assertEqual(prog_track.update_best(), True)

    def test_bool_best2(self):
        # increasing error rate
        prog_track = progress.progress_tracker(wait_period=5, max_patience=5)
        prog_track.track = prepare_data2()
        prog_track.update_best()
        self.assertEqual(prog_track.update_best(), False)

    def test_bool_brk(self):
        prog_track = progress.progress_tracker(wait_period=5, max_patience=5)
        prog_track.patience = 10
        self.assertEqual(prog_track.update_break(), True)

    def test_bool_brk(self):
        prog_track = progress.progress_tracker(wait_period=5, max_patience=5)
        prog_track.patience = 4
        self.assertEqual(prog_track.update_break(), False)

    def test_update1(self):
        # increasing error rate
        prog_track = progress.progress_tracker(wait_period=5, max_patience=3)
        for error_rate in range(10):
            prog_track.update(error_rate)
        self.assertEqual(prog_track.patience, 9)
        self.assertEqual(prog_track.break_bool, True)


    def test_update2(self):
        # decreasing error rate
        prog_track = progress.progress_tracker(wait_period=5, max_patience=3)
        for error_rate in range(10,0,-1):
            prog_track.update(error_rate)
        self.assertEqual(prog_track.patience, 0)
        self.assertEqual(prog_track.break_bool, False)

    def test_update3(self):
        # decresing, increasing, decreasing, increasing error rate
        prog_track = progress.progress_tracker(wait_period=5, max_patience=3)
        log = [10, 9, 8, 7, 6, 5, 4.5, 3.9, 4, 4.1, 2.1, 6, 7, 8, 9, 10, 11, 12]
        for error_rate in log:
            prog_track.update(error_rate)

        self.assertEqual(prog_track.patience, 7)
        self.assertEqual(prog_track.break_bool, True)
        self.assertEqual(prog_track.best_bool, False)
        self.assertEqual(prog_track.best_epoch, 10)
        self.assertEqual(prog_track.best_error, 2.1)

    def plot_update1(self):
        # decreasing error rate
        best_bool = []
        best_epoch = []
        best_error = []
        break_bool = []
        patience = []
        prog_track = progress.progress_tracker(wait_period=5, max_patience=3)
        for error_rate in range(10):
            prog_track.update(error_rate)
            best_bool.append(prog_track.best_bool)
            best_epoch.append(prog_track.best_epoch)
            best_error.append(prog_track.best_error)
            break_bool.append(prog_track.break_bool)
            patience.append(prog_track.patience)

        self.assertEqual(prog_track.patience, 9)
        self.assertEqual(prog_track.break_bool, True)
        self.assertEqual(prog_track.best_bool, False)

        plt.plot(best_bool, label='best_bool')
        plt.plot(best_epoch, label='best_epoch')
        plt.plot(best_error, label='best_error')
        plt.plot(break_bool, label='break_bool')
        plt.plot(patience, label = 'patience')
        plt.plot(prog_track.track, label='error_rate')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_update2(self):
        # decreasing error rate
        best_bool = []
        best_epoch = []
        best_error = []
        break_bool = []
        patience = []
        prog_track = progress.progress_tracker(wait_period=5, max_patience=3)
        for error_rate in range(10, 0, -1):
            prog_track.update(error_rate)
            best_bool.append(prog_track.best_bool)
            best_epoch.append(prog_track.best_epoch)
            best_error.append(prog_track.best_error)
            break_bool.append(prog_track.break_bool)
            patience.append(prog_track.patience)

        self.assertEqual(prog_track.patience, 0)
        self.assertEqual(prog_track.break_bool, False)
        self.assertEqual(prog_track.best_bool, True)

        plt.plot(best_bool, label='best_bool')
        plt.plot(best_epoch, label='best_epoch')
        plt.plot(best_error, label='best_error')
        plt.plot(break_bool, label='break_bool')
        plt.plot(patience, label = 'patience')
        plt.plot(prog_track.track, label='error_rate')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_update3(self):
        # decreasing error rate
        best_bool = []
        best_epoch = []
        best_error = []
        break_bool = []
        patience = []
        prog_track = progress.progress_tracker(wait_period=5, max_patience=3)
        log=[10,9,8,7,6,5,4.5,3.9,4,4.1,2.1,6,7,8,9,10,11,12]
        for error_rate in log:
            prog_track.update(error_rate)
            best_bool.append(prog_track.best_bool)
            best_epoch.append(prog_track.best_epoch)
            best_error.append(prog_track.best_error)
            break_bool.append(prog_track.break_bool)
            patience.append(prog_track.patience)

        self.assertEqual(prog_track.patience, 7)
        self.assertEqual(prog_track.break_bool, True)
        self.assertEqual(prog_track.best_bool, False)
        self.assertEqual(prog_track.best_epoch, 10)
        self.assertEqual(prog_track.best_error, 2.1)


        plt.plot(best_bool, label='best_bool')
        plt.plot(best_epoch, label='best_epoch')
        plt.plot(best_error, label='best_error')
        plt.plot(break_bool, label='break_bool')
        plt.plot(patience, label = 'patience')
        plt.plot(prog_track.track, label='error_rate')
        plt.grid()
        plt.legend()
        plt.show()

