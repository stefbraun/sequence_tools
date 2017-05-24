import log
import unittest
import collections
import os
import numpy as np

class log_tests(unittest.TestCase):
    def test_write_log(self):

        # prepare
        try:
            os.remove('test_write_log.csv')
        except:
            pass

        # test
        new_dict = collections.OrderedDict()
        for epoch in range(3):
            new_dict['epoch'] = epoch
            new_dict['loss'] = epoch/2.0
            new_dict['wer'] =  epoch/3.0
            log.write_log('test_write_log.csv', log_dict=new_dict)

        filesize=os.path.getsize('test_write_log.csv')
        self.assertEqual(filesize, 79)

        # remove written data
        try:
            os.remove('test_write_log.csv')
        except:
            pass

    def test_show_log(self):

        # prepare
        try:
            os.remove('test_write_log.csv')
        except:
            pass

        # test
        new_dict = collections.OrderedDict()
        for epoch in range(50):
            new_dict['epoch'] = epoch
            new_dict['loss'] = epoch/1.05
            new_dict['wer'] =  epoch/1.10
            new_dict['ep_type'] = 'superLSTM'
            log.write_log('test_write_log.csv', log_dict=new_dict)
        new_dict = collections.OrderedDict()
        for epoch in range(50):
            new_dict['epoch'] = epoch
            new_dict['loss'] = epoch / 1.20
            new_dict['wer'] = epoch / 1.33
            new_dict['ep_type'] = 'superGRU'
            log.write_log('test_write_log.csv', log_dict=new_dict)


        fig, ax = log.show_log('test_write_log.csv',category = 'ep_type', selection = [0,1], x_mode='epoch', y_mode = 'wer')
        x_plot, y_plot = ax.lines[0].get_xydata().T

        self.assertEqual((np.sum(x_plot), np.sum(y_plot)), (1225.0, 1113.6363636363637))

        # remove written data
        try:
            os.remove('test_write_log.csv')
        except:
            pass




