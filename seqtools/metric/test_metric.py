import unittest
import numpy as np

from . import metric

class metrics_tests(unittest.TestCase):
    def test_convert_from_ctc_to_easy_labels(self):
        easy_labels = metric.convert_from_ctc_to_easy_labels(bY=[1, 2, 3, 4, 2, 4, 3], lenY=[3, 4])

        self.assertEqual(easy_labels, [[1, 2, 3], [4, 2, 4, 3]])

    def test_vec2str(self):
        string = metric.vec2str(guess_vec=[2, 4, 1, 0])
        self.assertEqual(string, '2-4-1-0')

    def test_greedy_decoder_warp_ctc(self):
        sample_prediction = np.diag([1, 1, 1])
        decoded = metric.greedy_decoder(sample_prediction=sample_prediction, blank=0,joiner='')
        self.assertEqual(decoded, '1-2')

    def test_greedy_decoder_tf(self):
        sample_prediction = np.diag([1, 1, 1])
        decoded = metric.greedy_decoder(sample_prediction=sample_prediction, blank=2, joiner='')
        self.assertEqual(decoded, '0-1')

    def test_eliminate_duplicates_and_blanks_warp_ctc(self):
        eliminated=metric.eliminate_duplicates_and_blanks(guess_vec=[0, 0, 1, 2, 2, 3, 1, 0, 3, 3], blank=0)
        self.assertEqual(eliminated, [1, 2, 3, 1, 3])

    def test_eliminate_duplicates_and_blanks_tf_ctc(self):
        eliminated=metric.eliminate_duplicates_and_blanks(guess_vec=[0, 0, 1, 2, 2, 3, 1, 0, 3, 3], blank=3)
        self.assertEqual(eliminated, [0,1,2,1,0])

    def test_calculate_error_rates_matching(self):

        error_rates=metric.calculate_error_rates(target_labels=['38-1-42-3-37-22', '2-44-24'],
                                       guessed_labels=['38-1-42-3-37-22', '2-44-24'])
        self.assertEqual(error_rates, (0.0, 0.0, 0.0))

    def test_calculate_error_rates_non_matching(self):

        error_rates = metric.calculate_error_rates(target_labels=['38-1-42-3-37-22', '2-44', '1'],
                                                   guessed_labels=['38-1-42-37-22', '2-44-24', '1'])
        self.assertEqual(error_rates, (0.6666666666666667, 0.75, 0.2222222222222222))

    def test_calculate_error_rates_kaldi1(self):
        error_rates = metric.calculate_error_rates(target_labels=['1-3-2-3-33-3-4', '7-3-8-3-9', '5-3-7-3-33', '4-3-33-3-2-3-1', '1', '1'],
                                                   guessed_labels=['1-3-2-3-33-3-4-3-5', '7-3-8', '5-3-8-3-33', '1-3-33-3-5-3-2', '1', '1-3-2-3-4'])
        self.assertEqual(error_rates, (0.8333333333333334, 0.5, 0.46153846153846156))

    def test_calculate_error_rates_kaldi2(self):
        error_rates = metric.calculate_error_rates(target_labels=['1'], guessed_labels=['1-3-2-3-4'])
        self.assertEqual(error_rates, (1.0, 2.0, 4.0))

    def test_convert_prediction_to_transcription_warp_ctc(self):
        prediction = np.zeros((2,5,4)) # as a Batch X Timesteps X Features 3D matrix
        prediction[0,:4,:]=np.diag([1, 1, 1, 1])
        prediction[1, :4, :] = np.roll(np.diag([1,1,1,1]),4)
        prediction[1,4,2] = 1
        prediction=np.swapaxes(prediction,0,1) # as a Time X Batch X Features 3D matrix
        transcription=metric.convert_prediction_to_transcription(prediction=prediction, blank=0, joiner='')
        self.assertEqual(transcription, ['1-2-3', '3-1-2'])

    def test_convert_prediction_to_transcription_tf_ctc(self):
        prediction = np.zeros((2,5,4)) # as a Batch X Timesteps X Features 3D matrix
        prediction[0,:4,:]=np.diag([1, 1, 1, 1])
        prediction[1, :4, :] = np.roll(np.diag([1,1,1,1]),4)
        prediction[1,4,2] = 1
        prediction=np.swapaxes(prediction,0,1) # as a Time X Batch X Features 3D matrix
        transcription=metric.convert_prediction_to_transcription(prediction=prediction, blank=3, joiner='')
        self.assertEqual(transcription, ['0-1-2-0', '0-1-2'])

    def test_metrics_warp_ctc1(self):
        meter = metric.meter(blank=0)

        prediction = np.zeros((2, 5, 4))  # as a Batch X Timesteps X Features 3D matrix
        prediction[0, :4, :] = np.diag([1, 1, 1, 1])
        prediction[1, :4, :] = np.roll(np.diag([1, 1, 1, 1]), 4)
        prediction[1, 4, 2] = 1
        prediction = np.swapaxes(prediction, 0, 1)  # as a Time X Batch X Features 3D matrix

        meter.extend_guessed_labels(prediction=prediction)
        meter.extend_target_labels([1,2,3,3,1,2],[3,3])
        measure = meter.get_metrics()
        self.assertEqual(measure, (0.0, 0.0, 0.0))

    def test_metrics_warp_ctc2(self):
        meter = metric.meter(blank=0)

        prediction = np.zeros((2, 5, 4))  # as a Batch X Timesteps X Features 3D matrix
        prediction[0, :4, :] = np.diag([1, 1, 1, 1])
        prediction[1, :4, :] = np.roll(np.diag([1, 1, 1, 1]), 4)
        prediction[1, 4, 2] = 1
        prediction = np.swapaxes(prediction, 0, 1)  # as a Time X Batch X Features 3D matrix

        meter.extend_guessed_labels(prediction=prediction)
        meter.extend_target_labels([1,2,3,3,1],[3,2])
        measure = meter.get_metrics()
        self.assertEqual(measure, (0.5, 0.5, 0.2)) #0.2 because 1 insertion over 5 target labels, 0.5 because half of the sequences is wrong

    def test_metrics_tf_ctc(self):
        meter = metric.meter(blank=3)

        prediction = np.zeros((2, 5, 4))  # as a Batch X Timesteps X Features 3D matrix
        prediction[0, :4, :] = np.diag([1, 1, 1, 1])
        prediction[1, :4, :] = np.roll(np.diag([1, 1, 1, 1]), 4)
        prediction[1, 4, 2] = 1
        prediction = np.swapaxes(prediction, 0, 1)  # as a Time X Batch X Features 3D matrix

        meter.extend_guessed_labels(prediction=prediction)
        meter.extend_target_labels([0,1,2,0,0,1],[4,2])
        measure = meter.get_metrics()
        self.assertEqual(measure, (0.5, 0.5, 0.16666666666666666)) #0.16 because 1 insertion over 6 target labels, 0.5 because half of the sequences is wrong

    def test_list_reset_tf_ctc(self):
        # test if the lists containing labels get resetted after we get the error rates!
        meter = metric.meter(blank=3)

        prediction = np.zeros((2, 5, 4))  # as a Batch X Timesteps X Features 3D matrix
        prediction[0, :4, :] = np.diag([1, 1, 1, 1])
        prediction[1, :4, :] = np.roll(np.diag([1, 1, 1, 1]), 4)
        prediction[1, 4, 2] = 1

        prediction = np.swapaxes(prediction, 0, 1)  # as a Time X Batch X Features 3D matrix
        meter.extend_guessed_labels(prediction=prediction)
        meter.extend_target_labels([0, 1, 2, 0, 0, 1], [4, 2])
        measure = meter.get_metrics()

        meter.extend_guessed_labels(prediction=prediction)
        meter.extend_target_labels([0,1,2,0,0,1],[4,2])
        measure = meter.get_metrics()
        self.assertEqual(measure, (0.5, 0.5, 0.16666666666666666)) #0.16 because 1 insertion over 6 target labels, 0.5 because half of the sequences is wrong

    def test_softmax(self):
        prediction = np.zeros((2, 5, 4))  # as a Batch X Timesteps X Features 3D matrix
        prediction[0, :4, :] = np.diag([1, 1, 1, 1])
        prediction[1, :4, :] = np.roll(np.diag([1, 1, 1, 1]), 4)
        prediction[1, 4, 2] = 1
        prediction = np.swapaxes(prediction, 0, 1)  # as a Time X Batch X Features 3D matrix

        sm = metric.softmax(prediction, axis=-1)

        self.assertEqual(sm.sum(), 10.0) 


