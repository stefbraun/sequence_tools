import editdistance
import numpy as np

class meter:
    def __init__(self, blank=None):
        self.guessed_labels = []
        self.target_labels = []
        self.blank = blank

    def extend_guessed_labels(self, prediction):
        prediction_sm = softmax(prediction, axis=-1) # compute softmax in last dimension
        guessed_batch_labels = convert_prediction_to_transcription(prediction_sm, self.blank,
                                                                        joiner='')  # greedy path, remove repetitions, prepare string
        self.guessed_labels.extend(guessed_batch_labels)

        return self.guessed_labels

    def extend_target_labels(self, bY, b_lenY):
        # TODO remove the easier batch labels step once the hdf5 are fixed
        easier_batch_labels = convert_from_ctc_to_easy_labels(bY, b_lenY)  # ease access to warp-ctc labels
        target_batch_labels = [vec2str(label, joiner='') for label in
                               easier_batch_labels]  # prepare string
        self.target_labels.extend(target_batch_labels)

        return self.target_labels

    def get_metrics(self):
        PER, WER, CER = calculate_error_rates(self.target_labels, self.guessed_labels)
        self.guessed_labels = []
        self.target_labels = []
        return PER, WER, CER

def convert_from_ctc_to_easy_labels(bY, lenY):
    """
    Convert labels in warp_ctc format to nested list of labels

    The returned list is easier to handle for calculation of error rates.

    :param bY: 1d vector with labels flattened over batch
    :param lenY: 1d vector with label length per sample in batch
    :return: nested list of labels
    >>> convert_from_ctc_to_easy_labels(bY=[1,2,3,4,2,4,3],lenY=[3,4])
    [[1, 2, 3], [4, 2, 4, 3]]

    """
    curr_idx = 0
    curr_label = 0
    labels = []
    while curr_idx < len(bY):
        curr_len = lenY[curr_label]
        label_list = bY[curr_idx:curr_idx + curr_len]
        labels.append([item for item in label_list])
        curr_idx += curr_len
        curr_label += 1
    return labels

def vec2str(guess_vec, joiner=''):
    """
    Join a label sequence with separators '-' for integers or '' for characters

    :param guess_vec: 1d label vector
    :param int_to_hr: 1d vector, each index is filled by a string
    :param joiner: string between labels
    :return: 1d vector with guessed label

    >>> vec2str(guess_vec=[2,4,1,0])
    '2-4-1-0'
    """
    guessed_label = '-'.join([str(item) for item in guess_vec])

    return guessed_label

def greedy_decoder(sample_prediction, blank, joiner=''):
    """Return the greedy path throug a single sample and eliminate duplicates and blanks

    sample_prediction : Time X Features 2d Matrix of label probabilities

    Get greedy decoding of diagonal matrix. Index 0: blank label
    >>> greedy_decoder(sample_prediction=np.diag([1,1,1]), blank=0, joiner='')
    '1-2'

    """
    guess_vec = np.argmax(sample_prediction, axis=1)
    guess_vec_elim = eliminate_duplicates_and_blanks(guess_vec, blank)
    return vec2str(guess_vec_elim, joiner)


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def eliminate_duplicates_and_blanks(guess_vec, blank):
    """
    Map sequences of frame-level CTC labels to single lexicon units

    :param guess_vec: 1d vector with guessed label for each time frame
    :return: guess_vec with duplicates and blanks eliminated
    >>> eliminate_duplicates_and_blanks(guess_vec=[0,0,1,2,2,3,1,0,3,3], blank = 0)
    [1, 2, 3, 1, 3]
    """

    rv = []
    # Remove duplicates
    for item in guess_vec:
        if (len(rv) == 0 or item != rv[-1]):
            rv.append(item)
    # Remove blanks (warp ctc label: label 0, tensorflow: last label)
    final_rv = []
    for item in rv:
        if item != blank:
            final_rv.append(item)
    return final_rv

def calculate_error_rates(target_labels, guessed_labels, space_idx=3):
    """
    Calculate phrase error rate, word error rate and character error rate

    Warning: label '3' is considered as <space> label by default.

    Notice:  In early CTC training stages, WER and CER are suspect to be exactly one. This happens because the network
             only outputs blank labels. This blank labels should be removed before being passed to this function. This
             leads to guessed labels that are only lists of empty strings ''. The editdistance between
             an empty string '' and a target string sequence is equal to the length of the target
             string. If then divided by the target string length, the result is 1.


    :param target_labels: 1d vector of strings with characters separated by '-', e.g ['38-1-42','2-44-24']
    :param guessed_labels: 1d vector of strings with characters separated by '-', e.g ['38-1-42','2-44-24']
    :param type: currently only 'int' supported
    :return: phrase error rate, word error rate and character error rate - NOT capped at 100%

    Check for matching case
    >>> calculate_error_rates_dbg(target_labels=['38-1-42-3-37-22','2-44-24'], guessed_labels=['38-1-42-3-37-22','2-44-24'])
    (0.0, 0.0, 0.0)

    Check for non-matching case
    >>> calculate_error_rates_dbg(target_labels=['38-1-42-3-37-22', '2-44', '1'], guessed_labels=['38-1-42-37-22', '2-44-24', '1'])
    (0.6666666666666667, 0.75, 0.2222222222222222)

    Check for kaldis compute-wer 1/2
    >>> calculate_error_rates_dbg(target_labels=['1-3-2-3-33-3-4', '7-3-8-3-9', '5-3-7-3-33','4-3-33-3-2-3-1','1','1'], guessed_labels=['1-3-2-3-33-3-4-3-5', '7-3-8','5-3-8-3-33', '1-3-33-3-5-3-2','1','1-3-2-3-4'])
    (0.8333333333333334, 0.5, 0.46153846153846156)

    Check for kaldis computer-wer 2/2
    >>> calculate_error_rates_dbg(target_labels=['1'], guessed_labels=['1-3-2-3-4'])
    (1.0, 2.0, 4.0)
    """

    # Get Phrase Error Rate Match
    phrases_correct = 0
    for idx, target in enumerate(target_labels):
        if len(target) == len(guessed_labels[idx]) and np.all(target == guessed_labels[idx]):
            phrases_correct += 1
    PER = 1. - (float(phrases_correct) / len(target_labels))

    # Word error rate
    words_wrong = 0
    total_words = 0
    for lbl_idx, target in enumerate(target_labels):
        guess_words = guessed_labels[lbl_idx].split('-' + str(space_idx) + '-')
        target_words = target.split('-' + str(space_idx) + '-')
        errors = int(editdistance.eval(guess_words, target_words))
        words_wrong += errors
        total_words += len(target_words)
    WER = float(words_wrong) / total_words

    # Character error rate
    chars_wrong = 0
    total_chars = 0
    for idx, target in enumerate(target_labels):
        guess_chars = guessed_labels[idx].split('-')
        target_chars = target.split('-')
        errors = int(editdistance.eval(target_chars, guess_chars))
        chars_wrong += errors
        total_chars += len(target_chars)
    CER = float(chars_wrong) / total_chars

    return PER, WER, CER

def convert_prediction_to_transcription(prediction, blank, joiner):
    # Prediction input : Time X Batch X Features 3D matrix
    prediction = prediction.transpose([1, 0, 2])
    guessed_labels = [greedy_decoder(phrase, blank, joiner) for phrase in prediction]
    return guessed_labels