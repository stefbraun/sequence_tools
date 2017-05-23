import numpy as np

class progress_tracker:
    def __init__(self, wait_period, max_patience):
        self.wait_period = wait_period
        self.max_patience = max_patience
        self.track = []
        self.best = None
        self.best_epoch = 0
        self.patience = 0
        self.brk = False

    def get_patience(self):
        if len(self.track) > self.wait_period:
            current_epoch = len(self.track)-1 #counting is index based: 0th, 1st, 2nd epoch...
            self.best = np.min(self.track)
            self.best_epoch = np.argmin(self.track)  # numpy.argmin only finds first occurence
            self.patience = current_epoch - self.best_epoch
        else:
            self.patience = 0
        return self.patience

    def check_patience(self):
        patience = self.get_patience()
        bool_break = self.bool_break()
        return bool_break

    def bool_break(self):
        if self.patience > self.max_patience:
            self.brk = 1
        else:
            self.brk = 0
        return self.brk

    def bool_best(self):
        bool_best = False
        if len(self.track) > self.wait_period: # work after wait period
            if self.track[-1] < np.min(self.track[:-1]): # check if last epoch was best epoch
                bool_best = True
            else:
                bool_best = False
        return bool_best

    def check_best(self):
        bool_best = self.bool_best()
        return bool_best