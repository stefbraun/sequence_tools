import numpy as np

class progress_tracker:
    def __init__(self, wait_period, max_patience):
        self.wait_period = wait_period
        self.max_patience = max_patience
        self.track = []
        self.best_error = None
        self.best_epoch = 0
        self.best_bool = False
        self.patience = 0
        self.break_bool = False

    def update_patience(self):
        if len(self.track) > self.wait_period:
            current_epoch = len(self.track)-1 #counting is index based: 0th, 1st, 2nd epoch...
            self.patience = current_epoch - self.best_epoch
        else:
            self.patience = 0
        return self.patience

    def update_break(self):
        if self.patience > self.max_patience:
            self.break_bool = 1
        else:
            self.break_bool = 0
        return self.break_bool

    def update_best(self):
        if len(self.track) > self.wait_period: # work after wait period
            self.best_error = np.min(self.track)
            self.best_epoch = np.argmin(self.track) # numpy.argmin only finds first occurence
            if self.track[-1] < np.min(self.track[:-1]): # check if last epoch was best epoch
                self.best_bool = True
            else:
                self.best_bool = False
        return self.best_bool

    def update(self, error_rate):
        # append error rate
        self.track.append(error_rate)
        # get best condition
        self.update_best()
        # get patience
        self.update_patience()
        # get break condition
        self.update_break()