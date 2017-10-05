import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def write_log(logfile, log_dict):
    """
    Writes a '.csv' logfile
    :param logfile (str): output filename
    :param log_dict (ordered dictionary): dictionary that contains parameters to log
    :return: None
    """
    with open(logfile, 'a') as f:
        c = csv.writer(f)
        if log_dict['epoch'] == 0:  # write header for first epoch (dubbed as 0th epoch)
            c.writerow(log_dict.keys())

        c.writerow(log_dict.values())


def show_log(logfile, category, selection, x_mode, y_mode):
    """
    Takes a logfile and creates a plot of the desired log parameter over epochs
    :param logfile (str): .csv logfile 
    :param category (str): select the column that defines the plot legend 
    :param selection (list of ints): select the runs to plot 
    :param x_mode (str): default is epoch, do not change 
    :param y_mode (str): select the parameter to plot over epochs, e.g. WER
    :return: (fig, ax) matplotlib objects
    """
    cat_dict = dict()
    cat = 0
    with open(logfile, 'rt') as f:
        f = csv.reader(f)
        cat_rows = []
        for idx, row in enumerate(f):
            if (idx != 0) and row[0] == 'epoch':
                cat_dict['{}'.format(cat)] = np.asarray(cat_rows)
                cat = cat + 1
                cat_rows = []
            cat_rows.append(row)
        cat_dict['{}'.format(cat)] = np.asarray(cat_rows)

    fig, ax = plt.subplots()
    min_vals = []
    for key in selection:
        key = str(key)
        cat_idx = np.where(cat_dict[key][0] == category)[0][0]
        cat_name = cat_dict[key][1, cat_idx].astype(str)

        if x_mode == 'epoch':
            x_values = cat_dict[key][1:, 0].astype(int)
            unit = ' #'
        y_column = np.where(cat_dict[key][0] == y_mode)[0][0]

        y_values = cat_dict[key][1:, y_column].astype(float)
        min_vals.append(np.min(y_values))

        ax.plot(x_values, y_values, label=cat_name, lw=2.0)
        ax.set_ylabel(y_mode)
        ax.set_xlabel(x_mode + unit)

    plt.grid()
    plt.legend()
    return fig, ax