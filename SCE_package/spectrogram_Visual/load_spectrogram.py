import pickle
import numpy as np


def load_data_and_pred(data_loc=r'D:\Master thesis\training_data_full', pred_loc=r'D:\Master thesis\training_pred_full'):
    """This function loads the data and the prediction labels.

    Args:
        data_loc : Location of the data files . Defaults to r'D:\Master thesis\training_data_full'.
        pred_loc : Location of the prediction label files. Defaults to r'D:\Master thesis\training_pred_full'.

    Returns:
        data, pred: Loaded data and prediction labels.
    """

    with open(pred_loc, 'rb') as input_file_pred:
        pred = pickle.load(input_file_pred)

    with open(data_loc, 'rb') as input_file_data:
        data = pickle.load(input_file_data)

    return data, pred
