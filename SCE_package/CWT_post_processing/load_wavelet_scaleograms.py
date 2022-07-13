import pickle
import numpy as np


def load_data_and_pred(data_loc=r'D:\Master thesis\wavelet_data\training_data_full_wavelet', pred_loc=r'D:\Master thesis\wavelet_data\pred_values_after_segmentation'):
    """This function is used to load the data and predictions.

    Args:
        data_loc : The path location of data. Defaults to r'D:\Master thesis\wavelet_data\training_data_full_wavelet'.
        pred_loc : The path location of prediction labels. Defaults to r'D:\Master thesis\wavelet_data\pred_values_after_segmentation'.

    Returns:
        data, pred: Loaded data and prediction labels.
    """

    with open(data_loc, 'rb') as input_file_data:
        data = pickle.load(input_file_data)

    with open(pred_loc, 'rb') as input_file_pred:
        pred = pickle.load(input_file_pred)

    pred = np.array(pred)

    return data, pred
