import pickle
import numpy as np


def load_data_and_time(data_signal=r'D:\Master thesis\wavelet_data\segmented_acoustic_signal', time_signal=r'D:\raw_time_signal'):
    """This function loads the raw acoustic data and the time signals.


    Args:
        data_signal : Location of the data signal files. Defaults to r'D:\Master thesis\wavelet_data\segmented_acoustic_signal'.
        time_signal : Location of the time signal files. Defaults to r'D:\raw_time_signal'.

    Returns:
        data, time: Loaded data and time signals.
    """

    with open(time_signal, 'rb') as input_file_time:
        time = pickle.load(input_file_time)

    with open(data_signal, 'rb') as input_file_data:
        data = pickle.load(input_file_data)

    return data, time
