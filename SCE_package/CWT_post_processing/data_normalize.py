import numpy as np


def normalize(data):
    """This is to perform the normalization on data

    Args:
        data : Input data.

    Returns:
        data: Normalized data that is ready to be processed later.
    """
    for i in range(len(data)):

        data_i = data[i]
        transform_1d_data = np.reshape(data_i, newshape=(112, 100))
        normalized_data = normalize(transform_1d_data, axis=1)
        data_N = np.reshape(normalized_data, newshape=(112, 100))
        data[i] = data_N

    return data
