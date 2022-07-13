import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import sys


def train_test_splits(data, pred):
    """Split the train and test data from the original dataset.

    Args:
        data : The original dataset to be split.
        pred : The original prediction labels to be split.

    Returns:
        data, pred, validation_data, validation_pred: The split training data, training labels, validation data, validation labels.
    """

    # Find the indexes of those manually selected coordinates from the dataset.
    index = np.where(np.all(pred == (0, 0), axis=1))[0]
    index = np.append(index, np.where(np.all(pred == (3, 0), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (6, 0), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (9, 0), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (0, 3), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (0, 6), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (0, 9), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (3, 3), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (3, 6), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (3, 9), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (6, 3), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (6, 6), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (6, 9), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (9, 3), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (9, 6), axis=1))[0])
    index = np.append(index, np.where(np.all(pred == (9, 9), axis=1))[0])
    print(index)

    # This selects validation data and its corresponding labels.
    validation_data = data[index]
    validation_pred = pred[index]

    print(np.shape(validation_data))
    print(np.shape(validation_pred))

    # This deletes the validation data and its corresponding labels from the original dataset.
    pred = np.delete(pred, index, axis=0)
    data = np.delete(data, index, axis=0)

    print(np.shape(pred))
    print(np.shape(data))

    return data, pred, validation_data, validation_pred


def visualize_train_test_split(pred, validation_pred):
    """This function visualizes the train and test data split by plotting their coordinates.

    Args:
        pred : Prediction labels in the training dataset.
        validation_pred : Prediction labels in the validation dataset.
    """
    fig, axes = plt.subplots(1, figsize=(5, 4))

    axes.scatter(validation_pred[:, 0],
                 validation_pred[:, 1], s=2, c='r', marker='x')
    axes.scatter(pred[:, 0], pred[:, 1], color='none',
                 edgecolor='blue', marker='o')
    axes.set_xlabel('X coordinates [cm]')
    axes.set_ylabel('Y coordinates [cm]')
    axes.legend([
        'training data', 'test data'], bbox_to_anchor=(0.9, 0.5, 0.5, 0.5))

    locator_major_x = ticker.MultipleLocator(1)
    locator_major_y = ticker.MultipleLocator(1)

    axes.xaxis.set_major_locator(locator_major_x)
    axes.yaxis.set_major_locator(locator_major_y)

    plt.show()
