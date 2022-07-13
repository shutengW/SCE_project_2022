import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import matplotlib
import sys


def normalize_data(data):
    """This is to perform the normalization on data

    Args:
        data : Input data. In this case, this is the spectrogram data saved.

    Returns:
        data_to_visualize : Normalized data that is ready to be visualized.
    """

    data_to_visualize = data[0]
    data_to_visualize = np.reshape(data_to_visualize, (21 * 22, 112))
    normalized_data = normalize(data_to_visualize, axis=0)
    data_to_visualize = np.reshape(normalized_data, (21, 22, 112))

    return data_to_visualize


def visualize_data(data_to_visualize):
    """This function is to visualize the data

    Args:
        data_to_visualize : Normalized data that is to be visualized.
    """

    fig = plt.figure(num=2, figsize=(15, 15))

    locator_major_x = ticker.MultipleLocator(5)
    locator_minor_x = ticker.MultipleLocator(1)

    locator_major_y = ticker.MultipleLocator(5)
    locator_minor_y = ticker.MultipleLocator(1)

    grid = GridSpec(nrows=1, ncols=1)

    ax = []

    ax.append(fig.add_subplot(grid[0, 0]))

    im = ax[0].imshow(data_to_visualize[:, :, 0],
                      cmap=plt.get_cmap('jet'))

    fig.supxlabel('Local temporal domain index', y=0.05, fontsize=30)
    fig.supylabel('Selected Frequncy index', x=0.05, fontsize=30)

    for i in range(1):

        ax[i].set_title('Microphone ' + str(i + 1), fontsize=40)
        ax[i].xaxis.set_major_locator(locator_major_x)
        ax[i].xaxis.set_minor_locator(locator_minor_x)

        ax[i].yaxis.set_major_locator(locator_major_y)
        ax[i].yaxis.set_minor_locator(locator_minor_y)

        ax[i].tick_params(axis='both', which='major', labelsize=25)

    cbar_ax = fig.add_axes([0.95, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=25)

    plt.show()


if __name__ == '__main__':

    # Add the package to the python execution path
    sys.path.append(
        r'C:\Users\ASUS\Desktop\RWTH 1semester\RWTH 4th semester\SCE\SCE_package')

    from spectrogram_Visual import load_spectrogram

    # Load the data and prediction labels
    data, pred = load_spectrogram.load_data_and_pred()

    # Normalize the data
    data_to_visualize = normalize_data(data)

    # Visualize the data
    visualize_data(data_to_visualize)
