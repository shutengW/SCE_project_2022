import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import matplotlib
import sys


def visualize_data(data_to_visualize, time_to_visualize):
    """This function is to visualize the raw data.

    Args:
        data_to_visualize : Raw data to visualize.
        time_to_visualize : Raw time signals to visualize.
    """
    fig = plt.figure(num=1, figsize=(15, 15))

    locator_major_x = ticker.MultipleLocator(0.05)
    locator_minor_x = ticker.MultipleLocator(0.01)

    locator_major_y = ticker.MultipleLocator(2)
    locator_minor_y = ticker.MultipleLocator(0.5)

    grid = GridSpec(nrows=1, ncols=1)

    ax = []

    ax.append(fig.add_subplot(grid[0, 0]))

    # Load the first segment of the raw signals, which is set to have a length of 10000 points.
    ax[0].plot(time_to_visualize[0][0:10000, 0], data_to_visualize[0][0][0])

    fig.supxlabel('time [s]', y=0.05, fontsize=20)
    fig.supylabel('Acoustic amplitude [dB]', x=0.05, fontsize=20)

    for i in range(1):

        ax[i].set_title('Microphone ' + str(i + 1))
        ax[i].xaxis.set_major_locator(locator_major_x)
        ax[i].xaxis.set_minor_locator(locator_minor_x)

        ax[i].yaxis.set_major_locator(locator_major_y)
        ax[i].yaxis.set_minor_locator(locator_minor_y)

        ax[i].tick_params(axis='both', which='major', labelsize=25)

    plt.show()


if __name__ == '__main__':

    # Add the package to the python execution path
    sys.path.append(
        r'C:\Users\ASUS\Desktop\RWTH 1semester\RWTH 4th semester\SCE\SCE_package')

    from raw_signal_FFT_CWT_Visual import load_segmented_signals

    # Load the data and prediction labels
    data, time = load_segmented_signals.load_data_and_time()

    # Visualize the data
    visualize_data(data, time)
