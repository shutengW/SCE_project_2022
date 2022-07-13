import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import matplotlib
from scipy.fftpack import rfft
from scipy.fftpack import rfftfreq
import sys


def visualize_data(data_to_visualize):
    """This function is to visualize the data.

    Args:
        data_to_visualize : Data to be visualized for fast fourier transform features.
    """

    # Sample rate is the frequency for the data to be sampled for the microphone.
    sample_rate = 48000

    # Perform the real fast fourier transform on the data.
    xf = rfftfreq(n=10000, d=1 / sample_rate)

    fft_res_1 = (rfft(data_to_visualize[0][1][0]))

    fig = plt.figure(num=2, figsize=(15, 15))

    locator_major_x = ticker.MultipleLocator(5000)
    locator_minor_x = ticker.MultipleLocator(1000)

    locator_major_y = ticker.MultipleLocator(100)
    locator_minor_y = ticker.MultipleLocator(10)

    grid = GridSpec(nrows=1, ncols=1)

    ax = []

    ax.append(fig.add_subplot(grid[0, 0]))

    ax[0].plot(xf, np.abs(fft_res_1))

    fig.supxlabel('Frequency [Hz]', y=0.05, fontsize=20)
    fig.supylabel('Frequency amplitude', x=0.05, fontsize=20)

    for i in range(1):

        ax[i].set_title('Microphone ' + str(i + 1))

        ax[i].xaxis.set_major_locator(locator_major_x)
        ax[i].xaxis.set_minor_locator(locator_minor_x)

        ax[i].yaxis.set_major_locator(locator_major_y)
        ax[i].yaxis.set_minor_locator(locator_minor_y)

        ax[i].tick_params(axis='both', which='major', labelsize=25)

        # Draw a vertical line at frequency = 11 KHz.
        ax[i].axvline(x=11000, color='r')

        ax[i].legend(['Data', 'Separation Line'],
                     loc='upper right', fontsize=25)

    plt.show()


if __name__ == '__main__':

    # Add the package to the python execution path
    sys.path.append(
        r'C:\Users\ASUS\Desktop\RWTH 1semester\RWTH 4th semester\SCE\SCE_package')

    from raw_signal_FFT_CWT_Visual import load_segmented_signals

    # Load the data and prediction labels
    data, time = load_segmented_signals.load_data_and_time()

    # Visualize the data
    visualize_data(data)
