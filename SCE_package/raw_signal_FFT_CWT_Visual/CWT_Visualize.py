import sys
import numpy as np
import matplotlib.pyplot as plt
import scaleogram


def visualize_data(data_to_visualize, time_to_visualize):
    """This function is to visualize the data, the continuous wavelet transform features.

    Args:
        data_to_visualize : Data to be visualized for continuous wavelet transform features.
        time_to_visualize : Data to be visualized for continuous wavelet transform features.
    """

    # This function plots parts of the continuous wavelet transform scaleograms.
    ax = scaleogram.CWT(time_to_visualize[0][100:500, 0], data_to_visualize[10]
                        [0][0][100:500], scales=np.arange(5, 50), wavelet='cmor1-1.5')

    scaleogram.cws(ax,
                   figsize=(14, 3), cmap='jet', yaxis='frequency',
                   ylabel='Frequency [Hz]', xlabel='Time [s]', coi=False)

    plt.show()


if __name__ == '__main__':

    # Add the package to the python execution path
    sys.path.append(
        r'C:\Users\ASUS\Desktop\RWTH 1semester\RWTH 4th semester\SCE\SCE_package')

    from raw_signal_FFT_CWT_Visual import load_segmented_signals

    # Load the data and time signals
    data, time = load_segmented_signals.load_data_and_time()

    # Visualize the data
    visualize_data(data, time)
