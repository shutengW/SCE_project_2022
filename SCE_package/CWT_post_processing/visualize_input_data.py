import matplotlib.pyplot as plt


def visualize_input_data(data, range=10):

    plt.imshow(data[0, :, 0:range], aspect='auto', cmap=plt.get_cmap('jet'))
    plt.xlabel('Principle component number')
    plt.ylabel('Microphone number')
    plt.colorbar()
    plt.title('Wavelet spectrograms magnitude after PCA')
