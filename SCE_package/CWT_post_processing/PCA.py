import numpy as np
import numpy as np
from sklearn import decomposition


def data_crop(data, start_index=0, end_index=100):
    """This function is to crop important parts of the data given with the start_index and end_index specifying the range.

    Args:
        data: Input data. 
        start_index : The start index specifies the starting point for the cropped data. Defaults to 0.
        end_index : The end index specifies the end point for the cropped data. Defaults to 100.

    Returns:
        data, start_index, end_index: data is the cropped data, start_index and end_index are the new start and end index.
    """
    data = data[:, :, :, start_index:end_index]

    return data, start_index, end_index


def PCA_fit_transform(data, pred, start_index, end_index, n_components=100):
    """This function performs the principal component analysis on the data and returns the transformed data.

    Args:
        data: Cropped data from the raw data.
        pred: The prediction labels.
        start_index: This is the start index for the cropped data.
        end_index: This is the end index for the cropped data.
        n_components (int, optional): Number of components used for principal component analysis. Defaults to 100.

    Returns:
        data, pred: Post-PCA data and prediction labels.
    """
    data_PCA_test = np.reshape(
        data, (426 * 112, 45 * (end_index - start_index)))

    pca_test = decomposition.PCA(n_components=n_components)

    # Use PCA to fit the data
    pca_test.fit(data_PCA_test)

    # Print the sum of the variance ratio that can be explained using this PCA techinque.
    print(np.sum(pca_test.explained_variance_ratio_))

    # Transform the data
    data_after_PCA = pca_test.transform(data_PCA_test)

    data = np.reshape(data_after_PCA, (426, 112, n_components))

    return data, pred
