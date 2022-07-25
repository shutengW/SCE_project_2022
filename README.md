# Sustainable Computational Engineerng Project 2022

This folder is created to satisfy the passing requirement of the course "Sustainable Computational Engineering" at RWTH Aachen University. The following gives the structure of the package containing python files of the project.

## Python package structure
```
├── CWT_post_processing
│   ├── data_normalize.py
│   ├── load_wavelet_scaleograms.py
│   ├── Network.py
│   ├── PCA.py
│   ├── train_test_splits.py
│   ├── visualize_input_data.py
│   ├── __init__.py
│   └── __pycache__
│       ├── load_wavelet_scaleograms.cpython-39.pyc
│       ├── PCA.cpython-39.pyc
│       └── __init__.cpython-39.pyc
├── draw_tree.py
├── raw_signal_FFT_CWT_Visual
│   ├── CWT_Visualize.py
│   ├── FFT_visualize.py
│   ├── load_segmented_signals.py
│   ├── raw_signal_visualize.py
│   ├── __init__.py
│   └── __pycache__
│       ├── CWT_Visualize.cpython-39.pyc
│       ├── FFT_visualize.cpython-39.pyc
│       ├── load_segmented_signals.cpython-39.pyc
│       ├── raw_signal_visualize.cpython-39.pyc
│       └── __init__.cpython-39.pyc
├── spectrogram_Visual
│   ├── load_spectrogram.py
│   ├── Main_pectrogram_visualize.py
│       ├── load_spectrogram.cpython-39.pyc
│       ├── Spectrogram_visualize.cpython-39.pyc
│       └── __init__.cpython-39.pyc
└── __init__.py
```
## Introduction on the structure of this package
In this package, it provides different feature visualization utilities. Spectrogram_Visual is used to visualize spectrograms produced by short-time Fourier transforms. raw_signal_FFT_CWT_Visual is used to visualize fast fourier transforms, continuous wavelet transforms etc. CWT_post_processing is to use the CWT features as input features, and use these features for further processing.

## Comment on the reproducibility of the results
Since the datapath is set according to where the data is put and this is different for every one, it might need to be manually adjusted to run the code. Click [here](https://gigamove.rwth-aachen.de/en/download/4f94039c1e0896892c073dcd5dc2d9b6) to download the data. If the data is not reachable anymore, please contact the author at:
shuteng.wang@rwth-aachen.de

## Install
```
git clone https://github.com/shutengW/SCE_project_2022.git
cd SCE_package/CWT_post_processing
conda env create -f SCE_environment.yml
```

## Implement
```
python3 Network.py
```
