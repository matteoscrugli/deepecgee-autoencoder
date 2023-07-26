# Autoencoder for ECG Data Analysis

This repository contains the Python code for a Autoencoder used for replicating electrocardiogram (ECG) signals containing anomalies. The model is trained on a collection of ECG signals and corresponding accelerometer data, and then used to generate simulated ECG signals from test data.

The objective here is to recreate ECG-like signals that include errors. This serves as a useful tool for testing and validating other ECG analysis algorithms and systems. 

## Table of Contents

- [Autoencoder for ECG Data Analysis](#autoencoder-for-ecg-data-analysis)
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Usage](#usage)
- [Running the Code](#running-the-code)
- [License](#license)

## Prerequisites

- Python 3
- TensorFlow
- Numpy
- Matplotlib
- json

## Overview

The script reads ECG and accelerometer data and converts them into appropriate input and output sets for the autoencoder. The autoencoder is then trained on this data using the Adam optimizer and Mean Absolute Error (MAE) loss. The trained model can then be used to generate reconstructed ECG signals from test data and evaluate the reconstruction error, which can be used as an indicator of anomalies in the ECG signal.

The model can handle both filtered and unfiltered accelerometer data across different axes. The ECG and accelerometer data are normalized before being fed to the autoencoder.

The script also includes functionality for splitting the data into training, validation, and test sets, and for saving the model weights and training history. It includes plots for visualizing the reconstructed signals against the original signals.

## Usage

1. **Reading Data**: ECG and accelerometer data is read and transformed into suitable input/output sets.
2. **Data Splitting**: Data is divided into training, validation, and testing sets.
3. **Data Normalization**: Data is normalized.
4. **Model Compilation and Training**: Anomaly Detector model (a Convolutional Autoencoder) is compiled and trained using the training data.
5. **Saving Model Weights**: Model weights are saved.
6. **Signal Reconstruction**: Test data is run through the autoencoder to generate reconstructed signals.
7. **Error Computation and Analysis**: Reconstruction errors are computed and analyzed for anomaly detection.
8. **Plotting**: Original signals, reconstructed signals, and error signals are plotted for visual inspection.

## Running the Code

To run the code, navigate to the directory where the Python script is located using your terminal or command prompt, then type the following command:

```bash
python3 autoencoder_cnn.py
```

Please note, the code expects certain data formats and file paths, which may need to be adjusted based on your specific setup. Also, you may need to tune hyperparameters such as the number of epochs, batch size, and learning rate based on your specific dataset and training requirements.

## License

This project is licensed under the terms of the MIT license.
