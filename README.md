# U-Net: medical imaging segmentation   ## Under construction :)

## Introduction: 

Here, I have implemented a U-Net to segment tumor in MRI images of brain.
using a learning algorithm for automatic segmentation not only saves radiologists time, but also improves surgical planning. 

## Dataset feature Extraction and DataFrame creation
Data set https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/data

This repository include a Python script that facilitates the extraction of important features from JSON files containing annotations and images in a dataset. The extracted features are used to create a structured DataFrame for easier data manipulation and analysis.

Extracted essential information (Features ) needed for tumor segmentation task 

The extracted features include:

## The UNet Architecture

U-Net is a convolutional NN architecture widely used for tasks like image segmentation. The U-Net was originally developed for biomedical image segmentation.The shape reflects its method, it is a fully convolutional network consisting of an **encoder** that learns to **downsample the input image** while preserving spatial information, and a **decoder** that learns to **upsample** the feature maps to the original image size while incorporating the encoder features.



 

