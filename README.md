# U-Net: An effective approach to medical imaging segmentation

## Under construction :)

##Objective: Performing segmentation analysis on magnetic resonance images of brain tumours.

## Dataset feature Extraction and DataFrame creation
Data set https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/data

This repository include a Python script that facilitates the extraction of important features from JSON files containing annotations and images in a dataset. The extracted features are used to create a structured DataFrame for easier data manipulation and analysis.

Extracted essential information (Features ) needed for tumor segmentation task 

The extracted features include:

    File names of images
    Dimensions (width and height) of images
    Category IDs associated with annotations
    Segmentation data for annotations
    Subset information (train, valid, test)

        file_name: This column stores the names of the image files, which are essential for identifying and retrieving the corresponding images during further analysis.

    width and height: These columns represent the dimensions of the images. In the example you provided, all images have the same size (640x640 pixels). While it's true that the width and height are constant in this case, it's still beneficial to include these columns for completeness and consistency in the dataset. Additionally, if your dataset contains images of varying sizes in the future, these columns will be essential for understanding the data distribution and potentially for preprocessing steps.

    category_id: This column indicates the category or class associated with each annotation. In your example, all annotations have a category ID of 1, suggesting that they belong to the same class.

    segmentation: This column stores the segmentation data for each annotation. Segmentation data typically consists of pixel coordinates delineating the boundaries of the segmented region within the image. It's a crucial component for training and evaluating segmentation algorithms.

    subset: This column indicates the subset of the dataset to which each image belongs (e.g., train, valid, test). It's useful for partitioning the data for training, validation, and testing purposes.

## The UNet Architecture
