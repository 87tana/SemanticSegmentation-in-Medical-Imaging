# U-Net: medical imaging segmentation   ## Under construction :)

## objective: 

Manual tumour segmentation in MRI scans is time-consuming. However, using a learning algorithm for automatic segmentation not only saves radiologists time, but also improves surgical planning. UNet, the driving force behind this efficiency, is a powerful tool for medical image segmentation.

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

U-Net is a convolutional NN architecture widely used for tasks like image segmentation.  The shape reflects its method, This U-shaped structure is characterized by the initial contraction, followed by expansion, mirroring the process of reducing and then gradually restoring the size of the image.

The contracting path involves successive layers of convolutional operations, each followed by a pooling layer. These pooling layers serve to reduce the spatial dimensions (width and height) of the feature maps while increasing their depth, effectively capturing and abstracting features from the input image. This process creates a hierarchical representation of the input, enabling the network to extract increasingly abstract and high-level features.

Following the contracting path, the network enters the expansive path, which consists of upsampling layers. These upsampling layers gradually restore the spatial dimensions of the feature maps while reducing their depth, allowing the network to recover spatial details lost during the contraction stage. Additionally, skip connections are introduced between corresponding layers in the contracting and expansive paths. These skip connections facilitate the flow of information across different scales of abstraction, aiding in the precise localization of objects in the final segmentation mask.

 

