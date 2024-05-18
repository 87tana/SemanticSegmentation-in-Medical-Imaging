# U-Net: segmentation in medical imaging

## Overview:

* ### Dataset

The dataset was obtained from [kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation), compromise a single class with two category IDs, 1 and 2. Despite this distinction, both categories belong to the same class.

* ### Data Augumentation

The basic forms of data augmentation are used here to diversify the training data. All the augmentation methods are used from Pytorch's Torchvision module.

    Apply random brightness augmentation
    Rotation Between 75°-15°

* ### Model Architechture 

* ### Training Process

In this project, the U-Net architecture was implemented to enhance the segmentation of brain MRI images. Leveraging the computational power of GPUs and utilizing the PyTorch framework, the model training process was significantly accelerated. Various image augmentation techniques were employed to improve the model's robustness and generalizability. Additionally, a systematic approach was adopted to tune hyperparameters, optimizing the model's performance.



<div align="center">
    <img width="600" src= "/Images//U-net_example_wikipedia.png" alt="Material Bread logo">   
    <p   style="text-align: center;"> Photo from Wikipedia </p> 
</div>



The TumorSeg Computer Vision Project is dedicated to Semantic Segmentation, which involves classifying every pixel in an image as part of a tumor or non-tumor region. This fine-grained approach provides an accurate understanding of the spatial distribution of tumors within medical images.

Utilizing the powerful [U-Net Convolutional Network](https://arxiv.org/pdf/1505.04597.pdf ) implemented in Pytorch,  our project tackles the challenging task of brain tumor segmentation. Training and testing are conducted on the Google Colab platform, leveraging its GPU capabilities for efficient computation.

 

The distributions for each subset train, validation, and test: 

<div align="center">
    <img width="300" src="/Images/seg_subset_distribution.png" alt="Material Bread logo">   
</div>


  




 

