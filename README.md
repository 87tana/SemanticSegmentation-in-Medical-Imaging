# segmentation in medical imaging

<div align="center">
    <img width="800" src="/Images/ec_cnn_mri.png" alt="Material Bread logo">
    <p style="text-align: center;">Photo created by autor</p> 
</div>

* ### Overview

This repository contains the code and results for a series of experiments conducted to identify the optimal configuration for semantic segmentation tasks. The experiments involve various configurations of deep learning models, focusing on performance metrics such as Intersection over Union (IoU), Accuracy (ACC), Dice coefficient, Precision (Prec), and Recall.

* ### Dataset

The dataset was obtained from [kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation), compromise a single class with two category IDs, 1 and 2. Despite this distinction, both categories belong to the same class.

* ### Data Augumentation

The basic forms of data augmentation are used here to diversify the training data. All the augmentation methods are used from Pytorch's Torchvision module.
* ### Loss Function

* ### Experiment with different backbone


* ### Training Process and Validation loss
  
<div align="center">
    <img width="700" src="/Plots/Exp18.png" alt="Material Bread logo">
    <p style="text-align: center;">Photo created by autor</p> 
</div>


* ### References

[U-Net Convolutional Network](https://arxiv.org/pdf/1505.04597.pdf ) 

[DeepLabv3](https://arxiv.org/pdf/1706.05587) 

 



  




 

