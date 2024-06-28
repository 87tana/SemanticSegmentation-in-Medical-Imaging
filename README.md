## segmentation in medical imaging

<div align="center">
    <img width="800" src="/Images/ec_cnn_mri.png" alt="Material Bread logo">
    <p style="text-align: center;">Figure 1: CNN architecture for brain tumor segmentation task. Created by author.</p>   
</div>


<div align="center">
    <img width="800" src="/Images/ec_cnn_mri.png" alt="Material Bread logo">
    <p style="text-align: center;">Figure 1: CNN architecture for brain tumor segmentation task. Created by author.</p>   
</div>

* ### Overview

This repository contains the code and results for a series of experiments conducted to identify the optimal configuration for semantic segmentation tasks. The experiments involve various configurations of deep learning models, focusing on performance metrics such as Intersection over Union (IoU), Accuracy (ACC), Dice coefficient, Precision (Prec), and Recall.

* ### Dataset

The dataset, sourced from [kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation), consists of 2146 images with tumors annotated in COCO Segmentation format. It's part of the TumorSeg Computer Vision Project, which focuses on Semantic Segmentation and aims to accurately identify tumor regions within Medical Images using advanced techniques

* ### Data Augumentation

The basic forms of data augmentation are used here to diversify the training data. All the augmentation methods are used from Pytorch's Torchvision module.

* ### Model Training
 
* ### Model Evaluation:
  
<div align="center">
    <img width="700" src="/Plots/Exp25.png" alt="Material Bread logo">
    <p style="text-align: center;">Photo created by autor</p> 
</div>

* ### Conclusion and Future work:
  


* ### References

[U-Net Convolutional Network](https://arxiv.org/pdf/1505.04597.pdf ) 
[DeepLabv3](https://arxiv.org/pdf/1706.05587) 

 



  




 

