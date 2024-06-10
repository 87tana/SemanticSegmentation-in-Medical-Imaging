# segmentation in medical imaging

<div align="center">
    <img width="700" src="/Images/Grandtruth.png" alt="Material Bread logo">
    <p style="text-align: center;">Photo created by autor</p> 
</div>


In this project, the U-Net architecture was implemented to enhance the segmentation of brain MRI images. Leveraging the computational power of GPUs and utilizing the PyTorch framework, the model training process was significantly accelerated. Various image augmentation techniques were employed to improve the model's robustness and generalizability. Additionally, a systematic approach was adopted to tune hyperparameters, optimizing the model's performance.



* ### Dataset

The dataset was obtained from [kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation), compromise a single class with two category IDs, 1 and 2. Despite this distinction, both categories belong to the same class.

* ### Data Augumentation

The basic forms of data augmentation are used here to diversify the training data. All the augmentation methods are used from Pytorch's Torchvision module.

    Apply random brightness augmentation
    Rotation Between 75°-15°

* ### Model Architechture

<p align="center">
    <img width="600" src="Images/U-Net_Architecture.png" alt="Material Bread logo">
</p>

* ### Training Process


[U-Net Convolutional Network](https://arxiv.org/pdf/1505.04597.pdf ) 

 



  




 

