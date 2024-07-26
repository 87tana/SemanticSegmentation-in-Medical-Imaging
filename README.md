# Exploring CNN Components for Tumor Segmentation in MRI Images: An Ablation Study

This repository hosts the source code and resources for the project [Exploring CNN Components for Tumor Segmentation in MRI Images: An Ablation Study](https://medium.com/@t.mostafid/exploring-cnn-components-for-tumor-segmentation-in-mri-images-an-ablation-study-d79cdfd25083). 

<div align="center">
    <img width="700" src="/Images/ResNet-18 Segmentation Network.png" alt="Material Bread logo">
    <p style="text-align: center;">Figure 1: Proposed segmentation network,Created by author.</p>   
</div>

### Dataset

The dataset, sourced from [kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation), consists of 2146 images with tumors annotated in COCO Segmentation format. It's part of the TumorSeg Computer Vision Project, which focuses on Semantic Segmentation and aims to accurately identify tumor regions within Medical Images using advanced techniques

### Aim of study

The project focuses on evaluating different configurations of an encoder-decoder convolutional neural network (CNN) to identify the most effective techniques for tumor segmentation using 2D MRI brain tumor images.

<div align="center">
    <img width="700" src="/Images/ec_cnn_mri.png" alt="Material Bread logo">
    <p style="text-align: center;">Figure 2: CNN architecture for brain tumor segmentation task. Created by author.</p>   
</div>



#### Highlights:

**ResNet-18** showed better performance compared to VGG-16.
    
**Binary Cross Entropy** achieved slightly better training results than the Dice coefficient.

Incorporation of **dilated convolutions** in the decoder significantly enhanced segmentation accuracy.

**Data augmentation** was used to increase the model's robustness and generalizability.


### Experiments

To study the effects of network structure on segmentation results, I conducted ablation studies on different parts of my base network (see Figure 1). All experiments were performed on Google Colab with GPU acceleration, and the learning rate was adjusted based on the learning curve of each experiment.

I used standard metrics such as IoU, Dice, precision, and recall to evaluate the performance of each model. For an explanation of these metrics, please refer to this Medium article. I use a threshold of 0.5 to generate the prediction masks from the probability maps.


### Final experiment

In the final experiment, I incorporated the findings from the previous ablation studies. I used the baseline model with dilated convolutions as explained earlier, applied the BCE loss function, and randomly applied augmentation to 50% of the samples. Table 5 shows the segmentation results on the validation and test sets. Compared to Table 4, the test results show a significant improvement over the previous experiments.



### Results


<div align="center">
    <img width="800" src="/Images/final.png" alt="Material Bread logo">
    <p style="text-align: center;">Figure 3: CNN architecture for brain tumor segmentation task. Created by author.</p>   
</div>


<div align="center">
    <img width="800" src="/Images/table.png" alt="Material Bread logo">
    <p style="text-align: center;">Figure 4: CNN architecture for brain tumor segmentation task. Created by author.</p>   
</div>


###

  




 

