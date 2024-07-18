#import torchvision.transforms as transforms
from torchvision.transforms import v2
import numpy as np
import torch
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image


class default_transform():
  def __init__(self, outputsize=None):

    self.outputsize=outputsize
    # Define the mean and standard deviation for normalization(based on image net)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    self.imgtrans = v2.Compose([
    v2.ToTensor(), # converting image data from a numpy array or PIL image format to a PyTorch tensor format.
    v2.Normalize(mean=mean, std=std) # normalize the image
    ])

    
    self.masktrans = v2.Compose([
        v2.ToTensor(),
        v2.Resize((self.outputsize, self.outputsize)), # convert mask to PyTorch tensor
    ])

  def trans(self, img, mask):

    img = self.imgtrans(img)
    mask = self.masktrans(mask)

    return img, mask 



class augmentation(default_transform):
  def __init__(self, outputsize, auglist=None):
    super().__init__(outputsize)
  
    self.affine = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                        v2.RandomVerticalFlip(p=0.5),
                        v2.RandomAffine(degrees=(10, 70), translate=(0.2, 0.2), shear=(-10,10,-10,10), scale=(0.8, 1.2)),
                        ])

    self.imgaug = v2.Compose([v2.ElasticTransform(alpha=10, sigma=0.08),
                                v2.ColorJitter(brightness=(0.1,1.0), contrast=(0.7,1.0), saturation=(0.7,1.0)) # Apply random brightness augmentation
                              ])
      

  def augment(self, img, mask):

    img = Image.fromarray(img)
    mask = Image.fromarray(mask)     

    img, mask = self.affine(img, mask)
    img = self.imgaug(img)

    img, mask = self.trans(img, mask)
    


    return img, mask







  #transform_img = transforms.Compose([
      #transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
      #transforms.RandomVerticalFlip(), # Randomly flip the image vertically
      #transforms.RandomRotation(degrees=30), # Randomly rotate the image
      #transforms.ColorJitter(brightness=0.1), # Apply random brightness augmentation
      #transforms.ToTensor(), # Convert image data to a PyTorch tensor format
      #transforms.Normalize(mean=mean, std=std) # Normalize the image
  #])

  #transform_mask = transforms.Compose([
      #transforms.RandomHorizontalFlip(), # Randomly flip the mask horizontally
      #transforms.RandomVerticalFlip(), # Randomly flip the mask vertically
      #transforms.RandomRotation(degrees=30), # Randomly rotate the mask
      #transforms.ToTensor(), # Convert mask to a PyTorch tensor
      #transforms.Resize((320, 320)), # Resize the mask
  #])




