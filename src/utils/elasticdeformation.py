import torchvision.transforms.v2 as transforms
import numpy as np
import torch
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image

class default_transform():
    def __init__(self, outputsize=None):
        self.outputsize = outputsize
        # Define the mean and standard deviation for normalization(based on image net)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.imgtrans = transforms.Compose([
            transforms.ToTensor(),  # converting image data from a numpy array or PIL image format to a PyTorch tensor format.
            transforms.Normalize(mean=mean, std=std)  # normalize the image
        ])

        self.masktrans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.outputsize, self.outputsize)),  # convert mask to PyTorch tensor
        ])

    def trans(self, img, mask):
        img = self.imgtrans(img)
        mask = self.masktrans(mask)
        return img, mask 

class augmentation(default_transform):
    def __init__(self, outputsize, auglist=None):
        super().__init__(outputsize)
    
        self.affine = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(p=0.5),  # Randomly flip the image vertically
            transforms.RandomRotation(degrees=90),  # Randomly rotate the image
        ])

        self.imgaug = transforms.Compose([
            transforms.ColorJitter(brightness=0.1),  # Apply random brightness augmentation
        ])

    
    def elastic_transform(self, image_, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in Simard, Steinkraus and Platt, 'Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis', in Proc. of the International
        Conference on Document Analysis and Recognition, 2003.
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image_.shape
        #dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        #dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)
        
        # Image
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        distored_image = map_coordinates(image_, indices, order=1, mode='reflect')
        
        
        return distored_image.reshape(shape)

    
    def augment(self, img, mask):
        # Apply affine transformations
        img, mask = self.affine(img, mask)
        img = self.imgaug(img)

        # Convert images to numpy arrays for elastic transformation
        img_np = np.array(img)
        mask_np = np.array(mask)

        # Apply elastic transformation
        #alpha = img_np.shape[1] * 20
        #sigma = img_np.shape[1] * 0.08
        #random_state = np.random.RandomState(None)
        #img_np = self.elastic_transform(img_np, alpha, sigma, random_state)
        #mask_np = self.elastic_transform(mask_np, alpha, sigma, random_state)

        # Convert back to PIL image
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        mask = Image.fromarray((mask_np * 255).astype(np.uint8))

        #img, mask = self.trans(img, mask)
        return img, mask
 