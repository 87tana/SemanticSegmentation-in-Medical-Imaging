import os
from tqdm import tqdm
import random
from src.utils.utils import create_mask
from src.utils.transform import default_transform, augmentation

import cv2



"""
Dataset class. Read images and create an iterator used in the dataloader.
"""
class SegmentationDataset():
    def __init__(self, root_dir, df, subset, outputsize, max_samp=None, augment=False, aug_rate=0.5):
        self.root_dir = root_dir
        self. augment = augment
        self.aug_rate = aug_rate
        
        if self.augment:
          self.transform = augmentation(outputsize)
        else:
          self.transform = default_transform(outputsize)


        if max_samp is None:
          max_samp = len(df)

        self.data_ = []  # Loading image-mask pairs from the dataset into memory and storing them in a list called self.data_.

        for i in tqdm(range(max_samp), desc=subset):
          img = cv2.imread(os.path.join(root_dir,subset,df.loc[i]['file_name']))
         # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          if img is not None:
            mask = create_mask(df.loc[i]) # Create mask
            if mask is not None:
              self.data_.append({'image': img, 'mask': mask}) # add image and its mask to a list.
            else:
              print('\nCannot create mask for ', df.loc[i]['file_name'])
          else:
            print('\nCannot read ', df.loc[i]['file_name'])


    def __len__(self):
        return len(self.data_)

    def __getitem__(self, idx):

        image_orig = self.data_[idx]['image']
        mask_orig = self.data_[idx]['mask']


        if self.augment and random.random() < self.aug_rate:
          image, mask = self.transform.augment(image_orig.copy(), mask_orig.copy())
        else:
          image, mask = self.transform.trans(image_orig, mask_orig)

        mask[mask>0] = 1.0
        mask_orig[mask_orig>0] = 1.0

        """
        img: transformed image
        msk: tarnsformed mask
        image: original image
        mask: original mask
        """
        
        return image, mask, image_orig, mask_orig


