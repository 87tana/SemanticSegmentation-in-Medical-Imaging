
import os
import json  # JSON serialization and deserialization
import pandas as pd
import numpy as np
import skimage



def read_json(root_dir):
  """
  Read JSON files and extract image and annotation information.
  Store them in lists and then create a dictionary.
  Finally create a pandas dataframe using the dctionary.
  """

  file_name = []
  width = []
  height = []
  category_id = []
  segmentation = []
  subset = []

  for s in ['train', 'valid', 'test']:
    with open(os.path.join(root_dir,s,'_annotations.coco.json'),'r') as file:
      data = json.load(file)
      for id in range(len(data['images'])):
        # Flag to check if the annotation is available
        ann_available=False
        # Look for the corresponding annotation
        for j in range(id,len(data['images'])):
          if data['annotations'][j]['image_id'] == data['images'][id]['id']: # if the aanotation is found,extracted the 'category_id' and 'segmentation' from the annotation and append.
            category_id.append(data['annotations'][j]['category_id'])
            segmentation.append(data['annotations'][j]['segmentation'])
            ann_available=True
            break

        if ann_available:
          file_name.append(data['images'][id]['file_name'])
          width.append(data['images'][id]['width'])
          height.append(data['images'][id]['height'])
          subset.append(s)
        else:
          print(f'Annotation is missing for image {id} in {s}')


  dic = {'file_name': file_name, 'width': width , 'height': height,\
          'category_id': category_id, 'segmentation': segmentation, 'subset': subset}

  df = pd.DataFrame.from_dict(dic)
  
  return df



def create_mask(image_info):

  """
  The function aims to create a binary mask based on the segmentation information provided.
  It iterates over each segmentation entry in the image_info['segmentation'] list.
  it extracts the polygon coordinates from the segmentation list.
  pixels inside the segmented regions are set to 1 and pixels outside are set to 0.
  """
  mask_np = np.zeros((image_info['height'],image_info['width']),dtype=np.uint8)

  for seg_idx ,seg in enumerate(image_info['segmentation']):
      rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
      mask_np[rr,cc]=1

  return mask_np


  
  



