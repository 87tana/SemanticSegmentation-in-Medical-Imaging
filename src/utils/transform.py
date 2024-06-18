#import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms



class deafault_transform():
  def __init__(self, outputsize=None):

    self.outputsize=outputsize
    # Define the mean and standard deviation for normalization(based on image net)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    self.imgtrans = transforms.Compose([
    transforms.ToTensor(), # converting image data from a numpy array or PIL image format to a PyTorch tensor format.
    transforms.Normalize(mean=mean, std=std) # normalize the image
    ])

    
    self.masktrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((self.outputsize, self.outputsize)), # convert mask to PyTorch tensor
    ])

  def trans(self, img, mask):

    img = self.imgtrans(img)
    mask = self.masktrans(mask)

    return img, mask 



class augmentation(deafault_transform):
  def __init__(self, outputsize, auglist=None):
    super().__init__(outputsize)
  
    self.affine = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5), # Randomly flip the image horizontally
      transforms.RandomVerticalFlip(p=0.5), # Randomly flip the image vertically
      transforms.RandomRotation(degrees=90), # Randomly rotate the image
      ])

    self.imgaug =  transform_img = transforms.Compose([
      transforms.ColorJitter(brightness=0.1), # Apply random brightness augmentation
       ])
      

  def augment(self, img, mask):

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