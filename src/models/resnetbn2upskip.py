import torch
import torch.nn as nn  # Neural network modules and loss functions
import torchvision.models as models # provide access to pre-trained models



class ResNet2upskip(nn.Module):
    def __init__(self, num_classes=1, out_channels=1,):
        super(ResNet2upskip, self).__init__()

        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)  # import torchvision

        # Remove fully connected layers (classifier) from ResNet
        self.encoder_early = nn.Sequential(*list(resnet.children())[:-3]) #remove two last layer,
       
        self.encoder_late = resnet.layer4

        # Decoder with upsampling (interpolation)
        self.decoder_early = nn.Sequential(           
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),   # added batchnorm to all layer in exp 13
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # add one conv layer in exp14
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
          )
          
           
        self.decoder_late = nn.Sequential(
            nn.Conv2d(256+256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1), # add one conv layer in exp15
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),            
           
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            #nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
            #nn.Sigmoid()
        )

    def forward(self, x):

        
        x_skip = self.encoder_early(x)
        x = self.encoder_late(x_skip)

        x = self.decoder_early(x)

        # skip connection
        x = torch.cat((x,x_skip),dim=1)

        # Decoder
        x = self.decoder_late(x)

        return x