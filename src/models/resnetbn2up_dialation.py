import torch
import torch.nn as nn  # Neural network modules and loss functions
import torchvision.models as models # provide access to pre-trained models



class ResNet2up_dialation(nn.Module):
    def __init__(self, num_classes=1, out_channels=1,):
        super(ResNet2up_dialation, self).__init__()

        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)  # import torchvision

        # Remove fully connected layers (classifier) from ResNet
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) #remove two last layer,


        # Decoder with upsampling (interpolation)
        self.decoder_up = nn.Sequential(
           
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),   # added batchnorm to all layer in exp 13
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # add one conv layer in exp14
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1,dilation=2),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1,dilation=2), # add one conv layer in exp15  
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
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder_up(x)
        return x