import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=1, out_channels=1):
        super(ResNet34UNet, self).__init__()

        # Load pre-trained ResNet-34 model
        resnet = models.resnet34(pretrained=True)

        # Encoder layers
        self.encoder1 = nn.Sequential(*list(resnet.children())[:3])  # First block
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5]) 
        self.encoder3 = resnet.layer2  
        self.encoder4 = resnet.layer3 
        self.encoder5 = resnet.layer4  

        # Decoder with upsampling (interpolation) and skip connections
        self.decoder_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # Upsamples and processes the concatenated features from encoder5 and encoder4.
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),  # Convolution after concatenation
            nn.BatchNorm2d(256),  
            nn.ReLU(inplace=True),  # Activation function

            nn.Upsample(scale_factor=2),  # Upsample the feature map
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),  # Convolution after concatenation
            nn.BatchNorm2d(128),  # Batch normalization for stability
            nn.ReLU(inplace=True),  # Activation function

            nn.Upsample(scale_factor=2),  # Upsample the feature map
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # Convolution after concatenation
            nn.BatchNorm2d(64),  # Batch normalization for stability
            nn.ReLU(inplace=True),  # Activation function

            nn.Upsample(scale_factor=2),  # Upsample the feature map
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),  # Convolution after concatenation
            nn.BatchNorm2d(32),  # Batch normalization for stability
            nn.ReLU(inplace=True),  # Activation function

            nn.Conv2d(32, num_classes, kernel_size=1)  # Final convolution layer
        )

    def forward(self, x):
        # Encoder forward pass
        e1 = self.encoder1(x)  # Output of first block
        e2 = self.encoder2(e1)  # Output of second block
        e3 = self.encoder3(e2)  # Output of third block
        e4 = self.encoder4(e3)  # Output of fourth block
        e5 = self.encoder5(e4)  # Output of fifth block

        # Decoder with skip connections
        d4 = self.decoder_up[0](e5)  # Upsample
        d4 = self.decoder_up[1](torch.cat([d4, e4], dim=1))  # Concatenate skip connection and conv

        d3 = self.decoder_up[4](d4)  # Upsample
        d3 = self.decoder_up[5](torch.cat([d3, e3], dim=1))  # Concatenate skip connection and conv

        d2 = self.decoder_up[8](d3)  # Upsample
        d2 = self.decoder_up[9](torch.cat([d2, e2], dim=1))  # Concatenate skip connection and conv

        d1 = self.decoder_up[12](d2)  # Upsample
        d1 = self.decoder_up[13](torch.cat([d1, e1], dim=1))  # Concatenate skip connection and conv

        out = self.decoder_up[16](d1)  # Final conv layer
        return out
