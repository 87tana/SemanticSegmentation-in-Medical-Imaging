import torch
import torch.nn as nn
import torchvision.models as models

class VGG2up(nn.Module):
    def __init__(self, num_classes=1, out_channels=1):
        super(VGG2up, self).__init__()

        # Load pre-trained VGG16 model
        vgg = models.vgg16_bn(pretrained=True)

        # Remove fully connected layers (classifier) from VGG16
        self.encoder = vgg.features

        # Decoder with upsampling (interpolation)
        self.decoder_up = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder_up(x)
        return x
