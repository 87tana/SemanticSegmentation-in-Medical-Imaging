import torch.nn as nn  # Neural network modules and loss functions
import torchvision.models as models  # provide access to pre-trained models

class ResNet50UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet50UNet, self).__init__()

        # Load pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=True)

        # Remove fully connected layers (classifier) from ResNet
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder with transposed convolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        # Decoder with upsampling (interpolation)
        self.decoder_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder_up(x)
        return x
