import torch.nn as nn  # Neural network modules and loss functions
import torchvision.models as models # provide access to pre-trained models



class ResNetUNet_4(nn.Module):
    def __init__(self, num_classes=1, out_channels=1,):
        super(ResNetUNet_4, self).__init__()

        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)  # import torchvision

        # Remove fully connected layers (classifier) from ResNet
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) #remove two last layer,

        # Decoder with transposed convolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )


        # Decoder with upsampling (interpolation)
        self.decoder_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
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