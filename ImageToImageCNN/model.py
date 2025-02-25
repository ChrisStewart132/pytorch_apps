import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureConv(nn.Module):
    """
    returns Batch, Out, H, W feature maps
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FeatureConv, self).__init__()
        self.custom_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.05), 
        )
    def forward(self, x):
        return self.custom_conv(x)

class DownsampleConv(nn.Module):
    """
    returns learned downsample of Batch, Out, H/scale_factor, W/scale_factor 
    """
    def __init__(self, in_channels=3, out_channels=3, scale_factor=2, kernel_size=3):
        super(DownsampleConv, self).__init__()
        self.custom_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=scale_factor),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.05), 
        )
    def forward(self, x):
        return self.custom_conv(x)

class TransposedUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3):  
        super(TransposedUpsampler, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=(kernel_size - 1) // 2, 
                output_padding=scale_factor - 1 
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.05), 
        )
    def forward(self, x):
        return self.upsample(x)

class ImageToImageCNN(nn.Module):
    """
    Image to Image CNN model, 
        both image height, and width require to be divisible by scale_factor 
    
    input: Batch, in_channels, Height, Width
    output: Batch out_channels, Height, Width
    """
    def __init__(self, in_channels=3, out_channels=3, features=8, scale_factor=2):
        super(ImageToImageCNN, self).__init__()
        print(f"\nImageToImageCNN in:{in_channels}, out:{out_channels}, features:{features}, scale_factor(input divisible):{scale_factor}\n")

        self.features = FeatureConv(in_channels, features)

        self.downsample = DownsampleConv(in_channels, 4*features, scale_factor=scale_factor)
        self.downsample_features1 = FeatureConv(4*features, 2*features)
        self.upsample = TransposedUpsampler(2*features, features, scale_factor)

        self.reduction = FeatureConv(2*features, features)

        self.out = nn.Sequential(
            nn.Conv2d(2*features, out_channels, kernel_size=1),
            nn.Sigmoid()#0-1
        )

    def forward(self, x):
        xf = self.features(x)

        dx = self.downsample(x)
        dxf1 = self.downsample_features1(dx)
        dxf = self.upsample(dxf1)

        combined = torch.cat((xf, dxf), dim=1)
        reduced = self.reduction(combined)

        combined2 = torch.cat((reduced, dxf), dim=1)
        out = self.out(combined2)

        return out

    

if __name__ == '__main__':  
    # Test
    model = ImageToImageCNN()
    x = torch.randn(1, 3, 64, 64)  # Even dimensions
    output = model(x)
    print(output.shape, output.shape == x.shape)  # Should be [1, 3, 64, 64]