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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.custom_conv(x)

class DownsampleConv(nn.Module):
    """
    returns centre kernel downsampled Batch, Out, H/scale_factor, W/scale_factor 
    """
    def __init__(self, in_channels=3, out_channels=3, scale_factor=2):
        super(DownsampleConv, self).__init__()
        self.custom_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=scale_factor),
        )
    def forward(self, x):
        return self.custom_conv(x)


class ImageToImageCNN(nn.Module):
    """
    Simple Image to Image CNN model, 
        both image height, and width require to be divisible by scale_factor 
    
    input: Batch, in_channels, Height, Width
    output: Batch out_channels, Height, Width
    """
    def __init__(self, in_channels=3, out_channels=3, features=8, scale_factor=2):
        super(ImageToImageCNN, self).__init__()
        print(f"\nImageToImageCNN in:{in_channels}, out:{out_channels}, features:{features}, scale_factor(input divisible):{scale_factor}\n")

        self.squash = FeatureConv(3, 1, kernel_size=1)
        self.gray_features = FeatureConv(1, features)

        self.downsample = DownsampleConv(in_channels, in_channels, scale_factor=scale_factor)
        self.color_features = FeatureConv(in_channels, features)
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')

        self.out = nn.Conv2d(2*features, out_channels, kernel_size=1)

    def forward(self, x):
        gray_x = self.squash(x)
        gray_f = self.gray_features(gray_x)

        color_x = self.downsample(x)
        color_f = self.color_features(color_x)
        color_f = self.upsample(color_f)

        combined = torch.cat((gray_f, color_f), dim=1)
        return self.out(combined)
    

if __name__ == '__main__':  
    # Test
    model = ImageToImageCNN()
    x = torch.randn(1, 3, 64, 64)  # Even dimensions
    output = model(x)
    print(output.shape, output.shape == x.shape)  # Should be [1, 3, 64, 64]