import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load the pretrained ConvNeXt-Tiny model
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        #self.backbone =  convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        # Put the backbone in evaluation mode
        self.backbone.eval()
        # Freeze the backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Store the layers that produce the feature maps
        self.layer_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        self.layers = nn.ModuleList([self.backbone[i] for i in self.layer_indices])


    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps

class DetectorHead(nn.Module):
    def __init__(self, in_channels=768, num_conv_layers=4, hidden_channels=256, out_channels=1):
        super(DetectorHead, self).__init__()
        layers = []
        #create num_conv_layers of conv2d
        for i in range(num_conv_layers):
            if i == 0:
              layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
            else:
              layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        #out conv2d
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CNNObjectDetector(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_features=8):
        super(CNNObjectDetector, self).__init__()
        self.backbone = FeatureExtractor()
        
        self.backbone_info = [
            (96,4),
            (96,4),
            (192,8),
            (192,8),
            (384,16),
            (384,16),
            (768,32),
            (768,32),
        ]
        self.detectors = nn.ModuleList()
        for i in range(len(self.backbone_info)):
            self.detectors.append(DetectorHead(in_channels=self.backbone_info[i][0], out_channels=out_channels, hidden_channels=n_features))

        # Calculate the total number of channels after concatenation
        total_channels = len(self.backbone_info) * out_channels
        
        self.out = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
            
        
    def forward(self, x):
        feature_maps = self.backbone(x)
        xs = []
        for i, feature_map in enumerate(feature_maps):
            f = self.detectors[i](feature_map)
            xs.append(f)

        upscaled_xs = []
        for i, x_ in enumerate(xs):
            scale_factor = self.backbone_info[i][1]
            upscaled_x = nn.functional.interpolate(x_, size=(224,224), mode='bilinear', align_corners=False)
            upscaled_xs.append(upscaled_x)
            #print(i, upscaled_x.shape) 

        x = torch.cat(upscaled_xs, dim=1)
        
        x = self.out(x)
        return x


    

if __name__ == '__main__':  
    # Test
    model = CNNObjectDetector()
    out_channels = 3
    x = torch.randn(1, out_channels, 224*2, 224*2)  # Even dimensions
    output = model(x)
    print(output.shape, output.shape == torch.Size([1, out_channels, 224, 224])) 
