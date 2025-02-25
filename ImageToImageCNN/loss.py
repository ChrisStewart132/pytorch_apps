import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights


class MaxPoolLoss(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.loss = nn.L1Loss()
        self.scale_factor = scale_factor
    def forward(self, input_images, output_images):
        kernel_size = self.scale_factor
        stride = self.scale_factor
        loss = 0
        input_pooled = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)(input_images)
        output_pooled = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)(output_images)
        loss += self.loss(input_pooled, output_pooled)
        return loss

class WeightedMSELoss(nn.Module):
    def __init__(self, weights=[1,1,1]):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)  # Weights as a tensor
    def forward(self, outputs, targets):
        batch_size = outputs.size(0)  # Get batch size
        loss = 0
        for c in range(outputs.size(1)):  # Iterate through channels
            mse = nn.MSELoss()(outputs[:, c, ...], targets[:, c, ...]) # MSE for channel c
            loss += self.weights[c] * mse  # Weighted sum of channel losses
        return loss / batch_size # Average over batch

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True, device="cpu", feature_layers=[0, 1, 2, 3], style_layers=[], feature_weights=None, style_weights=None):
        super(VGGPerceptualLoss, self).__init__()

        self.device = device
        self.resize = resize
        self.feature_layers = feature_layers
        self.style_layers = style_layers

        # Define weights if not provided
        if feature_weights is None:
          self.feature_weights = [1.0] * len(feature_layers)
        else:
          self.feature_weights = feature_weights

        if style_weights is None:
          self.style_weights = [1.0] * len(style_layers)
        else:
          self.style_weights = style_weights

        blocks = []
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[16:23].eval())

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
            bl.to(device) # Move blocks to device

        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) # Move mean to device
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))  # Move std to device

    def forward(self, input, target):

        if input.shape[1] != 3: # Handle grayscale or single-channel inputs
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            if i in self.feature_layers:
                loss += self.feature_weights[self.feature_layers.index(i)] * nn.functional.l1_loss(x, y) # Weighted Feature Loss

            if i in self.style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += self.style_weights[self.style_layers.index(i)] * nn.functional.l1_loss(gram_x, gram_y) # Weighted Style Loss

        return loss