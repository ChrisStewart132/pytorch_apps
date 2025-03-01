import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


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

class PerceptualLoss(nn.Module):
    def __init__(self, model, resize=True, device="cpu", style_layers=None, style_weights=None, feature_layers=None, feature_weights=None, l1_loss_factor=0.0):
        super().__init__()
        self.device = device
        self.resize = resize
        self.model = model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.transform = nn.functional.interpolate
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

        # Style Loss Configuration
        if style_layers is None:
            self.style_layers = list(range(7))  # Default to all 7 layers (0-6)
        else:
            self.style_layers = style_layers
        if style_weights is None:
            self.style_weights = [1.0] * len(self.style_layers)  # Default weights of 1.0
        else:
            assert len(style_weights) == len(self.style_layers), "Length of style_weights must match length of style_layers"
            self.style_weights = style_weights

        # Feature Loss Configuration
        if feature_layers is None:
            self.feature_layers = []  # No feature layers by default
        else:
            self.feature_layers = feature_layers
        if feature_weights is None:
            self.feature_weights = []  # No feature weights by default
        else:
            assert len(feature_weights) == len(self.feature_layers), "Length of feature_weights must match length of feature_layers"
            self.feature_weights = feature_weights

        self.l1_loss_factor = l1_loss_factor

        # normalize style and feature and l1
        total = sum(self.style_weights) + sum(self.feature_weights) + l1_loss_factor
        self.style_weights = [w / total for w in self.style_weights]
        self.feature_weights = [w / total for w in self.feature_weights]
        l1_loss_factor /= total
        print(f"Perceptual loss initialized with L1: {l1_loss_factor}, {[pair for pair in zip(self.style_layers, self.style_weights)]} style layers, and {[pair for pair in zip(self.feature_layers, self.feature_weights)]} feature layers")

    def gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(b * c * d)

    def perceptual_loss(self, input, target, feature_extractor):
        feature_loss = 0.0
        style_loss = 0.0

        x = input
        y = target

        for i, layer in enumerate(feature_extractor):
            if i > max(self.feature_layers + self.style_layers):
                break

            x = layer(x)
            y = layer(y)
            if i in self.feature_layers:
                layer_index = self.feature_layers.index(i)
                feature_loss += self.feature_weights[layer_index] * nn.functional.l1_loss(x, y)
            if i in self.style_layers:
                layer_index = self.style_layers.index(i)
                style_loss += self.style_weights[layer_index] * nn.functional.l1_loss(self.gram_matrix(x), self.gram_matrix(y))

        return feature_loss + style_loss + nn.functional.l1_loss(input, target) * self.l1_loss_factor


    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = self.perceptual_loss(input, target, self.model)
        return loss


class EffNetV2PerceptualLoss(PerceptualLoss):
    def __init__(self, resize=True, device="cpu", style_layers=[0,2,3], style_weights=[1,1,1], feature_layers=[], feature_weights=[], l1_loss_factor=0):
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).features
        super().__init__(model, resize, device, style_layers, style_weights, feature_layers, feature_weights, l1_loss_factor)


class ConvNeXtTinyPerceptualLoss(PerceptualLoss):
    def __init__(self, resize=True, device="cpu", style_layers=[0,1,2], style_weights=[1,1,1], feature_layers=[], feature_weights=[], l1_loss_factor=0):
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        super().__init__(model, resize, device, style_layers, style_weights, feature_layers, feature_weights, l1_loss_factor)