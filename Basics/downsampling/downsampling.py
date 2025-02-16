"""
pytorch program that:
    reads an image from image_path,
    resizes to a specified WIDTH, HEIGHT,
    downsamples by a DOWNSAMPLE_FACTOR,
    resizes back to WIDTH, HEIGHT,
    outputs to a tensorboard log

    in a cli in the same dir
        tensorboard --logdir=runs


uses 3 methods for downsampling (maxPool2d, avgPool2d, cnn2d with manual weights)
"""
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs

WIDTH, HEIGHT = 256,256
DOWNSAMPLE_FACTOR = 8

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Default device: {device}")

image_path = "../images/image1.jpg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB 
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.permute(2, 0, 1)  # Permute to (C, H, W)
img_tensor = img_tensor.float()/255# normalize 

pre_process_transforms = torch.nn.Sequential(
    transforms.Resize((WIDTH, HEIGHT)),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)

img_tensor = pre_process_transforms(img_tensor)
print(f"img tensor shape: {img_tensor.shape}")
writer.add_image(f"{image_path}", img_tensor, 0, dataformats='CHW')# dataformats=NCHW # n samples, n channels, height, width

class MaxDownsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.MaxPool2d_downsample  = nn.MaxPool2d(kernel_size=DOWNSAMPLE_FACTOR, stride=DOWNSAMPLE_FACTOR, padding=max(0,DOWNSAMPLE_FACTOR//2-1))  
    def forward(self, x):
        x = self.MaxPool2d_downsample(x)
        return x

class AvgDownsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.AvgPool2d_downsample = nn.AvgPool2d(kernel_size=DOWNSAMPLE_FACTOR, stride=DOWNSAMPLE_FACTOR, padding=max(0,DOWNSAMPLE_FACTOR//2-1))
    def forward(self, x):
        x = self.AvgPool2d_downsample(x)
        return x
    
class CNNDownsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_downsample = nn.Conv2d(3, 3, kernel_size=DOWNSAMPLE_FACTOR, stride=DOWNSAMPLE_FACTOR, padding=max(0,DOWNSAMPLE_FACTOR//2-1))
        # Initialize weights to "select" the center pixel
        #print(f"convv weights shape:", self.Conv2d_downsample.weight.data.shape)
        # manually set the kernels
        for i in range(3):# for each output filter/kernel
            for j in range(3):# for each input filter/kernel
                self.Conv2d_downsample.weight.data[i, j, :, :].fill_(0.0)  # Zero out all weights initially
                if i == j:# conv(3,3) has 9 kernels... 3 for each input channel, so take kernels 0,0 1,1 and 2,2 to output to channels 0,1,2 respectively.. could also do 0,0 ,1,0 ,2,0 ig aswell
                    self.Conv2d_downsample.weight.data[i, j, DOWNSAMPLE_FACTOR//2, DOWNSAMPLE_FACTOR//2] = 1.0 # Set the center pixel
        
        # Zero out the bias 
        if self.Conv2d_downsample.bias is not None:  # Check if bias exists (it should now)
            nn.init.constant_(self.Conv2d_downsample.bias, 0.0)
        #print(f"conv2d weights:", self.Conv2d_downsample.weight.data)

    def forward(self, x):
        x = self.Conv2d_downsample(x)
        return x

img_tensors = img_tensor.unsqueeze(0)

# Instantiate models
max_model = MaxDownsampling()
avg_model = AvgDownsampling()
cnn_model = CNNDownsampling()

# Perform downsampling and store results
downsampled_images = {}

with torch.no_grad(): # Important for inference
    downsampled_images["max"] = max_model(img_tensors)
    downsampled_images["avg"] = avg_model(img_tensors)
    downsampled_images["cnn"] = cnn_model(img_tensors)


post_process_transforms = torch.nn.Sequential(
    transforms.Resize((WIDTH, HEIGHT)),

)
for i, (key, image) in enumerate(downsampled_images.items()):
    print(f"{key} shape:", image.shape)
    resized_image = post_process_transforms(image)
    writer.add_image(f"{image_path}_{key}_{DOWNSAMPLE_FACTOR}_w{WIDTH}_h{HEIGHT}", resized_image, 0, dataformats='NCHW')

writer.close()
