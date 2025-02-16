"""

"""
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs

WIDTH, HEIGHT = 512,512
STRIDE = 1
DILATION = 2
KERNEL_SIZE = 3
ORIGINAL_TRANSPARENCY = -0.5# uses cnn bias
image_path = "../images/image1.jpg"
IMAGE_NAME = image_path.split("/")[-1].split(".")[0]
VIDEO_STREAMING = True

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Default device: {device}")
    
class CNN(nn.Module):
    def __init__(self, kernel_size=3, stride=1, bias_value=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=kernel_size, bias=True, padding=kernel_size//2, stride=stride, dilation=dilation)  # Single Conv2d layer
        self.conv.weight.data.zero_()
        self.conv.bias.data.fill_(bias_value)
        print(self.conv.weight.data.shape)

        horizontal = [[-1 if j == 0 else 1 if j == (kernel_size - 1) else 0 for j in range(kernel_size)] for i in range(kernel_size)]
        vertical = [[-1 if i == 0 else 1 if i == (kernel_size - 1) else 0 for j in range(kernel_size)] for i in range(kernel_size)]
        both = [[horizontal[i][j] + vertical[i][j] for j in range(kernel_size)] for i in range(kernel_size)]
        if kernel_size == 1:
            horizontal, vertical, both = [[1]], [[1]], [[1]]
        
        # x, y, xy edge detection filters
        self.conv.weight.data[0, :, :, :] = torch.tensor(horizontal, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        self.conv.weight.data[1, :, :, :] = torch.tensor(vertical, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        self.conv.weight.data[2, :, :, :] = torch.tensor(both, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        
        # input channel 0 edge detection filter
        self.conv.weight.data[3, 0, :, :] = torch.tensor(both, dtype=torch.float32)
        # input channel 1 edge detection filter
        self.conv.weight.data[4, 1, :, :] = torch.tensor(both, dtype=torch.float32)
        # input channel 2 edge detection filter
        self.conv.weight.data[5, 2, :, :] = torch.tensor(both, dtype=torch.float32)

    def forward(self, x):
        edges = self.conv(x)  # Output shape: (batch_size, n_channels, H, W)
        return edges

def process_input(img, logging=True):
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)  # Permute to (C, H, W)
    img_tensor = img_tensor.float()/255# normalize 
    pre_process_transforms = torch.nn.Sequential(
        transforms.Resize((WIDTH, HEIGHT)),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    img_tensor = pre_process_transforms(img_tensor)
    if logging:
        log_image("original", img_tensor)
    return img_tensor.unsqueeze(0).to(device)

def process_output(batch_output, logging=True):
    key = "extended_prewitt"
    post_process_transforms = torch.nn.Sequential(
        transforms.Resize((WIDTH, HEIGHT)),
    )
    resized_batch = post_process_transforms(batch_output)
    resized_image = resized_batch.squeeze(0)

    x = resized_image[0, :, :].unsqueeze(0).repeat(3, 1, 1)  # Extract channel 1 
    y = resized_image[1, :, :].unsqueeze(0).repeat(3, 1, 1)  # Extract channel 2 

    xy = resized_image[2, :, :].unsqueeze(0)
    empty_channel = torch.zeros_like(xy)
    xy = torch.cat([xy, xy, xy], dim=0)

    rxy = resized_image[3, :, :].unsqueeze(0)  # Extract channel 4
    rxy_rgb = torch.cat([rxy, empty_channel, empty_channel], dim=0)

    gxy = resized_image[4, :, :].unsqueeze(0)  # Extract channel 5
    gxy_rgb = torch.cat([empty_channel, gxy, empty_channel], dim=0)

    bxy = resized_image[5, :, :].unsqueeze(0)  # Extract channel 6
    bxy_rgb = torch.cat([empty_channel, empty_channel, bxy], dim=0)

    if logging:
        log_image(f"x", x)
        log_image(f"y", y)
        log_image(f"xy", xy)
        log_image(f"rxy", rxy_rgb)
        log_image(f"gxy", gxy_rgb)
        log_image(f"bxy", bxy_rgb)

    return x, y, xy, rxy_rgb, gxy_rgb, bxy_rgb

def log_image(tensor_name, img_tensor, step=0):
    name = f"{IMAGE_NAME}_{tensor_name}_kernel:{KERNEL_SIZE}x{KERNEL_SIZE}_stride:{STRIDE}_dilation:{DILATION}_bias:{ORIGINAL_TRANSPARENCY}"
    writer.add_image(name, img_tensor, step, dataformats='CHW')# dataformats=NCHW # n samples, n channels, height, width

# Instantiate model
model = CNN(KERNEL_SIZE, STRIDE, ORIGINAL_TRANSPARENCY, DILATION)
model.to(device)

if not VIDEO_STREAMING:
    ########################################################################################################################
    ############################################## load image(s) ##########################################################
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB 
    img_tensor_batch = process_input(img)

    # Inference
    with torch.no_grad():
        batch_output = model(img_tensor_batch)
        x, y, xy, rxy_rgb, gxy_rgb, bxy_rgb = process_output(batch_output)
    ############################################## load image(s) ##########################################################
    ########################################################################################################################
    exit(0)






########################################################################################################################
############################################## Video Streaming ##########################################################
video_path = "../videos/bird.mp4"  # Replace with your video file or 0 for webcam
output_path = "extended_prewitt_bird.avi"  # Path to save the output video

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file or webcam.")
    exit()

# Get the video properties
width = WIDTH
height = HEIGHT
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width*4, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Preprocessing
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Important: OpenCV reads BGR, PyTorch expects RGB
    img_tensor_batch = process_input(img, logging=False)

    # Inference
    with torch.no_grad():
        key = "extended_prewitt"
        batch_output = model(img_tensor_batch)
        x, y, xy, rxy_rgb, gxy_rgb, bxy_rgb = process_output(batch_output, logging=False)
        print(model.conv.bias.data)

    # Concatenate tensors
    img_tensor_concat = torch.cat((img_tensor_batch.squeeze(0), x, y, xy), dim=2)
    #img_tensor_concat = torch.cat((img_tensor_batch.squeeze(0), rxy_rgb, gxy_rgb, bxy_rgb), dim=2)

    # Convert to numpy and display
    img_concat = img_tensor_concat.cpu().numpy().transpose(1, 2, 0)
    img_concat = cv2.cvtColor(img_concat, cv2.COLOR_RGB2BGR)
    
    # Add text comments
    cv2.putText(img_concat, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_concat, "X Gradient", (width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_concat, "Y Gradient", (2*width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_concat, "XY Gradient", (3*width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Write the frame to the output video
    out.write(cv2.convertScaleAbs(img_concat, alpha=(255.0)))

    cv2.imshow("extended_prewitt_cnn",  img_concat)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
writer.close()