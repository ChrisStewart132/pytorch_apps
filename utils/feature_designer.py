import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from video import Video

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

kernel = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])


class ManualWeightedConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1):
        super(ManualWeightedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weight as a parameter with the correct shape
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Update the weights using the provided kernels
        self.update_weights(kernel, kernel, kernel)
        self.scaling = 2

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, self.scaling, self.scaling)
        x = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        # x = x.where(x > 0.0, torch.zeros_like(x)) #commented out as this made the ui unresponsive

        x = nn.functional.upsample(x, scale_factor=self.scaling, mode='bilinear', align_corners=False)
        return x

    def update_weights(self, r, g, b):
        # Properly assign the weights for each input channel
        self.weight.data[0, 0, :, :] = torch.tensor(r)
        self.weight.data[0, 1, :, :] = torch.tensor(g)
        self.weight.data[0, 2, :, :] = torch.tensor(b)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

manual_conv = ManualWeightedConv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
model = nn.Sequential(manual_conv).to(device)

v = Video(show=True, save=False)
video_path = '..\\ImageToImageCNN\\data\\validation\\input\\Validation.mp4'

# --- UI Setup ---
cv2.namedWindow("UI", cv2.WINDOW_NORMAL)  # Changed to WINDOW_NORMAL for resizable window
cv2.namedWindow("Kernel", cv2.WINDOW_NORMAL)

alpha = 0.9


def on_alpha_change(val):
    global alpha
    alpha = val / 100.0


cv2.createTrackbar("Alpha", "UI", int(alpha * 100), 100, on_alpha_change)

kernel_values = kernel.flatten().tolist()
def on_kernel_change(val, index):
    kernel_values[index] = val-10//2
    global kernel
    kernel = np.array(kernel_values).reshape(3, 3)
    manual_conv.update_weights(kernel, kernel, kernel)
    update_kernel_visualization()


for i in range(9):
    cv2.createTrackbar(f"Kernel {i + 1}", "UI", 10//2, 10,
                       lambda val, index=i: on_kernel_change(val, index))

scaling = 2


def on_scaling_change(val):
    global scaling
    if val < 1:
        val = 1
    scaling = val
    manual_conv.scaling = scaling


cv2.createTrackbar("Scaling", "UI", scaling, 10, on_scaling_change)

# --- End of UI Setup ---

# --- Kernel Visualization ---
def update_kernel_visualization():
    kernel_display = np.zeros((150, 150), dtype=np.uint8)
    cell_size = 50

    for i in range(3):
        for j in range(3):
            val = kernel[i, j]
            text = str(val)
            color = 255 if val >= 0 else 0
            cv2.rectangle(kernel_display, (j * cell_size, i * cell_size), ((j + 1) * cell_size, (i + 1) * cell_size), color, -1)
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = int(j * cell_size + (cell_size - text_width) / 2)
            text_y = int((i + 1) * cell_size - (cell_size - text_height) / 2)
            cv2.putText(kernel_display, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255-color), 2)

    cv2.imshow("Kernel", kernel_display)

update_kernel_visualization()  # Initial call to display the initial kernel

# --- End of Kernel Visualization ---

def fn(frame):
    # pre
    x = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device)

    # infer
    with torch.no_grad():
        y = model(x)

    # post
    y_np = y[0].permute(1, 2, 0).cpu().numpy()
    y_np = np.clip(y_np, 0, 1)

    # Convert to grayscale
    y_gray = (y_np * 255).astype(np.uint8)

    # Duplicate grayscale to create a 3-channel image
    y_rgb = np.stack([y_gray[:, :, 0], y_gray[:, :, 0], y_gray[:, :, 0]], axis=-1)

    frame = frame.astype(np.uint8)
    global alpha
    blended_frame = cv2.addWeighted(frame, 1 - alpha, y_rgb, alpha, 0)
    
    #the size of the zeros here is irrelevant.
    # it is the window type that causes the resize.
    cv2.imshow("UI", np.zeros((100, 300, 3), dtype=np.uint8))  
    return blended_frame


while 1:
    v.load_video(video_path)
    while v.process_frame(fn):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            v.release()
            cv2.destroyAllWindows()
            exit()
        time.sleep(0.01)
v.release()
