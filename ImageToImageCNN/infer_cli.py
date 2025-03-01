"""
python infer_cli.py --model_path models/model_32f_v1.pth --video_path data/test/input/ImageToImageCNN.mp4 --save_video
"""
import argparse
import os
import cv2
import torchvision.transforms.v2 as transforms
import torch
from model import ImageToImageCNN
import numpy as np
import re

# SCALE_FACTOR = 2  <- Remove hardcoded SCALE_FACTOR

def run_inference(model_path, video_path, save_video=False, scale_factor=2): # Add scale_factor as argument with default value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match = re.search(r"_([0-9]+f)_", model_path)  # Find a number followed by f
    N_FEATURES_str = match.group(1)  # Get the captured group (e.g., "32f")
    N_FEATURES = int(N_FEATURES_str[:-1])  # Convert to int, remove "f"

    model = ImageToImageCNN(3, 3, features=N_FEATURES, scale_factor=scale_factor) # Use scale_factor argument
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (frame_width, frame_height)

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(f"{model_path[:-4]}_inference.avi", fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)
        print(f"saving video to: {model_path}")

    if save_video and not out.isOpened():
        print("Error opening video writer!")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        output_image = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_image_bgr = np.clip(output_image_bgr, 0, 1)
        output_image_bgr = (output_image_bgr * 255).astype(np.uint8)

        if save_video:
            #print("write frame to video")
            out.write(output_image_bgr)
            

        cv2.imshow(f"{model_path} Inference Output (press Q to exit)", output_image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    if save_video:
        print("release video")
        out.release()
    cap.release()
    cv2.destroyAllWindows()

def process_path(path_arg):
    if os.path.isabs(path_arg):  # Check if it's an absolute path
        resolved_path = path_arg
    else:  # It's a relative path
        resolved_path = os.path.abspath(path_arg)  # Resolve relative to CW
    return resolved_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference on a video")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument("--save_video", action="store_true", help="Flag to save the output video")
    parser.add_argument("--scale_factor", type=int, default=2, help="Scale factor for upsampling (optional, default: 2)") # Add scale_factor argument
    args = parser.parse_args()
    run_inference(
        process_path(args.model_path),
        process_path(args.video_path),
        args.save_video,
        args.scale_factor # Pass scale_factor argument
    )
    print("inference finished")