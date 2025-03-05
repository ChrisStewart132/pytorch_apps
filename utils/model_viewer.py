import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from video import Video

N_FEATURES = 64
BLOCK_LAYER = 2

class ModelViewer:
    def __init__(self, model):
        self.model = model

    def _print_module_info(self, name, module, indent=0, max_indent=1):
        if indent > max_indent:
            return
        indent_str = "  " * indent
        print(f"{indent_str}{name}: {module.__class__.__name__}")
        if isinstance(module, nn.Sequential):
            for sub_name, sub_module in module.named_children():
                self._print_module_info(sub_name, sub_module, indent + 1)
        elif isinstance(module, nn.Module):
            for sub_name, sub_module in module.named_children():
                self._print_module_info(sub_name, sub_module, indent + 1)

    def forward_block_by_block(self, input=torch.randn(1, 3, 224, 224)):
            try:
                print(f"Initial Input shape: {input.shape}")
                x = input
                for name, block in self.model.named_children():
                    print(f"\n--- Processing Block: {name} ({block.__class__.__name__}) ---")
                    self._print_module_info(name, block)
                    print(f"  Block Input shape: {x.shape}")
                    x = block(x)
                    print(f"  Block Output shape: {x.shape}")
                    #print(f"  Block output (first 5 elements): {x.flatten()[:5]}")
                print(f"\nFinal Output shape: {x.shape}")
            except Exception as e:
                print(f"Error during block-by-block forward pass: {e}")


if __name__ == "__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
    model.eval()

    viewer = ModelViewer(model)  
    viewer.forward_block_by_block()

    model.to(device)

    v = Video(show=True, save=False) 
    video_path = '..\\ImageToImageCNN\\data\\validation\\input\\Validation.mp4'  # Replace with your video path
    v.load_video(video_path)

    def fn(frame):
        #pre
        x = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.to(device)

        #infer
        with torch.no_grad():
            y = x
            for i in range(BLOCK_LAYER):
                y = model[i](y) 

            y=torch.Tensor.cpu(y)
            
            num_feature_maps = min(N_FEATURES, y.shape[1]) 
            feature_maps = []

            for n in range(num_feature_maps):
                feature_map = y[0, n, :, :].numpy()
                #feature_map = cv2.resize(feature_map, (frame.shape[1], frame.shape[0]))
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255
                feature_map = feature_map.astype(np.uint8)
                feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_DEEPGREEN)
                feature_maps.append(feature_map)
            
            # Create a grid of feature maps
            grid_size = int(num_feature_maps**0.5)
            rows = []
            for i in range(0, num_feature_maps, grid_size):
                row = np.hstack(feature_maps[i:i+grid_size])
                rows.append(row)
            grid = np.vstack(rows)
            grid = cv2.resize(grid, (frame.shape[1], frame.shape[0]))
            y_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

        #post
        frame = frame.astype(np.uint8)
        blended_frame = cv2.addWeighted(frame, 0.0, y_rgb, 1, 0)

        # Add text overlay
        text = f"N_FEATURES: {N_FEATURES}, Block Layer: {BLOCK_LAYER}"
        cv2.putText(blended_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        return blended_frame
    
    while v.process_frame(fn):
        #get cv2 key space for nxt frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
    v.release()
