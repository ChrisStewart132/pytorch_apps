
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ModelViewer:
    def __init__(self, model):
        self.model = model

    def print_model_structure(self):
        print(f"Model: {self.model.__class__.__name__}")
        print("-" * 40)
        for name, module in self.model.named_children():
            self._print_module_info(name, module, 0)
        print("-" * 40)

    def _print_module_info(self, name, module, indent, max_indent=2):
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

    def forward_block_by_block(self, input_shape=(1, 3, 224, 224)):
            try:
                dummy_input = torch.randn(input_shape)
                print(f"Initial Input shape: {dummy_input.shape}")
                x = dummy_input
                
                for name, block in self.model.named_children():
                    print(f"\n--- Processing Block: {name} ({block.__class__.__name__}) ---")
                    print(f"  Block Input shape: {x.shape}")
                    x = block(x)
                    print(f"  Block Output shape: {x.shape}")
                    #print(f"  Block output (first 5 elements): {x.flatten()[:5]}")
                print(f"\nFinal Output shape: {x.shape}")

            except Exception as e:
                print(f"Error during block-by-block forward pass: {e}")

if __name__ == "__main__":
    # Example usage with convnext_tiny features
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
    viewer = ModelViewer(model)  
    viewer.print_model_structure()
    viewer.forward_block_by_block()
