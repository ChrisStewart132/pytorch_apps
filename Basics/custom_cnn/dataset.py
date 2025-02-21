import os
import cv2
import torch

class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.input_dir = os.path.join(root_dir, "input")
        self.output_dir = os.path.join(root_dir, "output")
        self.image_files = [] # List to store (input_path, output_path) tuples

        for filename in os.listdir(self.input_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                input_path = os.path.join(self.input_dir, filename)
                target_filename = filename.replace("_input", "_target") # Find corresponding target
                target_path = os.path.join(self.output_dir, target_filename)

                if os.path.exists(target_path): # Only add if target exists
                    self.image_files.append((input_path, target_path))
                else:
                    print(f"Warning: No matching target found for {input_path}")

        print(f"{root_dir}", "Found", len(self.image_files), "image pairs")
        print(root_dir, "First pair:", self.image_files[0])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        input_path, output_path = self.image_files[idx] # Get both paths

        try:
            input_image = cv2.imread(input_path)
            output_image = cv2.imread(output_path)

            if input_image is None or output_image is None:
                return None, None

            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)/255
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)/255

        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None, None

        if self.transform:
            # Apply transformations TOGETHER using a consistent random state
            seed = torch.randint(0, 2**32, (1,)).item()  # Generate random seed

            torch.manual_seed(seed)  # Set the seed for transformations
            input_image = self.transform(input_image)

            torch.manual_seed(seed)  # Reset to the same seed before transforming target
            output_image = self.transform(output_image)

        return input_image, output_image
    
    def close(self):
        pass