import os
import glob

def remove_jpg_images(root_dir):
    """
    Searches all child directories within the given root directory and removes all .jpg images.
    """
    try:
        for dirpath, dirnames, filenames in os.walk(root_dir):  # Walk through all subdirectories
            jpg_files = glob.glob(os.path.join(dirpath, "*.jpg"))  # Find .jpg files in current directory
            if len(jpg_files) > 0 and input(f"Delete {len(jpg_files)} images in {dirpath}? (y/n): ").lower() == "y":
                for file_path in jpg_files:
                    try:
                        os.remove(file_path) # Remove the .jpg image
                        print(f"Removed: {file_path}")
                    except OSError as e:
                        print(f"Error removing {file_path}: {e}")

    except FileNotFoundError:
        print(f"Error: Directory '{root_dir}' not found.")
    except Exception as e: # Catch any other potential errors
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    current_directory = os.getcwd()
    remove_jpg_images(current_directory)
    print("Finished processing.")