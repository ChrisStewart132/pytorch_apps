"""
extract frames from videos
usage:
    python video_to_images.py --video_dir <video_dir> --output_dir_for_images <output_dir> --frames_skipped <frames_skipped>


python video_to_images.py --video_dir train/input --output_dir train/input 
python video_to_images.py --video_dir train/output --output_dir train/output 

python video_to_images.py --video_dir test/input --output_dir test/input 
python video_to_images.py --video_dir test/output --output_dir test/output 
"""
import os
import cv2
import argparse

def video_to_images(video_path, output_dir, prefix="frame", frames_skipped=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames_saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_count % frames_skipped != 0:
            frame_count += 1
            continue

        frame_name = f"{prefix}_{frame_count:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)

        # Check if the image already exists *before* saving
        if not os.path.exists(frame_path):  
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            frames_saved += 1
        #else:
            #print(f"Warning: Frame {frame_path} already exists. Skipping.") 

    cap.release()
    print(f"Processed {video_path} and saved {frames_saved} (of {frame_count}) frames to {output_dir}") 


def main():
    parser = argparse.ArgumentParser(description="Convert videos to images.")
    parser.add_argument("--video_dir", required=True, help="Directory containing videos.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output images.")
    parser.add_argument("--frames_skipped", type=int, default=5, help="Frames skipped per frame saved")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.video_dir):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(args.video_dir, filename)
            video_to_images(video_path, args.output_dir, filename[:-4], args.frames_skipped)

if __name__ == "__main__":
    main()