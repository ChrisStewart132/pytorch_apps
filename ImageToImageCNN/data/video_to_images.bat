REM extracting frames from video files in train/input train/output test... etc
python video_to_images.py --video_dir train/input --output_dir train/input
python video_to_images.py --video_dir train/output --output_dir train/output
python video_to_images.py --video_dir test/input --output_dir test/input
python video_to_images.py --video_dir test/output --output_dir test/output
python video_to_images.py --video_dir validation/input --output_dir validation/input
python video_to_images.py --video_dir validation/output --output_dir validation/output
pause