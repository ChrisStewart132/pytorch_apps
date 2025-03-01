import cv2
import time
import numpy as np

class Video:
    def __init__(self, show=True, save=True):
        self.show = show
        self.save = save
        self.cap = None
        self.out = None
        self.width = 0
        self.height = 0
        self.fps = 0

    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video at {path}")
            return False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.save:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out = cv2.VideoWriter('output.avi', fourcc, self.fps, (self.width, self.height))
            
        return True


    def process_frame(self, process_function):
        ret, frame = self.cap.read()
        if frame is None:
            return False
        
        start_time = time.time()
        processed_frame = process_function(frame)
        end_time = time.time()
        
        if self.show:
            cv2.imshow('Video', processed_frame)

        if self.save:
            self.out.write(processed_frame)
        
        return True

    def release(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video = Video()
    if video.load_video('ImageToImageCNN/data/validation/input/Validation.mp4'):
        while video.process_frame(lambda x: x):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
