import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time


class TensorboardLogger:
    def __init__(self, log_dir="runs"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", timestamp)  
        self.writer = SummaryWriter(log_dir=log_dir)


    def log_epoch_loss(self, epoch, train_loss, val_loss):
       self.writer.add_scalar("Loss/Train", train_loss, epoch)  
       self.writer.add_scalar("Loss/Val", val_loss, epoch)

    def log_image(self, img_tensor, tag="Image", dataformats="CHW", step=0):
        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats) 

    def close(self):
        self.writer.close()
