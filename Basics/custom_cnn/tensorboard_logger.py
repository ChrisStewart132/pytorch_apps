import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time


class TensorboardLogger:
    def __init__(self, log_dir="runs"):
        """
        Initializes the Tensorboard logger.

        Args:
            log_dir (str, optional): Directory to save the log files. Defaults to "runs".
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", timestamp)  # Creates runs/20231027-103000
        self.writer = SummaryWriter(log_dir=log_dir)


    def log_epoch_loss(self, epoch, train_loss, val_loss):
       """Logs training and validation loss for an epoch.

       Args:
           epoch (int): The current epoch number.
           train_loss (float): The average training loss for the epoch.
           val_loss (float): The average validation loss for the epoch.
       """
       self.writer.add_scalar("Loss/Train", train_loss, epoch)  # Log at epoch level
       self.writer.add_scalar("Loss/Val", val_loss, epoch)
       # Example of logging at each step within an epoch (if needed) - uncomment if required
       # self.writer.add_scalar("Loss/Train", train_loss, self.step)  
       # self.writer.add_scalar("Loss/Val", val_loss, self.step)
       # self.step +=1 # Increment if logging per step

    def log_image(self, img_tensor, tag="Image", dataformats="CHW", step=0):
        """Logs a single image tensor.

        Args:
            img_tensor (torch.Tensor): The image tensor to log.  Should be 3D (C, H, W) or 4D (N, C, H, W).
            tag (str, optional): The tag/name for the image in Tensorboard. Defaults to "Image".
            dataformats (str, optional):  Specify the data format of the image tensor. Defaults to "CHW".  
                                           Other options include "NCHW" (batch, channel, height, width)
        """

        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats) # Log image at current step

    def close(self):
        """Closes the Tensorboard writer."""
        self.writer.close()
