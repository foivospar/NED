import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.io
import numpy as np

class StarganV2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(StarganV2Logger, self).__init__(logdir)

    def log_training(self, key, value,
                     iteration):
        self.add_scalar(key,value,iteration)

    def log_mesh(self, video, name, iteration):
        # video = video.permute(0,3,1,2)
        self.add_video(
            '{}'.format(name),
            video.unsqueeze(0),
            fps=30,
            global_step=iteration
        )
