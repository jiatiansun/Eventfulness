from .labelLoader import *
import torch

class NoneLoader(LabelReadWriter):

    def __init__(self, num_acc_dirs, num_vel_dirs, num_blurrs, num_label=0):
        self.num_dirs = 1 + num_acc_dirs + num_vel_dirs + num_blurrs
        if num_label > 0:
            self.num_dirs = num_label
        return

    def setLabelDir(self, videoPath):
        return

    def getLabelDir(self):
        return ""

    @staticmethod
    def getName():
        return "none"

    def loadLabel(self, label_fileWOExt, ov_fps, target_fps, num_frames, frame_sampling_idx, startT, endT):
        return torch.zeros((self.num_dirs, num_frames))

    def getNumLabel(self):
        return None