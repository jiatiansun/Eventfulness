from .labelLoader import *
from .directionSampler import *
from deepVisualBeatUtil.gaussianFilterGenerator import *
from deepVisualBeatUtil.fileAndMediaWriters import CSVWriter

import os
from os.path import dirname as dirname
import torch
import numpy as np

class AccVelLabelLoader(LabelReadWriter):

    def __init__(self, num_acc_dirs, num_vel_dirs, gaussFilterGen):
        if num_acc_dirs == 0:
            self.acc_dir_sampler = None
        else:
            self.acc_dir_sampler = DirectionSampler(num_acc_dirs)

        if num_vel_dirs == 0:
            self.vel_dir_sampler = None
        else:
            self.vel_dir_sampler = DirectionSampler(num_vel_dirs)

        self.gaussFilterGen = gaussFilterGen

    def setLabelDir(self, videoPath):
        self.label_dir = dirname(dirname(os.path.realpath(videoPath))) + '_beatLabel'

    def getLabelDir(self):
        return self.label_dir

    @staticmethod
    def getName():
        return "beat_random_time"

    def getNumLabel(self):
        return None

    def loadLabel(self, label_fileWOExt, ov_fps, target_fps, num_frames, frame_sampling_idx, startT, endT):
        keyFrame_filename = label_fileWOExt + '_envelope.csv'
        keyFrame_fullpath = os.path.join(self.label_dir, keyFrame_filename)

        motion_count_filename = label_fileWOExt + '_envelope_val.csv'
        motion_count_path = os.path.join(self.label_dir, motion_count_filename)


        # accS_filename = label_fileWOExt + '_envelope_accW.csv'
        # velS_filename = label_fileWOExt + '_envelope_velW.csv'

        keyFrames = CSVWriter.readFloatLabelPerRow(keyFrame_fullpath)
        keyFrame_time_stamps = keyFrames[1:]
        duration =  keyFrames[0]
        np_motion_count = CSVWriter.readFloatLabelPerRow(motion_count_path)

        clip_idx = np.logical_and(keyFrame_time_stamps >= startT, keyFrame_time_stamps <= endT)
        clip_time_stamps = keyFrame_time_stamps[clip_idx]
        clip_motion_count = np_motion_count[clip_idx]

        # filter_beat: index of the beat for this video clip's label
        beat_idx = np.floor((clip_time_stamps - startT) * target_fps).astype(int)
        filter_beat = np.logical_and(beat_idx >= 0, beat_idx < num_frames)
        clip_beat_idx = beat_idx[filter_beat]
        clip_motion_count = clip_motion_count[filter_beat]

        accS_clip_labels = np.empty((0, num_frames), dtype=np.float32)
        if self.acc_dir_sampler is not None:
            accS_filename = label_fileWOExt + '_envelope_accS.csv'
            accS_path = os.path.join(self.label_dir, accS_filename)
            accS = CSVWriter.read2DFloatLabelPerRow(accS_path)
            accS = np.concatenate([accS[2:accS.shape[0], :], np.zeros((min(max(0, len(accS) - 2), 2), accS.shape[1]))])
            accS_clip = accS[frame_sampling_idx, :2]
            acc_dirs2D = self.acc_dir_sampler.generate_2D_directions()
            accS_clip_labels = np.clip(np.transpose(np.matmul(accS_clip, acc_dirs2D)), 0.0, np.Inf)

        velS_clip_labels = np.empty((0, num_frames), dtype=np.float32)
        if self.vel_dir_sampler is not None:
            velS_filename = label_fileWOExt + '_envelope_velS.csv'
            velS_path = os.path.join(self.label_dir, velS_filename)
            velS = CSVWriter.read2DFloatLabelPerRow(velS_path)
            velS_clip = velS[frame_sampling_idx, :2]
            vel_dirs2D = self.vel_dir_sampler.generate_2D_directions()
            velS_clip_labels = np.clip(np.transpose(np.matmul(velS_clip, vel_dirs2D)), 0.0, np.Inf)

        eventfulness = np.zeros(num_frames)
        eventfulness[clip_beat_idx] = np.power(clip_motion_count, 0.7)
        gauss_filter = torch.tensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136]).to(torch.float32)
        eventfulness_tch = torch.from_numpy(eventfulness).to(torch.float32)
        eventfulness_label = GaussianKernelGenerator.convolve(eventfulness_tch,
                                                              gauss_filter)
            # .numpy()

        blurred_labels = torch.empty((0, num_frames), dtype=torch.float32)
        if self.gaussFilterGen is not None:
            blurred_labels = GaussianKernelGenerator.convolve(eventfulness_tch,
                                                              self.gaussFilterGen.torch_kernels())

        # labels = np.concatenate([eventfulness_label[np.newaxis, :],
        #                          accS_clip_labels, velS_clip_labels, blurred_labels])
        # print(f"type eventfulness label {type(eventfulness_label)}")
        labels = np.concatenate([torch.unsqueeze(eventfulness_label, 0),
                                 accS_clip_labels, velS_clip_labels, blurred_labels])
        labels_torch = torch.from_numpy(labels).float()

        return labels_torch
            # , clip_beat_idx