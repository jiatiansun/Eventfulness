from .labelLoader import *
from .directionSampler import *
from deepVisualBeatUtil.gaussianFilterGenerator import *
from deepVisualBeatUtil.fileAndMediaWriters import CSVWriter

import os
from os.path import dirname as dirname
import torch
import numpy as np

class AccVelNoRotLabelLoader(LabelReadWriter):

    def __init__(self, gaussFilterGen):
        self.gaussFilterGen = gaussFilterGen

    def setLabelDir(self, videoPath):
        self.label_dir = dirname(dirname(os.path.realpath(videoPath))) + '_beatLabel'

    def getLabelDir(self):
        return self.label_dir

    @staticmethod
    def getName():
        return "accVelNoRot"

    def getNumLabel(self):
        return 19

    def readVecFromFile(self, label_fileWOExt, suffix):
        filename = label_fileWOExt + '_envelope' + suffix + '.csv'
        filepath = os.path.join(self.label_dir, filename)
        return CSVWriter.read2DFloatLabelPerRow(filepath)

    def readFloatFromFile(self, label_fileWOExt, suffix):
        filename = label_fileWOExt + '_envelope' + suffix + '.csv'
        filepath = os.path.join(self.label_dir, filename)
        return CSVWriter.readFloatLabelPerRow(filepath)

    def timeStamps2FrameIdx(self, clip_idx, filter_beat, label_fileWOExt, suffix):
        labelAtTS = self.readFloatFromFile(label_fileWOExt, suffix)
        clip_label = labelAtTS[clip_idx]
        clip_beat_label = clip_label[filter_beat]

        return clip_beat_label


    def timeStamps2Label(self, clip_idx, clip_beat_idx, filter_beat, label_fileWOExt, suffix, num_frame):
        frameIdx = self.timeStamps2FrameIdx(clip_idx, filter_beat, label_fileWOExt, suffix)
        label = np.zeros((1, num_frame))
        label[0, clip_beat_idx] = frameIdx
        return label

    def loadLabel(self, label_fileWOExt, ov_fps, target_fps, num_frames, frame_sampling_idx, startT, endT):

        keyFrames = self.readFloatFromFile(label_fileWOExt, "")
        keyFrame_time_stamps = keyFrames[1:]
        # duration =  keyFrames[0]

        clip_idx = np.logical_and(keyFrame_time_stamps >= startT, keyFrame_time_stamps <= endT)
        clip_time_stamps = keyFrame_time_stamps[clip_idx]
        beat_idx = np.floor((clip_time_stamps - startT) * target_fps).astype(int)

        # filter_beat: index of the beat for this video clip's label
        filter_beat = np.logical_and(beat_idx >= 0, beat_idx < num_frames)
        clip_beat_idx = beat_idx[filter_beat]

        clip_motion_count = self.timeStamps2FrameIdx(clip_idx, filter_beat, label_fileWOExt, "_val")
        # dtls = self.timeStamps2Label(clip_idx, clip_beat_idx, filter_beat, label_fileWOExt, "_dtl", num_frames)
        # dtrs = self.timeStamps2Label(clip_idx, clip_beat_idx, filter_beat, label_fileWOExt, "_dtr", num_frames)
        # dals = self.timeStamps2Label(clip_idx, clip_beat_idx, filter_beat, label_fileWOExt, "_dal", num_frames)
        # dars = self.timeStamps2Label(clip_idx, clip_beat_idx, filter_beat, label_fileWOExt, "_dar", num_frames)

        accSPos = np.transpose(self.readVecFromFile(label_fileWOExt, "_accSPos")[frame_sampling_idx, :2])
        accSNeg = np.transpose(self.readVecFromFile(label_fileWOExt, "_accSNeg")[frame_sampling_idx, :2])
        velSPos = np.transpose(self.readVecFromFile(label_fileWOExt, "_velSPos")[frame_sampling_idx, :2])
        velSNeg = np.transpose(self.readVecFromFile(label_fileWOExt, "_velSNeg")[frame_sampling_idx, :2])
        angSPos = np.transpose(self.readVecFromFile(label_fileWOExt, "_angSPos")[frame_sampling_idx, :3])
        angSNeg = np.transpose(self.readVecFromFile(label_fileWOExt, "_angSNeg")[frame_sampling_idx, :3])
        # ammtSPos = np.transpose(self.readVecFromFile(label_fileWOExt, "_ammtSPos")[frame_sampling_idx, :3])
        # ammtSNeg = np.transpose(self.readVecFromFile(label_fileWOExt, "_ammtSNeg")[frame_sampling_idx, :3])
        # lmmtSPos = np.transpose(self.readVecFromFile(label_fileWOExt, "_lmmtSPos")[frame_sampling_idx, :2])
        # lmmtSNeg = np.transpose(self.readVecFromFile(label_fileWOExt, "_lmmtSNeg")[frame_sampling_idx, :2])

        eventfulness = np.zeros(num_frames)
        eventfulness[clip_beat_idx] = np.power(clip_motion_count, 0.7)
        gauss_filter = torch.tensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136]).to(torch.float32)
        eventfulness_tch = torch.from_numpy(eventfulness).to(torch.float32)
        eventfulness_label = GaussianKernelGenerator.convolve(eventfulness_tch,
                                                              gauss_filter)

        blurred_labels = torch.empty((0, num_frames), dtype=torch.float32)
        if self.gaussFilterGen is not None:
            blurred_labels = GaussianKernelGenerator.convolve(eventfulness_tch,
                                                              self.gaussFilterGen.torch_kernels())

        labels = np.concatenate([torch.unsqueeze(eventfulness_label, 0),
                                 accSPos, accSNeg,
                                 velSPos, velSNeg,
                                 blurred_labels,
                                 angSPos, angSNeg])

        labels_torch = torch.from_numpy(labels).float()
        return labels_torch