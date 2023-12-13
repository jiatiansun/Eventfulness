from .utils import list_dir
from deepVisualBeatUtil.LabelLoader.accVelLabelLoader import AccVelLabelLoader
from .folder import make_dataset
from .video_utils import VideoClips
from .vision import VisionDataset
import torch
import numpy as np
import csv
import os
from os.path import dirname as dirname
from torch.utils.data import Dataset
import math

from .simulation_util import make_random_flicker_video

def scaleLabelByName(expected_dist, label_filename):
    scale = 1.0
    if "_0.3_0.1" in label_filename or "_0.3_0.08" in label_filename:
        scale = expected_dist / 0.3 # * 3.33
    elif "_0.6_0.1" in label_filename:
        scale = expected_dist / 0.6 # * 1.67
    elif "_1_0.2" in label_filename:
        scale = expected_dist / 1.0 # * 1
    elif "_2_0.4" in label_filename:
        scale = expected_dist / 2.0 # * 0.5
    elif "_0.5_0.1" in label_filename:
        scale = expected_dist / 0.5
    elif "_1_envelope" in label_filename:
        scale = expected_dist / 2.0 # * 0.5
    elif "_2_envelope" in label_filename:
        scale = expected_dist / 1.0 # * 1
    elif "_3_envelope" in label_filename:
        scale = expected_dist / 2.0 * 3.0 # * 1.5
    elif "_4_envelope" in label_filename:
        scale = expected_dist / 0.5 # * 2

    # print(f"label_filename {label_filename} scale {scale}")
    return scale

# class (VisionDataset):
class InfiniteRandomClip(Dataset):
    """Infinite Random Clip dataset.
       Each single time being requested for a video clp,
       it would return a video clip with random background
       and a flickering pixel """

    def __init__(self, nbatch, batch_size, frames_per_clip, h, w, c, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.batch_size = batch_size
        self.nbatch = nbatch
        self.nframe = frames_per_clip
        self.size = nbatch * batch_size
        self.h = h
        self.w = w
        self.c = c
        self.transform = transform


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        video, label = make_random_flicker_video(self.nframe, self.h, self.w, self.c)

        video = torch.from_numpy(video).float()
        label = torch.from_numpy(label).float()
        if self.transform is not None:
            video = self.transform(video)

        return video, torch.empty(self.nframe), label

class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, frames_per_clip, blur=False, step_between_clips=1, frame_rate=None,
                 extensions=('avi',), transform=None, _precomputed_metadata=None,
                 num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, _audio_channels=0,
                 peakPickModelPath=None, slidingWindowTest=False, labelType="original",
                 subSample=1, subSampleData=1,
                 labelLoader=None
                 ):
        super(Kinetics400, self).__init__(root)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
            slidingWindowTest=slidingWindowTest,
            subsample_rate=subSampleData
        )
        self.subsample_rate=subSampleData
        self.target_fps = frame_rate
        self.labelType = labelType
        self.transform = transform
        self.blur = blur
        self.peakPickMPath = peakPickModelPath
        self.peakPickM = None
        self.labelLoader = labelLoader
        self.subSample = subSample

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def subSampleData(self, video, label, num_frames, subSampleRate):
        num_label = label.size(0) // subSampleRate
        end = subSampleRate * num_label
        label_strided = torch.as_strided(label, (num_label, subSampleRate), (subSampleRate, 1))
        label_temp = torch.max(label_strided, 1).values
        if end < label.size(0):
            last = torch.max(label[end:])
            new_label = torch.cat((label_temp, torch.tensor(last)))
        else:
            new_label = label_temp

        new_video = video[0:num_frames:subSampleRate, :, :, :]

        return new_video, new_label

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        num_frames = self.video_clips.num_frames
        ov_fps = self.video_clips.get_original_fps(idx)

        # indices of frames picked out from the original video
        frame_sampling_idx = self.video_clips.get_sampled_video_frame_idx(idx)
        resampled_num_frame = self.video_clips.get_resampled_vid_len(idx) # number of frames in the resampled video
        video_path = info['video_path']
        video_path_n = info['video_path_len']
        resampled_fps = self.target_fps
        if self.target_fps is None:
            resampled_fps = ov_fps / self.subsample_rate

        startT = frame_sampling_idx[0].item() / float(ov_fps)
        endT = frame_sampling_idx[-1].item() / float(ov_fps)

        label_filename = os.path.basename(video_path)
        label_fileWOExt = os.path.splitext(label_filename)[0]

        self.labelLoader.setLabelDir(video_path)
        label = self.labelLoader.loadLabel(label_fileWOExt, ov_fps, resampled_fps, num_frames,
                                   frame_sampling_idx, startT, endT)

        if self.transform is not None:
            video = self.transform(video)
        return video, label, video_path, video_path_n, ov_fps, resampled_num_frame

