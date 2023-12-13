import bisect
import math
from fractions import Fraction
from typing import List

import torch
from ..io import (
    _probe_video_from_file,
    _read_video_from_file,
    _read_video_timestamps_from_file,
    read_video,
    read_video_timestamps,
)

from .utils import tqdm


def pts_convert(pts, timebase_from, timebase_to, round_func=math.floor):
    """convert pts between different time bases
    Args:
        pts: presentation timestamp, float
        timebase_from: original timebase. Fraction
        timebase_to: new timebase. Fraction
        round_func: rounding function.
    """
    new_pts = Fraction(pts, 1) * timebase_from / timebase_to
    return round_func(new_pts)


def unfold(tensor, size, step, dilation=1):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors

    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0) # Get the dimension/shape of the tensor
    numel = tensor.numel() # return total number of elements in tensor
    new_stride = (step * o_stride, dilation * o_stride)
    clip_num_float = (numel - (dilation * (size - 1) + 1)) / step
    # look into Caroline's one note to get details for this computation
    # (numel - (dilation * (size - 1) + 1)) / step+ 1
    new_size = (int(math.ceil(clip_num_float)), size)
    # print(f"numel {numel} dilation {dilation} size {size} step {step} "
    #       f"clip num float {clip_num_float} new size {new_size}")

    # if (math.ceil(clip_num_float) - clip_num_float) > 1e-8
    if new_size[0] < 1:
        new_size = (0, size)
        return torch.as_strided(tensor, new_size, new_stride)

    woLastClip = torch.as_strided(tensor, new_size, new_stride)
    lastClip = torch.unsqueeze(tensor[-(dilation * (size - 1) + 1):],0)
    # print(f"wolastclip {woLastClip.size()} lastClip {lastClip.size()} total size {size} step {step} dilation {dilation}")

    wLastClip = torch.cat((woLastClip, lastClip), 0)
    return wLastClip


class _VideoTimestampsDataset(object):
    """
    Dataset used to parallelize the reading of the timestamps
    of a list of videos, given their paths in the filesystem.

    Used in VideoClips and defined at top level so it can be
    pickled when forking.
    """

    def __init__(self, video_paths: List[str]):
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        return read_video_timestamps(self.video_paths[idx])

class VideoClips(object):
    """
    Given a list of video files, computes all consecutive subvideos of size
    `clip_length_in_frames`, where the distance between each subvideo in the
    same video is defined by `frames_between_clips`.
    If `frame_rate` is specified, it will also resample all the videos to have
    the same frame rate, and the clips will refer to this frame rate.

    Creating this instance the first time is time-consuming, as it needs to
    decode all the videos in `video_paths`. It is recommended that you
    cache the results after instantiation of the class.

    Recreating the clips for different clip lengths is fast, and can be done
    with the `compute_clips` method.

    Arguments:
        video_paths (List[str]): paths to the video files
        clip_length_in_frames (int): size of a clip in number of frames
        frames_between_clips (int): step (in frames) between each clip
        frame_rate (int, optional): if specified, it will resample the video
            so that it has `frame_rate`, and then the clips will be defined
            on the resampled video
        num_workers (int): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. (default: 0)
    """

    def __init__(
        self,
        video_paths,
        clip_length_in_frames=16,
        frames_between_clips=1,
        frame_rate=None,
        _precomputed_metadata=None,
        num_workers=0,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _video_max_dimension=0,
        _audio_samples=0,
        _audio_channels=0,
        subsample_rate=1,
        slidingWindowTest=False
    ):

        self.video_paths = video_paths
        self.num_workers = num_workers

        # these options are not valid for pyav backend
        self._video_width = _video_width
        self._video_height = _video_height
        self._video_min_dimension = _video_min_dimension
        self._video_max_dimension = _video_max_dimension
        self._audio_samples = _audio_samples
        self._audio_channels = _audio_channels
        self.isSlidingWindowTest = slidingWindowTest

        if _precomputed_metadata is None:
            self._compute_frame_pts()
        else:
            self._init_from_metadata(_precomputed_metadata)

        if self.isSlidingWindowTest:
            frame_rate = self.video_fps[0]
        # print(f"sliding window{self.isSlidingWindowTest} frame_rate {frame_rate}")
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate, subsample_rate=subsample_rate)

    def _collate_fn(self, x):
        return x

    # This is the base where
    def _compute_frame_pts(self):
        self.video_pts = []
        self.video_fps = []
        self.video_tbs = []

        # strategy: use a DataLoader to parallelize read_video_timestamps
        # so need to create a dummy dataset first
        import torch.utils.data

        dl = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(self.video_paths),
            batch_size=16,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                clips, fps, tbs = list(zip(*batch))
                clips = [torch.as_tensor(c) for c in clips]
                self.video_pts.extend(clips)
                self.video_fps.extend(fps)
                self.video_tbs.extend(tbs)

    def _init_from_metadata(self, metadata):
        self.video_paths = metadata["video_paths"]
        assert len(self.video_paths) == len(metadata["video_pts"])
        self.video_pts = metadata["video_pts"]
        assert len(self.video_paths) == len(metadata["video_fps"])
        self.video_fps = metadata["video_fps"]

    @property
    def metadata(self):
        _metadata = {
            "video_paths": self.video_paths,
            "video_pts": self.video_pts,
            "video_fps": self.video_fps,
        }
        return _metadata

    def subset(self, indices):
        video_paths = [self.video_paths[i] for i in indices]
        video_pts = [self.video_pts[i] for i in indices]
        video_fps = [self.video_fps[i] for i in indices]
        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        return type(self)(
            video_paths,
            self.num_frames,
            self.step,
            self.frame_rate,
            _precomputed_metadata=metadata,
            num_workers=self.num_workers,
            _video_width=self._video_width,
            _video_height=self._video_height,
            _video_min_dimension=self._video_min_dimension,
            _video_max_dimension=self._video_max_dimension,
            _audio_samples=self._audio_samples,
            _audio_channels=self._audio_channels,
        )

    @staticmethod
    def compute_clips_for_video(video_path, video_pts, num_frames, step, fps, frame_rate, subsample_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps / subsample_rate
        total_frames = len(video_pts) * (float(frame_rate) / fps) # resampled num of frames
        original_num_frames = len(video_pts) # original num of frames

        # if frame_rate != fps:
        #     print(f"error video path {video_path} fps {fps} frame_rate {frame_rate}")
        # else:
        #     print(f"correct video path {video_path}")

        # print(f"total number of frames {total_frames} original nframes {original_num_frames} pts num {len(video_pts)} video clip size {num_frames} frame_rate {frame_rate} fps {fps} ")
        idxs = VideoClips._resample_video_idx(
            int(math.floor(total_frames)), fps, frame_rate
        ) # This gives us resampled idxes, it could be a slice, this contains the idxs for entire video
        # print(f"video path {video_path}  idxs size {int(math.floor(total_frames))}")
        # print(f"1video clip idx content {idxs}")

        video_pts = video_pts[idxs] #resampled frames for entire video
        clips = unfold(video_pts, num_frames, step) #resampled frame clips
        output_idxs = torch.tensor([])
        if isinstance(idxs, slice):

            ov_idxs = torch.arange(original_num_frames) #original idx
            # resampled_idx = ov_idxs[idxs] #resampled idx
            output_idxs = unfold(ov_idxs, num_frames, step)

            idxs = [idxs] * len(clips)
            # torch.tensor(list(range(0, original_num_frames)))

            resampled_idxs = unfold(torch.arange(int(math.floor(total_frames))), num_frames, step)
            # resampled_idxs = unfold(torch.tensor(list(range(0,int(math.floor(total_frames))))), num_frames, step)
            # print(f"first option: num_frames {num_frames} original total frame {original_num_frames} total_frames {total_frames} "
            #       f"frame_rate {frame_rate} fps {fps} start pts {video_pts[0]} 2nd pts {video_pts[1]} end_pts {video_pts[-1]} start {output_idxs[-1, 0]} 2nd {output_idxs[-1, 1]} end {output_idxs[-1, -1]} output_idxs {output_idxs.size()} clips size {clips.size()}")
        else:
            idxs = unfold(idxs, num_frames, step)
            output_idxs = idxs
            resampled_idxs = unfold(torch.arange(int(math.floor(total_frames))), num_frames, step)
            # resampled_idxs = unfold(torch.tensor(list(range(0,int(math.floor(total_frames))))), num_frames, step)
            # print(f"second option: num_frames {num_frames} original total frame {original_num_frames} total_frames {total_frames} "
            #       f"frame_rate {frame_rate} fps {fps} start pts {video_pts[0]} 2nd pts {video_pts[1]} end_pts {video_pts[-1]} start {output_idxs[-1, 0]} 2nd {output_idxs[-1, 1]} end {output_idxs[-1, -1]} output idxs shape {output_idxs.size()} clips size {clips.size()}")

        # print(f"2video_pts clip idx size {output_idxs.size()} idx start {output_idxs[0]} idx end {output_idxs[-1]}")
        return int(math.floor(total_frames)), clips, idxs, output_idxs, resampled_idxs

    def compute_clips(self, num_frames, step, frame_rate=None, subsample_rate=1):
        """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.

        Arguments:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
        """
        self.num_frames = num_frames
        self.step = step
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        self.tbs = []
        self.fpss = []
        self.resampled_vid_len = []
        self.idxWCurrFPSs = []
        for video_path, video_pts, fps, tb in zip(self.video_paths,self.video_pts, self.video_fps, self.video_tbs):
            vidLen, clips, idxs, out_idxs, idxWCurrFPS = self.compute_clips_for_video(
                video_path, video_pts, num_frames, step, fps, frame_rate, subsample_rate
            )
            self.resampled_vid_len.append(vidLen)
            self.clips.append(clips)
            self.resampling_idxs.append(out_idxs)
            self.idxWCurrFPSs.append(idxWCurrFPS)
            self.fpss.append(fps)
            self.tbs.append(tb)
        clip_lengths = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def get_tb(self, idx):
        """
        Gets a subclip from a list of videos.

        Arguments:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )


        video_idx, clip_idx = self.get_clip_location(idx)
        return self.tbs[video_idx]

    def get_resampled_vid_len(self, idx):
        """
        Gets a subclip from a list of videos.

        Arguments:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )


        video_idx, clip_idx = self.get_clip_location(idx)
        return self.resampled_vid_len[video_idx]

    def get_original_fps(self, idx):
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )
        video_idx, clip_idx = self.get_clip_location(idx)
        v_fps = self.fpss[video_idx]

        return v_fps

    def get_sampled_video_frame_idx(self, idx):
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )
        video_idx, clip_idx = self.get_clip_location(idx)
        resampling_idx = self.resampling_idxs[video_idx][clip_idx]
        return resampling_idx

    def get_sampled_video_frame_clip_idx(self, idx):
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )
        video_idx, clip_idx = self.get_clip_location(idx)
        resampling_idx = self.idxWCurrFPSs[video_idx][clip_idx]
        return resampling_idx

    def get_clip(self, idx):
        """
        Gets a subclip from a list of videos.

        Arguments:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]



        # print('first clip size{} first several clip_pts {} {} {}'.format(len(self.clips[0]),
        #                                                                self.clips[0][0],
        #                                                                self.clips[0][1],
        #                                                             self.clips[0][2]))
        from torchvision_custom import get_video_backend

        backend = get_video_backend()
        # print(f"backend {backend}")
        if backend == "pyav":
            # check for invalid options
            if self._video_width != 0:
                raise ValueError("pyav backend doesn't support _video_width != 0")
            if self._video_height != 0:
                raise ValueError("pyav backend doesn't support _video_height != 0")
            if self._video_min_dimension != 0:
                raise ValueError(
                    "pyav backend doesn't support _video_min_dimension != 0"
                )
            if self._video_max_dimension != 0:
                raise ValueError(
                    "pyav backend doesn't support _video_max_dimension != 0"
                )
            if self._audio_samples != 0:
                raise ValueError("pyav backend doesn't support _audio_samples != 0")

        if backend == "pyav":
            # print(f"start pyav branch")
            start_pts = clip_pts[0].item()
            end_pts = clip_pts[-1].item()
            video, audio, info = read_video(video_path, start_pts, end_pts)
            info['start_pts'] = start_pts
            info['end_pts'] = end_pts
            info['video_path'] = video_path
            info['video_path_len'] = len(self.clips[video_idx])
            # print(f"pyav video_path_len {info['video_path_len']} start {start_pts} end {end_pts}")
        else:
            # print(f"start other branch")
            info = _probe_video_from_file(video_path)
            video_fps = info.video_fps
            audio_fps = None

            video_start_pts = clip_pts[0].item()
            video_end_pts = clip_pts[-1].item()
            info['start_pts'] = video_start_pts
            info['end_pts'] = video_end_pts
            info['video_path'] = video_path
            info['video_path_len'] = len(self.clips[video_idx])

            audio_start_pts, audio_end_pts = 0, -1
            audio_timebase = Fraction(0, 1)
            video_timebase = Fraction(
                info.video_timebase.numerator, info.video_timebase.denominator
            )
            if info.has_audio:
                audio_timebase = Fraction(
                    info.audio_timebase.numerator, info.audio_timebase.denominator
                )
                audio_start_pts = pts_convert(
                    video_start_pts, video_timebase, audio_timebase, math.floor
                )
                audio_end_pts = pts_convert(
                    video_end_pts, video_timebase, audio_timebase, math.ceil
                )
                audio_fps = info.audio_sample_rate
            # print('video_timebase:{} audio_timebase: {} video_start_pts{} end{} audio_start_pts{} end{} '.format(
            #     video_timebase, audio_timebase, video_start_pts, video_end_pts))
            video, audio, info = _read_video_from_file(
                video_path,
                video_width=self._video_width,
                video_height=self._video_height,
                video_min_dimension=self._video_min_dimension,
                video_max_dimension=self._video_max_dimension,
                video_pts_range=(video_start_pts, video_end_pts),
                video_timebase=video_timebase,
                audio_samples=self._audio_samples,
                audio_channels=self._audio_channels,
                audio_pts_range=(audio_start_pts, audio_end_pts),
                audio_timebase=audio_timebase,
            )

            info = {"video_fps": video_fps}
            if audio_fps is not None:
                info["audio_fps"] = audio_fps

            # print(f"read from file video_path_len {info['video_path_len']} start {video_start_pts} end {video_end_pts}")

        if self.frame_rate is not None:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            if isinstance(resampling_idx, torch.Tensor):
                # print(f"video {video_path} size {video.size()} resampling idx size {resampling_idx.size()} max {torch.max(resampling_idx)} min {torch.min(resampling_idx)}")
                resampling_idx = resampling_idx - resampling_idx[0]
                video = video[resampling_idx]
            # try:
            #     video = video[resampling_idx]
            #     print(f"success: {video_path}")
            # except:
            #     print(f"have trouble loading {video_path}")
            info["video_fps"] = self.frame_rate
        else:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            # print(f"video size {video.size()} resampling idx size {resampling_idx.size()} resampling idx {resampling_idx}")
            if isinstance(resampling_idx, torch.Tensor):
                try:
                    resampling_idx = resampling_idx - resampling_idx[0]
                    video = video[resampling_idx]
                except:
                    currFPS_idx = self.idxWCurrFPSs[video_idx][clip_idx]
                    currFPS_idx = currFPS_idx - currFPS_idx[0]
                    print(f"video size {video.size()} currFPS_idx {currFPS_idx}")
                    raise Exception(f"video size {video.size()} resampling idx size {resampling_idx.size()} resampling idx {resampling_idx}")

            # try:
            #     video = video[resampling_idx]
            #     print(f"success: {video_path}")
            # except:
            #     print(f"have trouble loading {video_path}")
            info["video_fps"] = self.frame_rate
        # try:
        video_err_str = f"video {video_path} start pts {info['start_pts']} end pts {info['end_pts']} idx start {resampling_idx[0]} idx end {resampling_idx[-1]}"
        assert len(video) == self.num_frames, "{} x {} {} ".format(
            video.shape, self.num_frames, video_err_str
        )
        # except:
        #     print(f"video path is too short for our desire {video_path}")
        return video, audio, info, video_idx
