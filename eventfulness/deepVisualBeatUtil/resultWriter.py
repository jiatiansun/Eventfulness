import os
from .fileAndMediaWriters import CSVWriter, JsonReadWriter, OSUtil
import numpy as np
import scipy as sp
from scipy.fft import fft, ifft
import librosa

class ResultPathSystem(object):
    def __init__(self, data_dir, resultDir="results"):
        self.resultDir = resultDir
        self.data_dir = data_dir
        self.data_results_dir = os.path.join(data_dir, resultDir)
        OSUtil.safe_mkdir(self.data_results_dir)

    def createResultSubDirFromName(self, subDirName):
        subdir = os.path.join(self.data_results_dir, subDirName)
        if os.path.exists(subdir):
            return
        OSUtil.safe_mkdir(subdir)
        return subdir

    def createResultSubDirFromPath(self, path):
        filename = os.path.basename(path)
        ext_idx = filename.rfind(".")
        if(filename[ext_idx:] != ".mp4"):
            print("The video path you try to create prediction for does't have .mp4 extension")
            assert(True)
        subDirName = filename[:ext_idx]
        subdir = os.path.join(self.data_results_dir, subDirName)
        if os.path.exists(subdir):
            return
        OSUtil.safe_mkdir(subdir)
        return subdir

    def getVideoResultWriterFromPath(self, path):
        subdir = self.createResultSubDirFromPath(path)
        return VideoResult(subdir)

    def getVideoResultWriterFromName(self, subDirName):
        subdir = self.createResultSubDirFromName(subDirName)
        return VideoResult(subdir)

    def initResults(self, func):
        func(self)

    def traverseResultsWInit(self, init, func):
        init(self)
        for subdir in os.listdir(self.data_results_dir):
            subdir_path = os.path.join(self.data_results_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            func(self, VideoResult(subdir_path))

    def traverseResults(self, func):
        for subdir in os.listdir(self.data_results_dir):
            subdir_path = os.path.join(self.data_results_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            func(self, VideoResult(subdir_path))

class VideoResult(object):
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.name = os.path.basename(result_dir)
        self.configFilePath = os.path.join(result_dir, "config.json")
        self.video_path = None
        self.fps = 0
        self.eventfulness = None
        self.label = None
        self.impactEnvelope = None
        self.audioEnvelope = None

        self.loadConfig()
    
    @staticmethod
    def normalize(x):
        min = np.min(x)
        max = np.max(x)
        return (x -min)/(max-min) + min

    def setConfig(self, key, value):
        if self.config is None:
            self.config = dict()
        self.config[key] = value

    def setVideoPath(self, video_path):
        if video_path[-4:] != ".mp4":
            print(f"Cannot save video path {video_path} because its extension is not mp4)")
            assert(False)
        self.video_path = video_path
        self.setConfig("video_path", video_path)

    def setFPS(self, fps):
        self.fps = fps
        self.setConfig("fps", fps)

    def loadConfig(self):
        self.config = None
        if(os.path.exists(self.configFilePath)):
            self.config = JsonReadWriter.readFromFile(self.configFilePath)
            self.__dict__.update(self.config)

        if self.eventfulness is not None:
            self.eventfulness = np.array(self.eventfulness)
        if self.label is not None:
            self.label = np.array(self.label)

    def saveConfig(self):
        if self.config is not None:
            JsonReadWriter.writeToFile(self.config, self.configFilePath)

    def setEventfulness(self, eventfulness):
        self.setConfig("eventfulness", eventfulness.tolist())
        self.eventfulness = eventfulness

    def setLabel(self, label):
        self.setConfig("label", label.tolist())
        self.label = label

    def setImpactEnvelope(self, impactEnvelope):
        self.setConfig("impactEnvelope", impactEnvelope.tolist())
        self.impactEnvelope = impactEnvelope

    def setAudioEnvelope(self, audioEnvelope):
        self.config["audioEnvelope"] = audioEnvelope.tolist()
        self.audioEnvelope = audioEnvelope

    @staticmethod
    def peakpickForFrame(preds):
        pre_max = 2
        post_max = 2
        pre_avg = 2
        post_avg = 2
        delta = 0.05 * np.max(preds)
        wait = 2
        preds = np.clip(preds, 0, np.max(preds))
        peak_frames = librosa.util.peak_pick(preds, pre_max, post_max, pre_avg, post_avg, delta, wait)
        return peak_frames

    @staticmethod
    def peakpickFixedForFrame(preds):
        pre_max = 2
        post_max = 2
        pre_avg = 2
        post_avg = 2
        delta = 0.05
        wait = 2
        peak_frames = librosa.util.peak_pick(preds, pre_max, post_max, pre_avg, post_avg, delta, wait)
        return peak_frames

    def setVideoAndFPS(self, path, fps):
        self.setVideoPath(path)
        self.setFPS(fps)

    def setResult(self, path, fps, eventfulness, label):
        self.setVideoPath(path)
        self.setFPS(fps)
        self.setEventfulness(eventfulness)
        self.setLabel(label)

    def saveResult(self):
        self.saveConfig()




