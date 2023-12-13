from abc import ABC, abstractmethod

class LabelReadWriter(ABC):
    @abstractmethod
    def setLabelDir(self, videoPath):
        pass

    @abstractmethod
    def getLabelDir(self):
        pass

    @abstractmethod
    def getName(self):
        pass

    @abstractmethod
    def loadLabel(self, label_fileWOExt, ov_fps, target_fps, num_frames, frame_sampling_idx, startT, endT):
        pass

    @abstractmethod
    def getNumLabel(self):
        pass

    # @abstractmethod
    # def writeLabel(self, label_fileWOExt):
    #     pass