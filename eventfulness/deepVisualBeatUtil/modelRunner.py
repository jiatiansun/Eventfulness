import time
import math
import os
import psutil
import torch
import torch.nn as nn
import numpy as np
import copy
# from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError
from torchmetrics.functional import mean_absolute_percentage_error

from .fileAndMediaWriters import CSVWriter, JsonReadWriter
from .resultWriter import ResultPathSystem

class TimeMark(object):
    def __init__(self, name, hash, t, cpu_usage):
        self.name = name
        self.hash = hash
        self.t = t
        self.cpu_usage = cpu_usage

    def changeT(self, t):
        self.t = t

class BenchMarkSystem(object):
    @staticmethod
    def getCurrTime():
        return time.time()

    @staticmethod
    def getCurrCPUUsage():
        return psutil.virtual_memory().percent

    def printGPUUsage(self):
        if not self.optimization:
            return
        printout = os.system("nvidia-smi")
        print(f"GPU usage: {printout}")

    def __init__(self, name="no name process"):
        self.name=name
        self.time_marks_name2hash = {}
        self.time_marks_hash2TimeMark = {}
        self.time_marks_accumTime = {}

    def setName(self, name):
        self.name = name
    def setOptimization(self, optimization):
        self.optimization = optimization

    def startTiming(self):
        if not self.optimization:
            return
        # self.addTimeMark("start")

    def addTimeMark(self, name):
        if not self.optimization:
            return
        curr_t = BenchMarkSystem.getCurrTime()
        curr_cpu_usage = BenchMarkSystem.getCurrCPUUsage()
        if name in self.time_marks_name2hash:
            hash = self.time_marks_name2hash[name]
            curr_timeMark = self.time_marks_hash2TimeMark[hash]
            curr_timeMark.changeT(curr_t)
        else:
            hash = len(self.time_marks_name2hash)
            self.time_marks_name2hash[name] = hash
            self.time_marks_hash2TimeMark[hash] = TimeMark(name, hash, curr_t, curr_cpu_usage)

    def accumulateTime(self):
        if not self.optimization:
            return
        prev_timeMark = None

        for hash in sorted(self.time_marks_hash2TimeMark.keys()):
            if isinstance(prev_timeMark, type(None)):
                prev_timeMark = self.time_marks_hash2TimeMark[hash]
                continue

            timeMark = self.time_marks_hash2TimeMark[hash]
            if hash not in self.time_marks_accumTime:
                self.time_marks_accumTime[hash] = TimeMark(timeMark.name, hash, timeMark.t - prev_timeMark.t, timeMark.cpu_usage)
            else:
                accum_time = self.time_marks_accumTime[hash]
                self.time_marks_accumTime[hash] = TimeMark(timeMark.name, hash, timeMark.t - prev_timeMark.t + accum_time.t,
                                                           timeMark.cpu_usage)
            prev_timeMark = timeMark

    def printBenchMark(self):
        if not self.optimization:
            return

        total_time = 0.0
        for hash in sorted(self.time_marks_accumTime.keys()):
            timeMark = self.time_marks_accumTime[hash]
            total_time = total_time + timeMark.t

        print("---" * 10 + f'{self.name} time and cpu usage report' + "---" * 10)
        for hash in sorted(self.time_marks_accumTime.keys()):
            timeMark = self.time_marks_accumTime[hash]
            print(
                f"{timeMark.name}: time spent: {timeMark.t} percentage: {timeMark.t / total_time} cpu percentage: {timeMark.cpu_usage}")
        print("-----------" * 10)

class ModelRunner(object):

    def __init__(self, visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter):
        self.model = visBeatModel
        self.config = config
        self.debugger = debugger
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.reporter = reporter
        self.benchMarker = BenchMarkSystem()


    def run(self):
        pass


class TrainModelRunner(ModelRunner):
    def __init__(self,visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter):
        super().__init__(visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter)
        self.clearEpochLoss()

    def clearEpochLoss(self):
        self.epoch_loss = 0.0
        self.epoch_correct_preds = 0
        self.epoch_clip_num = 0

    def addEpochLoss(self, epoch_loss, epoch_correct_preds, epoch_clip_num):
        self.epoch_loss += epoch_loss
        self.epoch_correct_preds += epoch_correct_preds
        self.epoch_clip_num += epoch_clip_num

    def clearBatchCount(self, dataloader):
        self.epoch_batch_num = len(dataloader)
        self.epoch_batch_count = 0

    def computeBatchPercentage(self):
        return float(self.epoch_batch_count) / self.epoch_batch_num

    def computeLossAndAcc(self):
        return self.epoch_loss / self.epoch_clip_num, self.epoch_correct_preds.double()/self.epoch_clip_num

    def increaseBatchCount(self):
        self.epoch_batch_count +=1

    def run(self, optimization=False, processName="general training process"):
        self.benchMarker.setName(processName)
        self.benchMarker.setOptimization(optimization)
        self.benchMarker.startTiming()
        device = self.model.getDevice()

        for epoch in range(self.config.nepoch):
            print('Epoch {}/{}'.format(epoch, self.config.nepoch-1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in sorted(self.dataloaders.keys()):
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                self.clearEpochLoss()
                self.clearBatchCount(self.dataloaders[phase])
                print("batch num is {}".format(self.epoch_batch_num))

                # Iterate over data.
                self.benchMarker.addTimeMark("epoch_start")
                for inputs, labels, video_paths in self.dataloaders[phase]:
                    self.benchMarker.addTimeMark("Load Data")
                    if(self.epoch_batch_count % 5 == 0):
                        print(f"batch {self.epoch_batch_count} / {self.epoch_batch_num}")
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad(set_to_none=True)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        unshaped_output = self.model.run(inputs)
                        labels = torch.reshape(labels, (labels.size(0),-1))
                        outputs = torch.reshape(unshaped_output, (unshaped_output.size(0),  -1))
                        self.benchMarker.addTimeMark("Forward Propagation time")
                        loss = self.criterion(outputs, labels)

                        if (self.model.model_type == "discriminant"):
                            sig = nn.Sigmoid()(outputs)
                        elif (self.model.model_type == "regressive"):
                            sig = outputs

                        self.benchMarker.addTimeMark("Evaluate Loss Function")
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            if self.scheduler != None:
                                self.scheduler.step(epoch + self.computeBatchPercentage())
                                print("learning rate is {}".format(self.optimizer.param_groups[0]['lr']))

                        self.benchMarker.addTimeMark("Backward Propagation")

                    if self.model.model_type == "discriminant":
                        preds = torch.where(sig > 0.5, 1.0, 0.0)
                        curr_epoch_loss = loss.item() * inputs.size(0)
                        curr_epoch_corrects = torch.sum(preds == labels, (0, 1)) / float(labels.size(1))
                    elif self.model.model_type == "regressive":
                        preds = sig
                        curr_epoch_loss = loss.item() * inputs.size(0)
                        curr_epoch_corrects = torch.sum(torch.square(preds - labels) < 0.1, (0, 1)) / float(labels.size(1))

                    self.addEpochLoss(curr_epoch_loss, curr_epoch_corrects, inputs.size(0))
                    self.benchMarker.addTimeMark("Loss Computation Time")

                    print(f"epoch: {epoch}, phase {phase}")
                    self.reporter.updateLossAccuracy(phase, loss, sig, preds, labels)
                    self.benchMarker.printGPUUsage()


                    if (phase == 'train' and epoch % 2 == 0 and self.epoch_batch_count > math.floor(self.epoch_batch_num * 0.95)):
                        self.reporter.writePredictionImage(epoch, self.epoch_clip_num, self.model.model_type,
                                                            loss, preds, labels, inputs, sig, video_paths)

                    self.increaseBatchCount()
                    self.benchMarker.addTimeMark("Report Time")
                    self.benchMarker.accumulateTime()
                    self.benchMarker.printBenchMark()
                    self.benchMarker.addTimeMark("epoch_start")

                #Update the tensorboard after we get data for one epoch.
                self.reporter.tensorBoardWriter.flushTensorBoard()

                # Compute loss and acc for one epoch
                epoch_loss, epoch_acc = self.computeLossAndAcc()

                self.model.updateBestModel(epoch_loss)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'val' and epoch % 1 == 0:
                    # Save Model
                    self.model.saveModel(self.reporter.filePaths.dir, epoch, self.optimizer, loss)
                
        print('Best Val loss: {:4f}'.format(self.model.best_model_loss))

        # load best model weights
        self.model.saveBestModel(self.reporter.filePaths.dir, self.optimizer)
        return self.model

class TestModelRunner(ModelRunner):
    def __init__(self, visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter):
        super().__init__(visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter)
        self.resultPaths = ResultPathSystem(self.config.data_dir, resultDir=self.config.result_dir)
        self.clearPrediction()

    def clearPrediction(self):

        self.curr_prediction = []
        self.curr_label = []
        self.curr_video_frames = None
        self.curr_test_video_path = ""
        self.curr_test_video_nframe = 0
        self.curr_test_video_predicted_frame = self.config.prediction_window_step // self.config.subSample
        self.curr_fps = -1

        print(f"clear batch curr predicted frame {self.curr_test_video_predicted_frame}")

    def clearBatchCount(self, dataloader):
        self.epoch_batch_num = len(dataloader)
        self.epoch_batch_count = 0

    def increaseBatchCount(self):
        self.epoch_batch_count += 1

    def savePrediction(self, prediction, label, video_path):
        video_name = os.path.basename(video_path)
        id = video_name[:-4]
        filename = f"{id}"
        prediction_path = os.path.join(self.reporter.filePaths.prediction_dir, filename + '_pred.csv')
        label_path = os.path.join(self.reporter.filePaths.prediction_dir, filename + '_label.csv')

        CSVWriter.writeLabel(prediction_path, prediction)
        CSVWriter.writeLabel(label_path, label)

    def saveResult(self, prediction, fps, video_path, test_label):
        resultWriter = self.resultPaths.getVideoResultWriterFromPath(video_path)
        print(f"Saving result of video path {type(video_path)}")
        print(f"fps {type(fps)}")
        print(f"resampled fps {fps/self.config.subSampleRate}")
        print(f"prediction {type(prediction)}")
        print(f"label type {type(test_label)}")
        print(f"config {resultWriter.config}")
        resultWriter.setResult(video_path, fps/self.config.subSampleRate, prediction, test_label)
        resultWriter.saveResult()

    def flattenFrames(self, samples):
        permuted_samples = torch.permute(samples, (0, 2, 3, 4, 1))
        flattened_samples = torch.reshape(permuted_samples, (-1, permuted_samples.size(2),
                                                                permuted_samples.size(3),
                                                                permuted_samples.size(4))).cpu().detach().numpy()
        return flattened_samples


    def updateFrame(self, preds, labels, samples, proceed_stride):

        window_step = self.config.prediction_window_step//self.config.subSample
        right_preds_size = self.config.clip_size// self.config.subSample - window_step *2
        middle_preds = preds[:,  window_step : window_step * 2].cpu().detach().numpy()
        middle_labels = labels[:, window_step : window_step * 2].cpu().detach().numpy()
        middle_frames = samples[:, :, window_step : window_step * 2, :, :]
        flattened_middle_frames = self.flattenFrames(middle_frames)

        vid_start_f = self.curr_test_video_predicted_frame
        vid_end_f = min(self.curr_test_video_predicted_frame + window_step * proceed_stride,
                        self.curr_test_video_nframe)
        self.curr_prediction[vid_start_f:vid_end_f] = middle_preds.flatten()
        self.curr_label[vid_start_f:vid_end_f] = middle_labels.flatten()
        if isinstance(self.curr_video_frames, type(None)):
            self.curr_video_frames = flattened_middle_frames
        else:
            self.curr_video_frames = np.concatenate((self.curr_video_frames, flattened_middle_frames), axis=0)

        if self.curr_test_video_predicted_frame == window_step:
            self.curr_prediction[0:window_step] = preds[0, :window_step].cpu().detach().numpy()
            self.curr_label[0:window_step] = labels[0, :window_step].cpu().detach().numpy()
            flattened_front_frames = self.flattenFrames(samples[:, :, 0:window_step, :, :])
            self.curr_video_frames = np.concatenate((flattened_front_frames , self.curr_video_frames), axis=0)


        if (vid_end_f + right_preds_size >= self.curr_test_video_nframe):
            last_clip_start_f = self.curr_test_video_predicted_frame + window_step * (proceed_stride - 1)
            last_clip_end_f = self.curr_test_video_nframe
            last_clip_size = last_clip_end_f - last_clip_start_f
            self.curr_prediction[last_clip_start_f:last_clip_end_f] = preds[-1, -last_clip_size:].cpu().detach().numpy()
            self.curr_prediction[-1] = self.curr_prediction[-2]
            self.curr_label[last_clip_start_f:last_clip_end_f] = labels[-1, -last_clip_size:].cpu().detach().numpy()
            flattened_last_frames = self.flattenFrames(torch.unsqueeze(samples[samples.size(0)-1,:,last_clip_start_f:last_clip_end_f,:,:],
                                                             0))
            self.curr_video_frames = np.concatenate((self.curr_video_frames[:last_clip_start_f], flattened_last_frames), axis=0)
            self.curr_test_video_predicted_frame = last_clip_end_f
            video_name = os.path.basename(self.curr_test_video_path)[:-4] + "_scroll.mp4"

            self.saveResult(self.curr_prediction, self.curr_fps, self.curr_test_video_path, self.curr_label)
            self.savePrediction(self.curr_prediction, self.curr_label, self.curr_test_video_path)

            self.curr_test_video_predicted_frame = last_clip_end_f
        else:
            self.curr_test_video_predicted_frame = vid_end_f

            
    def updatePrediction(self, preds, labels, samples, video_paths, fpss, resampled_video_nframe):
        batch_size = len(video_paths)
        start_clip = 0

        while(start_clip < batch_size):
            if self.curr_test_video_nframe == 0:
                self.curr_test_video_nframe = resampled_video_nframe[start_clip]//self.config.subSample
                self.curr_prediction = np.zeros(self.curr_test_video_nframe+1)
                self.curr_label = np.zeros(self.curr_test_video_nframe+1)
                self.curr_video_frames = None
                self.curr_test_video_path = video_paths[start_clip]
                self.curr_fps = fpss[start_clip].item()
            end_clip = start_clip + np.sum(np.where(np.array(video_paths) == self.curr_test_video_path, 1, 0))

            curr_vid_n = end_clip - start_clip
            if curr_vid_n > 0:
                self.updateFrame(preds[start_clip:end_clip, :], labels[start_clip:end_clip, :],
                                 samples[start_clip:end_clip,:,:,:], curr_vid_n)
            if self.curr_test_video_predicted_frame == self.curr_test_video_nframe:
                self.clearPrediction()
            start_clip = end_clip   

    def run(self, optimization=False):
        device = self.model.getDevice()
        phase = "val"
        print(f"dataloader keys {self.dataloaders.keys()}")
        for epoch in range(self.config.nepoch):
            print('Epoch {}/{}'.format(epoch, self.config.nepoch - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            self.model.eval()  # Set model to evaluate mode
            batch = 0
            # Iterate over data.
            for inputs, labels, video_paths, fpss, resampled_video_nframe in self.dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    unshaped_output = self.model.run(inputs)
                    outputs = torch.reshape(unshaped_output, (inputs.size(0), -1))
                    if (self.model.model_type == "discriminant"):
                        sig = nn.Sigmoid()(outputs)
                    elif (self.model.model_type == "regressive"):
                        sig = outputs

                batch = batch + 1
                self.updatePrediction(sig, labels, inputs, video_paths, fpss, resampled_video_nframe)

        return


class TestMultiLabelModelRunner(ModelRunner):
    def __init__(self, visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter):
        super().__init__(visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter)
        self.resultPaths = ResultPathSystem(self.config.data_dir, resultDir=self.config.result_dir)
        self.clearPrediction()

    def clearPrediction(self):

        self.curr_prediction = []
        self.curr_label = []
        self.curr_video_frames = None
        self.curr_test_video_path = ""
        self.curr_test_video_nframe = 0
        self.curr_test_video_predicted_frame = self.config.prediction_window_step // self.config.subSample
        self.curr_fps = -1

    def clearBatchCount(self, dataloader):
        self.epoch_batch_num = len(dataloader)
        self.epoch_batch_count = 0

    def increaseBatchCount(self):
        self.epoch_batch_count += 1

    def savePrediction(self, prediction, label, video_path):
        video_name = os.path.basename(video_path)
        id = video_name[:-4]
        filename = f"{id}"
        prediction_path = os.path.join(self.reporter.filePaths.prediction_dir, filename + '_pred.csv')
        label_path = os.path.join(self.reporter.filePaths.prediction_dir, filename + '_label.csv')

        CSVWriter.writeMultiLabel(prediction_path, prediction)
        CSVWriter.writeMultiLabel(label_path, label)

    def saveResult(self, prediction, fps, video_path, test_label):
        resultWriter = self.resultPaths.getVideoResultWriterFromPath(video_path)
        print(f"save to video path {video_path}")
        print(f"fps {fps}")
        print(f"resampled fps {fps / self.config.subSampleRate}")
        print(f"prediction size {prediction.shape}")
        print(f"label size {test_label.shape}")
        print(f"config {resultWriter.config}")
        resultWriter.setResult(video_path, fps / self.config.subSampleRate, prediction, test_label)
        resultWriter.saveResult()

    def flattenFrames(self, samples):
        permuted_samples = torch.permute(samples, (0, 2, 3, 4, 1))
        flattened_samples = torch.reshape(permuted_samples, (-1, permuted_samples.size(2),
                                                             permuted_samples.size(3),
                                                             permuted_samples.size(4))).cpu().detach().numpy()
        return flattened_samples

    def updateFrame(self, preds, labels, samples, proceed_stride):

        window_step = self.config.prediction_window_step // self.config.subSample
        right_preds_size = self.config.clip_size // self.config.subSample - window_step * 2
        middle_preds = preds[:, :, window_step: window_step * 2].cpu().detach().numpy()
        middle_labels = labels[:, :, window_step: window_step * 2].cpu().detach().numpy()
        middle_frames = samples[:, :, window_step: window_step * 2, :, :]
        flattened_middle_frames = self.flattenFrames(middle_frames)

        vid_start_f = self.curr_test_video_predicted_frame
        vid_end_f = min(self.curr_test_video_predicted_frame + window_step * proceed_stride,
                        self.curr_test_video_nframe)
        self.curr_prediction[:, vid_start_f:vid_end_f] = middle_preds.reshape((middle_preds.shape[0], -1))
        self.curr_label[:, vid_start_f:vid_end_f] = middle_labels.reshape((middle_labels.shape[0], -1))
        if isinstance(self.curr_video_frames, type(None)):
            self.curr_video_frames = flattened_middle_frames
        else:
            self.curr_video_frames = np.concatenate((self.curr_video_frames, flattened_middle_frames), axis=0)

        if self.curr_test_video_predicted_frame == window_step:
            self.curr_prediction[:, 0:window_step] = preds[:, 0, :window_step].cpu().detach().numpy()
            self.curr_label[:, 0:window_step] = labels[:, 0, :window_step].cpu().detach().numpy()
            flattened_front_frames = self.flattenFrames(samples[:, :, 0:window_step, :, :])
            self.curr_video_frames = np.concatenate((flattened_front_frames, self.curr_video_frames), axis=0)

        if (vid_end_f + right_preds_size >= self.curr_test_video_nframe):
            last_clip_start_f = self.curr_test_video_predicted_frame + window_step * (proceed_stride - 1)
            last_clip_end_f = self.curr_test_video_nframe
            last_clip_size = last_clip_end_f - last_clip_start_f
            self.curr_prediction[:, last_clip_start_f:last_clip_end_f] = preds[:, -1, -last_clip_size:].cpu().detach().numpy()
            self.curr_label[:, last_clip_start_f:last_clip_end_f] = labels[:, -1, -last_clip_size:].cpu().detach().numpy()
            flattened_last_frames = self.flattenFrames(
                torch.unsqueeze(samples[samples.size(0) - 1, :, last_clip_start_f:last_clip_end_f, :, :],
                                0))
            self.curr_video_frames = np.concatenate((self.curr_video_frames[:last_clip_start_f], flattened_last_frames),
                                                    axis=0)
            self.curr_test_video_predicted_frame = last_clip_end_f

            self.saveResult(self.curr_prediction, self.curr_fps, self.curr_test_video_path, self.curr_label)
            self.savePrediction(self.curr_prediction, self.curr_label, self.curr_test_video_path)

            self.curr_test_video_predicted_frame = last_clip_end_f
        else:
            self.curr_test_video_predicted_frame = vid_end_f

    def updatePrediction(self, preds, labels, samples, video_paths, fpss, resampled_video_nframe):
        preds = torch.permute(preds, (1,0,2))
        labels = torch.permute(labels, (1,0,2))

        batch_size = len(video_paths)
        num_label = len(preds)
        start_clip = 0

        # if self.curr_test_video_nframe == 0:
        #     self.curr_test_video_nframe = resampled_video_nframe[0]
        #     self.curr_test_video_path = video_paths[0]
        # end_clip = np.sum(np.where(np.array(video_paths) == self.curr_test_video_path, 1, 0))

        while (start_clip < batch_size):
            if self.curr_test_video_nframe == 0:
                self.curr_test_video_nframe = resampled_video_nframe[start_clip] // self.config.subSample
                self.curr_prediction = np.zeros((num_label, self.curr_test_video_nframe))
                self.curr_label = np.zeros((num_label, self.curr_test_video_nframe))
                # self.curr_prediction = np.zeros((num_label, self.curr_test_video_nframe + 1))
                # self.curr_label = np.zeros((num_label, self.curr_test_video_nframe + 1))
                self.curr_video_frames = None
                self.curr_test_video_path = video_paths[start_clip]
                self.curr_fps = fpss[start_clip].item()
            end_clip = start_clip + np.sum(np.where(np.array(video_paths) == self.curr_test_video_path, 1, 0))

            curr_vid_n = end_clip - start_clip
            if curr_vid_n > 0:
                self.updateFrame(preds[:, start_clip:end_clip, :], labels[:, start_clip:end_clip, :],
                                 samples[start_clip:end_clip, :, :, :], curr_vid_n)
            if self.curr_test_video_predicted_frame == self.curr_test_video_nframe:
                self.clearPrediction()
            start_clip = end_clip

    def run(self, optimization=False):
        device = self.model.getDevice()
        phase = "val"
        print(f"dataloader keys {self.dataloaders.keys()}")
        for epoch in range(self.config.nepoch):
            print('Epoch {}/{}'.format(epoch, self.config.nepoch - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            self.model.eval()  # Set model to evaluate mode
            batch = 0
            # Iterate over data.
            for inputs, labels, video_paths, fpss, resampled_video_nframe in self.dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    unshaped_output = self.model.run(inputs)
                    # outputs = torch.reshape(unshaped_output, (inputs.size(0), -1))
                    outputs = torch.reshape(unshaped_output, (unshaped_output.size(0), unshaped_output.size(1), -1))
                    if (self.model.model_type == "discriminant"):
                        sig = nn.Sigmoid()(outputs)
                    elif (self.model.model_type == "regressive"):
                        sig = outputs

                batch = batch + 1
                self.updatePrediction(sig, labels, inputs, video_paths, fpss, resampled_video_nframe)

        return

class TestMultiLabelModelAccuracyRunner(ModelRunner):
    def __init__(self, visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter):
        super().__init__(visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter)
        self.resultPaths = ResultPathSystem(self.config.data_dir, resultDir=self.config.result_dir)
        self.num_labels = visBeatModel.num_labels
        self.clearEpochLoss()

    def clearEpochLoss(self):
        self.epoch_RSE_loss = torch.zeros(self.num_labels)
        self.epoch_ARE_loss = torch.zeros(self.num_labels)
        self.epoch_clip_num = 0

    def addEpochLoss(self, rse, are, clip_num):
        self.epoch_RSE_loss += rse
        self.epoch_ARE_loss += are
        self.epoch_clip_num += clip_num


    def computeLossAndAcc(self):
        return self.epoch_RSE_loss / self.epoch_clip_num, self.epoch_ARE_loss.double()/self.epoch_clip_num

    def clearBatchCount(self, dataloader):
        self.epoch_batch_num = len(dataloader)
        self.epoch_batch_count = 0

    def increaseBatchCount(self):
        self.epoch_batch_count += 1

    def computeBatchPercentage(self):
        return float(self.epoch_batch_count) / self.epoch_batch_num


    def RSE(self, criterion, output, label):
        mt = torch.ones_like(output) *  torch.mean(output) 
        return criterion(output, label) / criterion(mt, label)

    def RSEperE(self, output, label):
        m = torch.mean(output, (0, 2)).unsqueeze(0).unsqueeze(2).repeat(output.size(0), 1, output.size(2))
        return torch.div(torch.square(output - label), torch.square(output - m))

    def accuracy(self, preds, labels):
        torch.sum(self.RSEperE(preds, labels) < 0.1, (0, 2)) / float(labels.size(2))

    def run(self, optimization=False):
        device = self.model.getDevice()
        phase = "val"
        for epoch in range(self.config.nepoch):
            print('Epoch {}/{}'.format(epoch, self.config.nepoch - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            self.model.eval()  # Set model to evaluate mode
            batch = 0
            # Iterate over data.
            for inputs, labels, video_paths, fpss, resampled_video_nframe in self.dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    unshaped_output = self.model.run(inputs)
                    outputs = torch.reshape(unshaped_output, (unshaped_output.size(0), unshaped_output.size(1), -1))
                    batch_size = inputs.size(0)
                    RSElosses = torch.zeros(outputs.size(1))
                    ARElosses = torch.zeros(outputs.size(1))
                    for i in range(outputs.size(1)):
                        RSElosses[i] = self.RSE(self.criterion, outputs[:, i, :], labels[:, i, :]) * batch_size
                        ARElosses[i] = mean_absolute_percentage_error(outputs[:, i, :], labels[:, i, :]) * batch_size

                self.addEpochLoss(RSElosses, ARElosses, batch_size)

                # Compute loss and acc for one epcoh
                epoch_RSE, epoch_ARE = self.computeLossAndAcc()
                print(f"RSE: {epoch_RSE} ARE: {epoch_ARE}")

        return
    
class ModelLossRunner(ModelRunner):
    def __init__(self, visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter):
        super().__init__(visBeatModel, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter)

    def clearBatchCount(self, dataloader):
        self.epoch_batch_num = len(dataloader)
        self.epoch_batch_count = 0

    def increaseBatchCount(self):
        self.epoch_batch_count += 1

    def clearEpochLoss(self):
        self.epoch_loss = 0.0
        self.epoch_norm_loss = 0.0
        self.epoch_correct_preds = 0
        self.epoch_clip_num = 0
        
    def computeLossAndAcc(self):
        return self.epoch_loss / self.epoch_clip_num, self.epoch_norm_loss / float(self.epoch_clip_num), self.epoch_correct_preds /float(self.epoch_clip_num)

    def addEpochLoss(self, epoch_loss, epoch_norm_loss, epoch_correct_preds, epoch_clip_num):
        self.epoch_loss += epoch_loss
        self.epoch_norm_loss += epoch_norm_loss
        self.epoch_correct_preds += epoch_correct_preds
        self.epoch_clip_num += epoch_clip_num
        print(f"loss {self.epoch_loss / self.epoch_clip_num }")
        print(f"norm loss {self.epoch_norm_loss / self.epoch_clip_num}")
        print(f"acc {float(self.epoch_correct_preds) / self.epoch_clip_num}")
        print(f"clip num {self.epoch_clip_num}")

    def saveLoss(self):
        loss, norm_loss, acc = self.computeLossAndAcc()
        loss_dict = dict()
        loss_dict["loss"] = loss
        loss_dict["norm loss"] = norm_loss
        loss_dict["acc"] = acc
        
        jsonReadWriter = JsonReadWriter()
        jsonReadWriter.writeToFile(loss_dict, os.path.join(self.reporter.filePaths.loss_dir, "loss.json"))


    def normalized(self, arr):
        mean = torch.mean(arr)
        std = torch.std(arr)
        zero_mean_arr = arr-mean
        if std == 0:
            std = std + 1e-4
        scaled_zero_mean_arr = zero_mean_arr/(3 * std)
        clamped_scaled_zero_mean_arr = torch.clamp(scaled_zero_mean_arr, -3.0, 3.0)
        return clamped_scaled_zero_mean_arr

    def run(self, optimization=False):
        self.clearEpochLoss()
        device = self.model.getDevice()
        phase = "val"
        print(f"dataloader keys {self.dataloaders.keys()}")
        for epoch in range(self.config.nepoch):
            print('Epoch {}/{}'.format(epoch, self.config.nepoch - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            self.model.eval()  # Set model to evaluate mode
            batch = 0
            # Iterate over data.
            for inputs, labels, video_paths, fpss, resampled_video_nframe in self.dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    unshaped_output = self.model.run(inputs)
                    outputs = torch.reshape(unshaped_output, (inputs.size(0), -1))
                    outputs =  torch.clamp(outputs, min=0.0)
                    loss = self.criterion(outputs, labels)
                    if (self.model.model_type == "discriminant"):
                        sig = nn.Sigmoid()(outputs)
                    elif (self.model.model_type == "regressive"):
                        sig = torch.clamp(outputs, min=0.0)

                norm_sig = self.normalized(sig)
                norm_label = self.normalized(labels)

                sig = norm_sig
                labels = norm_label
                norm_loss = self.criterion(sig, labels)
                curr_epoch_loss = loss.item() * inputs.size(0)
                curr_epoch_norm_loss = norm_loss.item() * inputs.size(0)
                curr_epoch_corrects = torch.sum(torch.square(sig - labels) < 0.1, (0, 1)).item() / float(
                    labels.size(1))
                
                self.addEpochLoss(curr_epoch_loss, curr_epoch_norm_loss, curr_epoch_corrects, inputs.size(0))
                batch = batch + 1
                self.saveLoss()

        self.saveLoss()
        return


class TestDataloaderRunner(object):
    def __init__(self,  config, dataloaders, reporter):
        self.config = config
        self.reporter = reporter
        self.dataloaders = dataloaders

    def run(self):
        phases = ["train", "val"]
        print(f"dataloader keys {self.dataloaders.keys()}")

        batch = 0
        # Iterate over data.
        for phase in phases:
            for inputs, labels, video_paths in self.dataloaders[phase]:
                print(f"batch {batch} input shape {inputs.size()} label shape {labels.size()}")
                batch = batch + 1

        return
