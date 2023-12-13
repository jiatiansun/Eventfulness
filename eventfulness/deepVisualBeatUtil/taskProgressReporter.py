import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import auc, precision_recall_curve, average_precision_score

from datetime import datetime
import matplotlib.pyplot as plt
import os

from .fileAndMediaWriters import OSUtil, JsonReadWriter, CSVWriter

class TensorBoardWriter(object):
    def __init__(self, config, precision_and_recall=False):
        self.name = '_{}_layer{}_lr{}_ep{}_bs{}_cs{}_vcs{}'.format(config.model_config,
                                                                   config.layer_num,
                                                                   config.lr,
                                                                   config.nepoch,
                                                                   config.batch_size,
                                                                   config.clip_size,
                                                                   config.val_clips_per_video)
        self.precision_and_recall = precision_and_recall
        self.writer = SummaryWriter(comment=self.name)


    def updateTensorBoard(self, phase, iteration, loss, acc, auc=0, precision=0, recall=0):
        self.writer.add_scalar("Loss/{}".format(phase), loss, iteration)
        self.writer.add_scalar("acc/{}".format(phase), acc, iteration)
        self.writer.add_scalar("auc/{}".format(phase), auc, iteration)
        if self.precision_and_recall:
            self.writer.add_scalar("precision/{}".format(phase), precision, iteration)
            self.writer.add_scalar("recall/{}".format(phase), recall, iteration)

    def flushTensorBoard(self):
        self.writer.flush()

    def closeTenosrBoard(self):
        self.writer.close()




class FilePathDict(object):
    def __init__(self, config):
        # Initialization for directories that store intermediate Results
        self.time = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        self.dir_prefix = './lossAccuracyReport'
        self.dir = os.path.join(self.dir_prefix, self.time)
        self.error_dir = os.path.join(self.dir, 'error')
        self.correct_dir = os.path.join(self.dir, 'correct')
        self.prediction_dir = os.path.join(self.dir, 'prediction')
        self.loss_dir = os.path.join(self.dir, 'loss')

        self.data_results_dir = os.path.join(config.data_dir, "results")

        # Stores videos at which the video fails.
        self.error_vid_paths_file = os.path.join(self.dir, 'error_paths.txt')
        self.correct_vid_paths_file = os.path.join(self.dir, 'correct_paths.txt')

        OSUtil.safe_mkdir(self.dir_prefix)
        OSUtil.safe_mkdir(self.dir)
        OSUtil.safe_mkdir(self.error_dir)
        OSUtil.safe_mkdir(self.correct_dir)
        OSUtil.safe_mkdir(self.prediction_dir)
        OSUtil.safe_mkdir(self.loss_dir)
        OSUtil.safe_mkdir(self.data_results_dir)

class LossAccuracyReport:
    def __init__(self, config):

        self.config = config
        self.filePaths = FilePathDict(config)
        self.tensorBoardWriter = TensorBoardWriter(config)


        # Write Configuration to a file containing
        JsonReadWriter.writeToFile(config.toDict(), os.path.join(self.filePaths.dir, 'config.txt'))

        # Initialization for Tensorboard Writing
        self.loss = {}
        self.accuracy = {}
        self.iteration = {}
        self.auc_score = {}
        self.precision = {}
        self.recall = {}

        for phase in ['train', 'val']:
            self.loss[phase] = []
            self.accuracy[phase] = []
            self.iteration[phase] = []
            self.auc_score[phase] = []
            self.precision[phase] = []
            self.recall[phase] = []

    def writePredictionImage(self, epoch, epoch_size,
                             model_type,
                             loss, preds, labels,
                             inputs, outputs, filePaths):
        print(f"write prediction Image {epoch}")

        if (self.config.model_type == "disriminant"):
            wrongPrediction = torch.sum(preds == labels, 1) > 0
        elif (model_type == "regressive"):
            sqrd_diff = torch.square(preds - labels)
            wrongPrediction = torch.sum(sqrd_diff >= 0.1, 1) > 0

        all_idx = torch.arange(preds.size(0))
        error_idx = all_idx[wrongPrediction].tolist()
        error_vids = inputs[wrongPrediction, :, :, :, :]
        error_labs = labels[wrongPrediction, :]
        error_preds = outputs[wrongPrediction, :]


        correctPrediction = torch.logical_not(wrongPrediction)
        correct_idx = all_idx[correctPrediction].tolist()
        correct_vids = inputs[correctPrediction, :, :, :]
        correct_labs = labels[correctPrediction, :]
        correct_preds = outputs[correctPrediction, :]

        for i in range(error_vids.size(0)):
            plt.figure()
            plt.subplot(1, 2, 1)
            filename = "{}/epoch_{}_{}_{}_loss{:.3f}.png".format(self.filePaths.error_dir, epoch, epoch_size, i,
                                                                 loss.item())

            error_lab = torch.squeeze(error_labs[i, :])
            error_pred = torch.squeeze(error_preds[i, :])

            plt.plot(error_lab.cpu().detach().numpy(), 'r', label='label')
            plt.plot(error_pred.cpu().detach().numpy(), 'b', label='prediction')
            plt.xticks(range(1, self.config.clip_size, 12))
            plt.title(filePaths[error_idx[i]])
            plt.legend()

            plt.subplot(1, 2, 2)
            input_indexed = torch.reshape(error_vids[i, :, 0, :, :],
                                          (1, error_vids.size(1), 1, error_vids.size(3), error_vids.size(4)))
            im = torch.squeeze(torch.squeeze(input_indexed, 0), 1).permute([1, 2, 0]).cpu().detach().numpy()
            plt.imshow(im)

            plt.savefig(filename)
            plt.show()
            # Clear the current axes.
            plt.cla()
            # Clear the current figure.
            plt.clf()
            # Closes all the figure windows.
            plt.close('all')

        for i in range(correct_vids.size(0)):
            plt.figure()
            plt.subplot(1, 2, 1)
            filename = "{}/epoch_{}_{}_{}_loss{:.3f}.png".format(self.filePaths.correct_dir, epoch, epoch_size, i,
                                                                 loss.item())

            correct_lab = torch.squeeze(correct_labs[i, :])
            correct_pred = torch.squeeze(correct_preds[i, :])

            plt.plot(correct_lab.cpu().detach().numpy(), 'r', label='label')
            plt.plot(correct_pred.cpu().detach().numpy(), 'b', label='prediction')

            plt.legend()
            plt.xticks(range(1, self.config.clip_size, 12))
            plt.title(filePaths[correct_idx[i]])

            plt.subplot(1, 2, 2)
            input_indexed = torch.reshape(correct_vids[i, :, 0, :, :],
                                          (1, correct_vids.size(1), 1, correct_vids.size(3), correct_vids.size(4)))
            im = torch.squeeze(torch.squeeze(input_indexed, 0), 1).permute([1, 2, 0]).cpu().detach().numpy()
            plt.imshow(im)

            plt.savefig(filename)
            plt.show()
            # Clear the current axes.
            plt.cla()
            # Clear the current figure.
            plt.clf()
            # Closes all the figure windows.
            plt.close('all')

    def updateLossAccuracy(self, phase, loss, output, bin_preds, labels, doPrint=False):

        batch_size = labels.size(0)
        clip_size = labels.size(1)

        iterations = self.iteration[phase]
        iterationNum = len(iterations)
        iterations.append(iterationNum)

        curr_loss = loss.item()

        if self.config.model_type == "discriminant" and not self.config.use_blur:
            curr_acc = torch.sum(bin_preds == labels, (0, 1)) / float(batch_size * clip_size)
            labels_np = labels.cpu().detach().numpy()
            output_np = output.cpu().detach().numpy()
            precision, recall, _ = precision_recall_curve(labels_np.ravel(), output_np.ravel())
            curr_auc = auc(recall, precision)

            batch_precision = torch.sum(torch.logical_and(bin_preds == labels, bin_preds == 1.0), (0, 1)) / float(
                torch.sum(bin_preds == 1.0, (0, 1)) + 1e-07)
            batch_recall = torch.sum(torch.logical_and(bin_preds == labels, labels == 1.0), (0, 1)) / float(
                torch.sum(labels == 1.0, (0, 1)) + 1e-07)
        elif (self.config.model_type == "regressive"):
            sqrd_diff = torch.square(bin_preds - labels)
            curr_acc = torch.sum(sqrd_diff < 0.1, (0, 1)) / float(batch_size * clip_size)
            curr_auc = 0
            batch_precision = 0
            batch_recall = 0
        else:
            curr_acc = 0
            curr_auc = 0
            batch_precision = 0
            batch_recall = 0

        self.loss[phase].append(curr_loss)
        self.accuracy[phase].append(curr_acc)
        self.auc_score[phase].append(curr_auc)
        self.precision[phase].append(batch_precision)
        self.recall[phase].append(batch_recall)

        if doPrint:
            print(10 * "-" + "Batch Statistics Printing" + "-" * 10)
            print('batch iter {} batch loss {} batch acc {}'.format(iterationNum, curr_loss, curr_acc))
            print(10 * "------" * 10)
        self.tensorBoardWriter.updateTensorBoard(phase, iterationNum, curr_loss, curr_acc, curr_auc, batch_precision, batch_recall)

