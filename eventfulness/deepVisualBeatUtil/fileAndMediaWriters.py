import numpy as np
import json
import os
import psutil
import csv
import torch
import librosa

class JsonReadWriter(object):
    @staticmethod
    def writeToFile(content, filepath):
        with open(filepath, 'w') as outfile:
            json.dump(content, outfile)


    @staticmethod
    def readFromFile(filepath):
        with open(filepath) as outfile:
            data = json.load(outfile)
            return data
        return None

class CSVWriter(object):

    @staticmethod
    def readFloatLabelPerRow(filepath):
        label = []
        with open(filepath, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                label.extend(row)

        label = np.array([float(n) for n in label])

        return label

    @staticmethod
    def read2DFloatLabelPerRow(filepath):
        label = []
        with open(filepath, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                entry = [float(n) for n in row]
                label.append(entry)

        label = np.array(label)

        return label

    @staticmethod
    def readIntLabelPerRow(filepath):
        label = []
        with open(filepath, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                label.extend(row)

        label = np.array([int(n) for n in label])
        return label


    @staticmethod
    def writeLabel(filename, labels):
        print("write label to {}".format(filename))
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            ls = list(labels.tolist())

            number = [str(d) for d in ls]
            csvwriter.writerow(number)

    @staticmethod
    def writeMultiLabel(filename, labels):
        assert(labels.ndim == 2)
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            for i in range(len(labels)):
                ls = list(labels[i].tolist())

                number = [str(d) for d in ls]
                csvwriter.writerow(number)

    @staticmethod
    def readLabelAsNP(filepath):
        label = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                label = np.array([float(e) for e in row])
        return label

    @staticmethod
    def readMultiLabelAsNP(filepath):
        labels = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                labels.append([float(e) for e in row])
        return np.array(labels)

    @staticmethod
    def writeList(filename, l):
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            paths = [str(s) for s in l]
            print("write list to {} with l {}".format(filename, paths))
            csvwriter.writerow(paths)


class OSUtil(object):
    @staticmethod
    def safe_mkdir(path):
        try:
            os.mkdir(path)
        except:
            print('already/ failed to create to {}'.format(path))

    @staticmethod
    def getCPUUsage():
        return psutil.virtual_memory().percent