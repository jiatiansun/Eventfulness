# %%
# %config Completer.use_jedi = False
# %run import_eventfulness.py

# %%
from __future__ import print_function
from __future__ import division
import torch.nn as nn
import torch.optim as optim
import numpy as np

from import_eventfulness import importParent
importParent()
# import sys
# print(sys.path)

from torchvision_custom import *
from deepVisualBeatUtil import TaskConfigurationParser, JsonReadWriter
from deepVisualBeatUtil import Debug, LossAccuracyReport
from deepVisualBeatUtil import VisBeatDetectionModel
from deepVisualBeatUtil import LossInitializer
from deepVisualBeatUtil import VideoDataLoader, TrainingVideoDataLoader, TestingVideoDataLoader
from deepVisualBeatUtil import SchedulerInitializer
from deepVisualBeatUtil import LossAccuracyReport
from deepVisualBeatUtil import TestMultiLabelModelRunner

taskParser = TaskConfigurationParser()
config = taskParser.parse()


# %%
"""
## load data
"""

# %%
TestDataLoaderCreater = TestingVideoDataLoader(config)
dataloaders = TestDataLoaderCreater.initializeDataloaders()
# num_labels = TestDataLoaderCreater.getNumLabel() 

# %%
"""
## Initialize Model and optimizer
"""

# %%
num_labels = None
if config.num_labels > 0:
    num_labels = config.num_labels

print(num_labels)
visbeat = VisBeatDetectionModel(config.model_config, config.model_type, config.ngpu, 
                                config.layer_num, config.feature_extract, use_pretrained=config.use_pretrained,
                                inplanes = config.inplanes,
                                num_acc_label=config.num_accS_dir, 
                                num_vel_label=config.num_velS_dir, 
                                num_blur_label=config.num_blurrs,
                                num_labels = num_labels)
                

if config.load_model:
    visbeat, optimizer_name, optimizer_dict, _ = VisBeatDetectionModel.fromSavedModel(config.load_model_dir, config.load_epoch, config.ngpu)      

visbeat.wrapModelForParallelComputing()
    
optimizer = optim.Adam(visbeat.get_parameters_to_update(), lr=config.lr)


if config.load_checkpoint:
    visbeat.loadCheckpoint(config.load_checkpoint_dir, None, optimizer)


# %%
"""
## Initialize Loss Function
"""

# %%
scheduler = SchedulerInitializer.initialize_scheduler(optimizer, scheduler_name=config.scheduler_name ,**config.scheduler_kwargs)
lossInitializer = LossInitializer()
criterion = lossInitializer.initializeLoss(config.model_type)

# %%
"""
## Load Data
"""

# %%
"""
## Train Network
"""

# %%

debugger = None
reporter = LossAccuracyReport(config)


testModelRunner = TestMultiLabelModelRunner(visbeat, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter)  
testModelRunner.run()


# %%
