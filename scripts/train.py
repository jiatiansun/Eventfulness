# %%
# %config Completer.use_jedi = False


# %%
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from import_eventfulness import importParent
importParent()
# print(sys.path)

import torchvision_custom
from deepVisualBeatUtil import TaskConfigurationParser, JsonReadWriter
from deepVisualBeatUtil import Debug, LossAccuracyReport
from deepVisualBeatUtil import VisBeatDetectionModel
from deepVisualBeatUtil import LossInitializer
from deepVisualBeatUtil import VideoDataLoader, TrainingVideoDataLoader, TestingVideoDataLoader
from deepVisualBeatUtil import SchedulerInitializer
from deepVisualBeatUtil import LossAccuracyReport
from deepVisualBeatUtil import TrainModelRunner

taskParser = TaskConfigurationParser()
config = taskParser.parse()


# %%
"""
## Load Data
"""

# %%
trainingDataLoaderCreater = TrainingVideoDataLoader(config)
dataloaders = trainingDataLoaderCreater.initializeDataloaders()
num_labels = trainingDataLoaderCreater.getNumLabel() 

# %%
"""
## Initialize Model and optimizer
"""

# %%
reporter = LossAccuracyReport(config)

print(f"# of planes for {config.inplanes} the initial layer")
visbeat = VisBeatDetectionModel(config.model_config, config.model_type, config.ngpu, 
                                config.layer_num, config.feature_extract, use_pretrained=config.use_pretrained,
                                inplanes = config.inplanes,
                                num_acc_label=config.num_accS_dir, 
                                num_vel_label=config.num_velS_dir, 
                                num_blur_label=config.num_blurrs,
                                num_labels = num_labels)

if config.load_model:
    print(f"load checkpoint from {config.load_model_dir} epoch {config.load_epoch}")
    visbeat, optimizer_name, optimizer_dict, _ = VisBeatDetectionModel.fromSavedModel(config.load_model_dir, config.load_epoch, config.ngpu)
else:
    print(f"don't load checkpoint")
    
visbeat.wrapModelForParallelComputing()
    
optimizer = optim.Adam(visbeat.get_parameters_to_update(), lr=config.lr)
    
    
if config.load_model:
    try:
        assert(type(optimizer).__name__ == optimizer_name)
    except:
        print(f"WARNING: misalignment between the optimizer you used for in the checkpoint ({optimizer_name})"
              f"and the optimzer you currently is using ({type(optimizer).__name__})")
    
    optimizer.load_state_dict(optimizer_dict)
    


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

"""

# %%


# %%
"""
# Train Network
"""

# %%

debugger = None



trainModelRunner = TrainModelRunner(visbeat, config, debugger, dataloaders, optimizer, scheduler, criterion, reporter)
trainModelRunner.run()


# %%
