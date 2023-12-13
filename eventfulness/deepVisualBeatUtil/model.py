import os
import time
import copy
import torch
import numpy as np

from torchvision_custom import models

class VisBeatDetectionModel(object):
    @staticmethod
    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def encode_model_checkpoint_str(model_dir, epoch, checkpoint_name):
        epoch_str = ""
        if epoch != None:
            epoch_str = f"_{epoch}"
        return os.path.join(model_dir, os.path.join('prediction', f'{checkpoint_name}{epoch_str}.pt'))

    def __init__(self, model_config, model_type, ngpu, layer_num, feature_extract, use_pretrained=True,
                 inplanes = 64,
                 num_acc_label=0, num_vel_label=0, num_blur_label=0, num_labels=None):
        self.model_config = model_config
        self.model_type = model_type
        self.inplanes = inplanes
        self.ngpu = ngpu
        self.layer_num = layer_num
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        if num_labels is not None:
            self.num_labels = num_labels
        else:
            self.num_labels = 1 + num_acc_label + num_vel_label + num_blur_label
        self.model = None
        self.bareboneModel = None

        torch.cuda.empty_cache()
        print("check if torch is on gpu {}".format(torch.cuda.is_available()))
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.ngpu > 0 else "cpu")

        self.best_model_loss = float('inf')
        self.best_model_weights = {}
        self.initialize_model()


    @classmethod
    def fromSavedModel(cls, model_dir, load_epoch, ngpu):
        model_path = VisBeatDetectionModel.encode_model_checkpoint_str(model_dir, load_epoch, 'model')
        print(f"Start loading in model from {model_path}")
        checkpoint = torch.load(model_path, map_location = "cuda:0")
        model_config = checkpoint['model_config']
        model_type = checkpoint['model_type']
        layer_num = checkpoint['layer_num']
        feature_extract = checkpoint['feature_extract']
        use_pretrained = checkpoint['use_pretrained']
        num_labels = 1
        inplanes = 64
        if 'num_labels' in checkpoint:
            num_labels = checkpoint['num_labels']
        if 'inplanes' in checkpoint:
            inplanes = checkpoint['inplanes']
        visbeatModel = cls(model_config, model_type, ngpu, layer_num, feature_extract,
                           use_pretrained=use_pretrained, inplanes=inplanes, num_labels=num_labels)
        assert (visbeatModel.model != None)
        visbeatModel.model.load_state_dict(checkpoint['model'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        visbeatModel.best_model_loss = loss

        return visbeatModel, checkpoint['optimizer_name'], checkpoint['optimizer'], epoch

    @classmethod
    def fromSavedModelToCPU(cls, model_path):
        print(f"Start loading in model from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_config = checkpoint['model_config']
        model_type = checkpoint['model_type']
        layer_num = checkpoint['layer_num']
        feature_extract = checkpoint['feature_extract']
        use_pretrained = checkpoint['use_pretrained']
        num_labels = 1
        if 'num_labels' in checkpoint:
            num_labels = checkpoint['num_labels']
        if 'inplanes' in checkpoint:
            inplanes = checkpoint['inplanes']

        visbeatModel = cls(model_config, model_type, 0, layer_num, feature_extract,
                           use_pretrained=use_pretrained, inplanes=inplanes, num_labels=num_labels)
        assert (visbeatModel.model != None)
        visbeatModel.model.load_state_dict(checkpoint['model'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        visbeatModel.best_model_loss = loss

        return visbeatModel, model_config

    def getModel(self):
        return self.model

    def getDevice(self):
        return self.device

    def initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None

        if self.layer_num == 9 and self.model_config == "2dp1w0t":
            """ Resnet with no temporal layer, 9
            """
            model_ft = models.video.r2plus1dw0tc_9(pretrained=self.use_pretrained,
                                                   num_labels=self.num_labels,
                                                   inplanes=self.inplanes)
                                                   # power=self.power)
        elif self.layer_num == 18 and self.model_config == "2dp1w0t":
            """ Resnet with no temporal layer, 18
            """
            model_ft = models.video.r2plus1dw0tc_18(pretrained=self.use_pretrained,
                                                    num_labels=self.num_labels,
                                                    inplanes=self.inplanes)
                                                    # power=self.power)
        elif self.layer_num == 9 and self.model_config == "2dp1w1t":
            """ Resnet  with one temporal layer, 9
            """
            model_ft = models.video.r2plus1dw1tc_9(pretrained=self.use_pretrained,
                                                   num_labels=self.num_labels,
                                                   inplanes=self.inplanes)
        elif self.layer_num == 18 and self.model_config == "2dp1w1t":
            """ Resnet  with one temporal layer, 18
            """
            model_ft = models.video.r2plus1dw1tc_18(pretrained=self.use_pretrained,
                                                    num_labels=self.num_labels,
                                                    inplanes=self.inplanes)
        elif self.layer_num == 18 and self.model_config == "2dp1w2t":
            """ Resnet  with one temporal layer, 18
            """
            model_ft = models.video.r2plus1dw2tc_18(pretrained=self.use_pretrained, 
                                                    num_labels=self.num_labels,
                                                    inplanes=self.inplanes)
        elif self.layer_num == 18 and self.model_config == "2dp1w3t":
            """ Resnet  with one temporal layer, 18
            """
            model_ft = models.video.r2plus1dw3tc_18(pretrained=self.use_pretrained, 
                                                    um_labels=self.num_labels,
                                                    inplanes=self.inplanes)
        else:
            print("Invalid model name, exiting...")
            exit()

        self.model = model_ft
        self.bareboneModel = model_ft
        self.set_parameter_grad_for_feature_extract()
        self.best_model_weights = copy.deepcopy(self.model.state_dict())

    def wrapModelForParallelComputing(self):
        ids = list(range(self.ngpu))
        if len(ids) > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=[x for x in range(len(ids))])

        self.model.to(self.device)
        
    def updateBestModel(self, new_loss):
        if new_loss <= self.best_model_loss:
            self.best_model_loss = new_loss
            if self.ngpu > 1:
                self.best_model_weights = copy.deepcopy(self.model.module.state_dict())
            else:
                self.best_model_weights = copy.deepcopy(self.model.state_dict())

    def loadBestModel(self):
        self.model.load_state_dict(self.best_model_weights)

    def saveBestModel(self, model_dir, optimizer, name="best", epoch=None):
        save_path = VisBeatDetectionModel.encode_model_checkpoint_str(model_dir, epoch, name)
        torch.save({'model': self.best_model_weights,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': self.best_model_loss,
                    'model_config': self.model_config,
                    'model_type': self.model_type,
                    'layer_num': self.layer_num,
                    'feature_extract': self.feature_extract,
                    'use_pretrained': self.use_pretrained,
                    'num_labels': self.num_labels,
                    'optimizer_name': type(optimizer).__name__
                    },
                   save_path)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def run(self, input):
        return self.model(input)

    def set_parameter_grad_for_feature_extract(self):
        # Iterate through the first 5 layers of the model and only
        # make these layer ignore gradient computation if the feature
        # extract is set to true:
        if self.feature_extract:
            cl = 0
            for child_layer in self.model.children():
                cl += 1
                if cl < 6:
                    VisBeatDetectionModel.set_parameter_requires_grad(child_layer)

    def get_parameters_to_update(self):
        if self.feature_extract:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
            return params_to_update
        else:
            return self.model.parameters()

    def loadCheckpoint(self, model_dir, load_epoch, optimizer):
        print(f"start loading model in {model_dir}")
        load_path = VisBeatDetectionModel.encode_model_checkpoint_str(model_dir, load_epoch,'val')
        print(f"load path : {load_path}")
        checkpoint = torch.load(load_path)
        
        self.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def saveCheckpoint(self, model_dir, save_epoch, optimizer, loss):
        print(f"start saving model of epoch {save_epoch}")
        save_path = VisBeatDetectionModel.encode_model_checkpoint_str(model_dir, save_epoch,'val')

        if self.ngpu > 1:
            model_dict = self.model.module.state_dict()
        else:
            model_dict = self.model.state_dict()
        torch.save({'model': model_dict, # model.main_module ?
                    'optimizer': optimizer.state_dict(),
                    'epoch': save_epoch,
                    'loss': loss},
                    save_path)


    def saveModel(self, model_dir, save_epoch, optimizer, loss):
        save_path = VisBeatDetectionModel.encode_model_checkpoint_str(model_dir, save_epoch,'model')

        if self.ngpu > 1:
            model_dict = self.model.module.state_dict()
        else:
            model_dict = self.model.state_dict()
        print(f"save epoch {save_epoch} model")
        torch.save({'model': model_dict,
                    'optimizer': optimizer.state_dict(),
                    'epoch': save_epoch,
                    'loss': loss,
                    'model_config':self.model_config,
                    'model_type':self.model_type,
                    'layer_num': self.layer_num,
                    'feature_extract':self.feature_extract,
                    'use_pretrained':self.use_pretrained,
                    'num_labels':self.num_labels,
                    'inplanes':self.inplanes,
                    'optimizer_name':type(optimizer).__name__
                    },
                   save_path)