import os
from deepVisualBeatUtil.LabelLoader.accVelLabelLoader import AccVelLabelLoader
from deepVisualBeatUtil.LabelLoader.accVelMagLabelLoader import AccVelMagLabelLoader
from deepVisualBeatUtil.LabelLoader.accVelNoRotLabelLoader import AccVelNoRotLabelLoader
from deepVisualBeatUtil.LabelLoader.accVelScaledLabelLoader import AccVelScaledLabelLoader
from deepVisualBeatUtil.LabelLoader.accVelMmtDeltaLabelLoader import AccVelMmtDeltaLabelLoader
from deepVisualBeatUtil.LabelLoader.rescaledAccVelMmtDeltaLabelLoader import AccVelMmtDeltaScaledLabelLoader
from deepVisualBeatUtil.LabelLoader.accVelShiftScaledLabelLoader import AccVelShiftScaledLabelLoader

from deepVisualBeatUtil.LabelLoader.caccVelShiftScaledLabelLoader import cAccVelShiftScaledLabelLoader
from deepVisualBeatUtil.LabelLoader.caccVelShiftScaledSumLabelLoader import cAccVelShiftScaledSumabelLoader
from deepVisualBeatUtil.LabelLoader.caccVelShift2ScaledLabelLoader import cAccVelShiftLeftScaledLabelLoader
from deepVisualBeatUtil.LabelLoader.caccVelShift3ScaledLabelLoader import cAccVelShiftRoundScaledLabelLoader

from deepVisualBeatUtil.LabelLoader.NoneLoader import NoneLoader
from torchvision_custom import datasets_custom, transforms
from torchvision_custom.datasets_custom.samplers import DistributedSampler, UniformClipSampler, RandomClipSampler

from transforms import ConvertBHWCtoBCHW, ConvertBCHWtoCBHW, CameraShake, CropAndRandomShuffle

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from .fileAndMediaWriters import JsonReadWriter
from .gaussianFilterGenerator import GaussianKernelGenerator


def collate_fn_train(batch):
    batch = [(d[0], d[1], d[2]) for d in batch]
    return default_collate(batch)


def collate_fn_test(batch):
    # , d[5], d[6]
    batch = [(d[0], d[1], d[2], d[4], d[5]) for d in batch]
    return default_collate(batch)


class VideoDataLoader(object):
    def __init__(self, config, use_train=True, use_val=True):
        self.config = config.getDataLoaderConfig()
        self.use_train = use_train
        self.use_val = use_val
        self.dataset_phases = set()
        if use_train:
            self.dataset_phases.add('train')
        
        if use_val:
            self.dataset_phases.add('val')

        if self.config.num_blurrs > 0:
            self.gaussFilterGen = GaussianKernelGenerator(self.config.d, self.config.min_sig, self.config.max_sig,
                                                      self.config.num_blurrs, debug=True)
        else:
            self.gaussFilterGen = None

        self.labelLoader = NoneLoader(self.config.num_accS_dir, self.config.num_velS_dir, self.config.num_blurrs,
                                      num_label = self.config.num_labels)

        if config.label_type == AccVelLabelLoader.getName():
            self.labelLoader = AccVelLabelLoader(self.config.num_accS_dir, self.config.num_velS_dir,self.gaussFilterGen)
        elif config.label_type == AccVelScaledLabelLoader.getName():
            self.labelLoader = AccVelScaledLabelLoader(self.config.num_accS_dir, self.config.num_velS_dir,self.gaussFilterGen)
        elif config.label_type == AccVelMagLabelLoader.getName():
            self.labelLoader = AccVelMagLabelLoader(self.config.num_accS_dir, self.config.num_velS_dir,self.gaussFilterGen)
        elif config.label_type == AccVelShiftScaledLabelLoader.getName():
            self.labelLoader = AccVelShiftScaledLabelLoader(self.config.num_accS_dir, self.config.num_velS_dir,self.gaussFilterGen)
        elif config.label_type == AccVelMmtDeltaLabelLoader.getName():
            self.labelLoader = AccVelMmtDeltaLabelLoader(self.gaussFilterGen)
        elif config.label_type == AccVelMmtDeltaScaledLabelLoader.getName():
            self.labelLoader = AccVelMmtDeltaScaledLabelLoader(self.gaussFilterGen)
        elif config.label_type == AccVelNoRotLabelLoader.getName():
            self.labelLoader = AccVelNoRotLabelLoader(self.gaussFilterGen)
        elif config.label_type == cAccVelShiftScaledLabelLoader.getName():
            print("Load c acc vel shift label loader")
            self.labelLoader = cAccVelShiftScaledLabelLoader(self.gaussFilterGen)
        elif config.label_type == cAccVelShiftScaledSumabelLoader.getName():
            print("Load c acc vel shift projected label loader")
            self.labelLoader = cAccVelShiftScaledSumabelLoader(self.config.num_accS_dir, self.config.num_velS_dir,self.gaussFilterGen)
        elif config.label_type == cAccVelShiftLeftScaledLabelLoader.getName():
            print("Load c acc vel left shift label loader")
            self.labelLoader = cAccVelShiftLeftScaledLabelLoader(self.gaussFilterGen)
        elif config.label_type == cAccVelShiftRoundScaledLabelLoader.getName():
            print("Load c acc vel round shift label loader")
            self.labelLoader =  cAccVelShiftRoundScaledLabelLoader(self.gaussFilterGen)
        
        

        self.dataloader_name = "dataloader_{}_{}_{}_{}_{}_{}".format(os.path.basename(self.config.data_dir),
                                                       self.config.resize,
                                                       self.config.clip_size,
                                                       self.config.num_blurrs,
                                                       self.config.num_velS_dir,
                                                       self.config.num_accS_dir,
                                                       self.config.num_labels)
        self.encode_dataloader_paths_from_name()

    def getNumLabel(self):
        return self.labelLoader.getNumLabel()

    @staticmethod
    def getDefaultTransform(resize):
        ltransforms = [ConvertBHWCtoBCHW(),
                       transforms.ConvertImageDtype(torch.float32),
                       transforms.Resize((resize, resize)),
                       ConvertBCHWtoCBHW()]

        return transforms.Compose(ltransforms)

    def encode_dataloader_paths_from_name(self):
        self.dataloader_path = os.path.join(self.config.data_dir, self.dataloader_name)
        self.dataloader_paths = {}
        if self.use_train:
            self.dataloader_paths['train'] = "{}_train.pth".format(self.dataloader_path)
        if self.use_val:
            self.dataloader_paths['val'] = "{}_val.pth".format(self.dataloader_path)
        self.dataloader_config_path = "{}_config.txt".format(self.dataloader_path)

    def get_clip_step_size(self):
        return 1

    def initializeDataloaders(self, force_restart = True):
        # Trying to read the cache from dataloader
        read_from_cache = False
        dataLoader_curr_config = self.config.toDict()
        if os.path.exists(self.dataloader_config_path) and not force_restart:
            dataloader_saved_config = JsonReadWriter.readFromFile(self.dataloader_config_path)
            try:
                assert(len(dataloader_saved_config) == len(dataloader_saved_config))
                for key in dataloader_saved_config:
                    assert(dataloader_saved_config[key] == dataLoader_curr_config[key])
                read_from_cache = True
            except:
                print(f"warning, there is misalignment between saved dataloader {dataloader_saved_config}\n"
                      f"and the the curr dataloader configuration {dataLoader_curr_config}"
                      f"We would overwrite that cache becasue of the updated parameters")

        if read_from_cache:
            dataloaders_dict = {x: torch.load(self.dataloader_paths[x]) for x in self.dataset_phases}
        else:
            step_size = self.get_clip_step_size()
            train_video_transform = self.getTrainTransforms()
            test_video_transform = self.getTestTransforms()
            transform_dict = {"train":train_video_transform, "val":test_video_transform}
            video_datasets = {x: datasets_custom.Kinetics400(
                os.path.join(self.config.data_dir, x),
                blur=self.config.use_blur,
                frames_per_clip=self.config.clip_size,
                step_between_clips=step_size,
                transform=transform_dict[x],
                frame_rate=self.config.fps if x == "train" else self.config.fps,
                labelType=self.config.label_type,
                subSample=self.config.subSample,
                subSampleData=self.config.subSampleRate,
                extensions=('avi', 'mp4',),
                labelLoader=self.labelLoader)
                for x in self.dataset_phases}

            # Create training and validation dataloaders
            samplers = self.getSamplers(video_datasets)
            collate_fn = self.getCollateFn()
            print(f" type {type(collate_fn)}")
            dataloaders_dict = {x: torch.utils.data.DataLoader(
                video_datasets[x],
                batch_size=self.config.batch_size,
                sampler=samplers[x],
                num_workers=self.config.nworker,
                collate_fn=collate_fn
            ) for x in self.dataset_phases}

            for phase in self.dataset_phases:
                torch.save(dataloaders_dict[phase], self.dataloader_paths[phase])

            JsonReadWriter.writeToFile(self.config.toDict(), self.dataloader_config_path)

        return dataloaders_dict

    def getTrainTransforms(self):
        ltransforms = [ConvertBHWCtoBCHW(),
                       transforms.ConvertImageDtype(torch.float32)]

        if self.config.cameraShake:
            print(f"has cam shake")
            ltransforms.append(CameraShake((self.config.cropShape,self.config.cropShape), self.config.velocityScale, self.config.springStiffness))

        ltransforms.append(transforms.Resize((self.config.resize, self.config.resize)))

        if self.config.random_crop:
            print(f"has random crop")
            ltransforms.append(transforms.RandomResizedCrop((self.config.crop_size,
                                                             self.config.crop_size),scale=(self.config.scale_size,1.0)))
        if self.config.vertical_flip:
            ltransforms.append(transforms.RandomVerticalFlip(p=self.config.vertical_flip_prob))

        if self.config.horizontal_flip:
            ltransforms.append(transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob))

        if self.config.random_rotation:
            ltransforms.append(transforms.RandomRotation(self.config.random_rotation_angle))

        if self.config.color_jitter:
            print(f"has color jitter")
            ltransforms.append(transforms.transforms.ColorJitter(brightness=self.config.brightness,
                                                                 contrast=self.config.contrast,
                                                                 saturation=self.config.saturation,
                                                                 hue=self.config.hue))

        ltransforms.append(ConvertBCHWtoCBHW())
        print("maximum augmentation")
        return transforms.Compose(ltransforms)


    def getTestTransforms(self):
        ltransforms = [ConvertBHWCtoBCHW(),
                       transforms.ConvertImageDtype(torch.float32),
                       transforms.Resize((self.config.resize, self.config.resize)),
                       ConvertBCHWtoCBHW()]

        print("minimum augmentation")
        return transforms.Compose(ltransforms)

    def getSamplers(self, video_datasets):
        samplers_dict = {}
        if self.use_train:
            samplers_dict['train'] = RandomClipSampler(video_datasets['train'].video_clips, self.config.train_clips_per_video)
        if self.use_val:
            samplers_dict['val'] = UniformClipSampler(video_datasets['val'].video_clips, self.config.val_clips_per_video)

        return samplers_dict

    def getCollateFn(self):
        return collate_fn_train


class TrainingVideoDataLoader(VideoDataLoader):
    
    def __init__(self, config, use_train=True, use_val=True):
        super().__init__(config, use_train=use_train, use_val=use_val)
        self.dataloader_name = "{}_train".format(self.dataloader_name)
        self.encode_dataloader_paths_from_name()


class AlternativeTrainingVideoDataLoader(VideoDataLoader):

    def __init__(self, config, use_train=True, use_val=True):
        super().__init__(config, use_train=use_train, use_val=use_val)
        self.dataloader_name = "{}_train".format(self.dataloader_name)
        self.encode_dataloader_paths_from_name()

    def getTrainTransforms(self):
        ltransforms = [ConvertBHWCtoBCHW(),
                       transforms.ConvertImageDtype(torch.float32)]

        ltransforms.append(transforms.Resize((self.config.resize, self.config.resize)))
        print("minimum augmentation")

        ltransforms.append(ConvertBCHWtoCBHW())

        return transforms.Compose(ltransforms)



class TestingVideoDataLoader(VideoDataLoader):

    def __init__(self, config, use_train=False, use_val=True):
        super().__init__(config, use_train=use_train, use_val=use_val)
        self.dataloader_name = "{}_test".format(self.dataloader_name)
        self.encode_dataloader_paths_from_name()

    def get_clip_step_size(self):
        return self.config.prediction_window_step

    def getSamplers(self, video_datasets):
        samplers_dict = {}
        if self.use_train:
            samplers_dict['train'] = torch.utils.data.SequentialSampler(video_datasets['train'])
        if self.use_val:
            samplers_dict['val'] = torch.utils.data.SequentialSampler(video_datasets['val'])

        return samplers_dict

    def getCollateFn(self):

        return collate_fn_test


class PreviewTestVideoDataLoader(VideoDataLoader):

    def __init__(self, config, use_train=False, use_val=True):
        super().__init__(config, use_train=use_train, use_val=use_val)
        self.dataloader_name = "{}_previewTest".format(self.dataloader_name)
        self.encode_dataloader_paths_from_name()

    def getTrainTransforms(self):
        ltransforms = [ConvertBHWCtoBCHW(),
                       transforms.ConvertImageDtype(torch.float32)]

        if self.config.cameraShake:
            ltransforms.append(CameraShake((self.config.cropShape,self.config.cropShape), self.config.velocityScale, self.config.springStiffness))

        ltransforms.append(transforms.Resize((self.config.resize, self.config.resize)))

        if self.config.random_crop:
            ltransforms.append(transforms.RandomResizedCrop((self.config.crop_size,
                                                             self.config.crop_size),scale=(self.config.scale_size,1.0)))
        if self.config.vertical_flip:
            ltransforms.append(transforms.RandomVerticalFlip(p=self.config.vertical_flip_prob))

        if self.config.horizontal_flip:
            ltransforms.append(transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob))

        if self.config.random_rotation:
            ltransforms.append(transforms.RandomRotation(self.config.random_rotation_angle))

        if self.config.color_jitter:
            ltransforms.append(transforms.transforms.ColorJitter(brightness=self.config.brightness,
                                                                 contrast=self.config.contrast,
                                                                 saturation=self.config.saturation,
                                                                 hue=self.config.hue))

        ltransforms.append(ConvertBCHWtoCBHW())
        print("maximum augmentation")
        return transforms.Compose(ltransforms)

    def getTestTransforms(self):
        ltransforms = [ConvertBHWCtoBCHW(),
                       transforms.ConvertImageDtype(torch.float32)]

        if self.config.cameraShake:
            ltransforms.append(CameraShake((self.config.cropShape, self.config.cropShape), self.config.velocityScale,
                                           self.config.springStiffness))

        ltransforms.append(transforms.Resize((self.config.resize, self.config.resize)))

        if self.config.random_crop:
            ltransforms.append(transforms.RandomResizedCrop((self.config.crop_size,
                                                             self.config.crop_size),
                                                            scale=(self.config.scale_size, 1.0)))
        if self.config.vertical_flip:
            ltransforms.append(transforms.RandomVerticalFlip(p=self.config.vertical_flip_prob))

        if self.config.horizontal_flip:
            ltransforms.append(transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob))

        if self.config.random_rotation:
            ltransforms.append(transforms.RandomRotation(self.config.random_rotation_angle))

        if self.config.color_jitter:
            ltransforms.append(transforms.transforms.ColorJitter(brightness=self.config.brightness,
                                                                 contrast=self.config.contrast,
                                                                 saturation=self.config.saturation,
                                                                 hue=self.config.hue))

        ltransforms.append(ConvertBCHWtoCBHW())
        print("maximum augmentation")
        return transforms.Compose(ltransforms)

    def get_clip_step_size(self):
        return self.config.prediction_window_step

    def getSamplers(self, video_datasets):
        samplers_dict = {}
        if self.use_train:
            samplers_dict['train'] = torch.utils.data.SequentialSampler(video_datasets['train'])
        if self.use_val:
            samplers_dict['val'] = torch.utils.data.SequentialSampler(video_datasets['val'])

        return samplers_dict

    def getCollateFn(self):

        return collate_fn_test

        




