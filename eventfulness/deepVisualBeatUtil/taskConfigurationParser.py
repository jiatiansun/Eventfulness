import argparse
import os
from .fileAndMediaWriters import JsonReadWriter
class TaskConfig(object):
    def __init__(self, args):
        self.__dict__.update(args)

    @classmethod
    def fromParsedConfiguration(cls, obj):
        return cls(obj.__dict__)

    def getDataLoaderConfig(self):
        dataloader_config_keys = {'data_dir','resize','random_crop', 'scale_size',
                                  'num_blurrs', 'num_accS_dir', 'num_velS_dir',
                                  'num_labels', 
                                  'min_sig', 'max_sig', 'd', 'power', 
                                  'crop_size','clip_size','subSample',"subSampleRate",
                                  "cameraShake", "cropShape", "velocityScale","springStiffness",
                                  'prediction_window_step','label_type', 
                                  'use_blur', 'data_augmentation', 'fps', 
                                  'train_clips_per_video','val_clips_per_video',
                                  'batch_size','nworker', 
                                  'cameraShake',
                                  'vertical_flip', 'vertical_flip_prob',
                                  'horizontal_flip', 'horizontal_flip_prob',
                                  'random_rotation', 'random_rotation_angle',
                                  'color_jitter', 'contrast',
                                  'brightness', 'contrast',
                                  'saturation', 'hue'}
        dataloader_dict = {key: self.getParam(key) for key in dataloader_config_keys}

        return TaskConfig(dataloader_dict)

    def getParam(self, attrname):
        return getattr(self, attrname)

    def setParam(self, attrname, value):
        setattr(self, attrname, value)
        return

    def toDict(self):
        return vars(self)


class TaskConfigurationParser(object):

    def __init__(self):
        currDir = os.path.dirname(os.path.realpath(__file__))
        grandparentDir = os.path.dirname(os.path.dirname(currDir))
        augmentationDir = os.path.join(grandparentDir, "dataAugParams")
        schedulerDir = os.path.join(grandparentDir, "schedulerParams")

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_dir", type=str, default="/share/davis/dance_playlists_worst",
                                 help="The directory that stores both the training and validation data")
        self.parser.add_argument("--load_model", default=False, action='store_true',
                                 help="With this option turn on, we would try to test the network against videos in the dataset")
        self.parser.add_argument("--load_model_dir", type=str, default="")
        self.parser.add_argument("--prediction", default=False, action='store_true',
                                 help="With this option turn on, we would try to test the network against video being passed as input")
        self.parser.add_argument("--test_video_path", type=str, default="")
        self.parser.add_argument("--load_model_path", type=str, default="")
        self.parser.add_argument("--load_checkpoint", type=bool, default=False)
        self.parser.add_argument("--load_checkpoint_dir", type=str, default="")
        self.parser.add_argument("--load_epoch", type=int, default=0)
        self.parser.add_argument("--result_dir", type=str, default="results")
        self.parser.add_argument("--prediction_window_step", type=int, default=1)
        self.parser.add_argument("--model_config", type=str, default="2dp1w0t",
                                 help="Enter the string that specify the number of temoral convolution layers in the network")
        self.parser.add_argument("--model_type", type=str, default="regressive",
                                 help="The string for that encodes the model type")
        self.parser.add_argument('--use_pretrained', dest='use_pretrained', action='store_true')
        self.parser.add_argument('--no_pretrain', dest='use_pretrained', action='store_false')
        self.parser.set_defaults(use_pretrained=True)
        # self.parser.add_argument("--use_pretrained",type=bool, default=True,
        #                          help="If the option is turned on we would use pretrained value of resnet being trained on kinetics")
        self.parser.add_argument("--ngpu", type=int, default=1,
                                 help="Number of GPU for training")
        self.parser.add_argument("--nworker", type=int, default=10,
                                 help="Number of workers for training")
        self.parser.add_argument("--resize", type=int, default=128,  help="Resize of simulation ")

        self.parser.add_argument("--batch_size", type=int, default=8,
                                 help="Batch Size")
        self.parser.add_argument("--nepoch", type=int, default=100,
                                 help="Number of Training epochs")
        self.parser.add_argument("--clip_size", type=int, default=72,
                                 help="Video clip sample size for training")
        self.parser.add_argument("--inplanes", type=int, default=64)
        self.parser.add_argument("--layer_num", type=int, default=18,
                                 help="Number of layers in the network")
        self.parser.add_argument("--fps", type=int, default=24,
                                 help="Fps that all the videos are downsampled to") 
        self.parser.add_argument("--lr", type=float, default=1e-4,
                                 help="Learning Rate for Training")
        self.parser.add_argument("--num_accS_dir", type=int, default=0,
                                 help="number of screen acceleration direction")
        self.parser.add_argument("--num_velS_dir", type=int, default=0,
                                 help="number of screen velocity direction")
        self.parser.add_argument("--num_blurrs", type=int, default=0,
                                 help="number of blurred label direction")
        self.parser.add_argument("--num_labels", type=int, default=0,
                                 help="Learning Rate for Training")
        self.parser.add_argument("--min_sig", type=float, default=0.1, help="Minimum Standard Deviation")
        self.parser.add_argument("--max_sig", type=float, default=1.0, help="Maximum Standard Deviation")
        self.parser.add_argument("--d", type=int, default=7, help="Kernel Diameter")
        self.parser.add_argument("--power", type=bool, default=True, help="if the gaussian's standard deviation increases exponentially")
        self.parser.add_argument("--scheduler",type=str, default="",
                                 help="Choose the type of learning rate scheduler")
        self.parser.add_argument("--feature_extract", default=False, action='store_true',
                                 help="If this option is being turned o4n, we are going to train "
                                      "the fc layers after the feature layer. Otherwise, we are going to "
                                      "fine tune the entire model.")
        self.parser.add_argument("--use_blur", default=False, action='store_true',
                                 help="If this is being turned on, we would blur out the envelope")
        self.parser.add_argument("--subSample", type=int, default=1,
                                 help="Subsample the video by this rate and keep the fps")
        self.parser.add_argument("--subSampleRate", type=int, default=1,
                                 help="subsample the video and corresponding decrease the fps")
        self.parser.add_argument("--train_clips_per_video", type=int, default=10,
                                 help="Number of clips we would randomly pick out of a training video")
        self.parser.add_argument("--val_clips_per_video", type=int, default=5,
                                 help="Number of clips we would randomly pick out of a validation video")
        self.parser.add_argument("--label_type", type=str, default="original")
        self.parser.add_argument("--data_augmentation", type=str,
                                 default=os.path.join(augmentationDir, "data_augmentation_null_list.json"),
                                 help="The list of data augmentation to run on the dataset")
        self.parser.add_argument("--scheduler_name", type=str, default="",
                                 help="Name of the learning rate scheduler")
        self.parser.add_argument("--scheduler_kwargs", type=str,
                                 default=os.path.join(schedulerDir, "scheduler_params.json"),
                                help="The list of scheduler parameters")

        self.args = None


    def parse(self):
        self.args = self.parser.parse_args()
        data_augmentation_config = JsonReadWriter.readFromFile(self.args.data_augmentation)
        scheduler_config = JsonReadWriter.readFromFile(self.args.scheduler_kwargs)
        self.args.__dict__.update(data_augmentation_config)
        self.args.scheduler_kwargs = scheduler_config
        return TaskConfig(self.args.__dict__)

