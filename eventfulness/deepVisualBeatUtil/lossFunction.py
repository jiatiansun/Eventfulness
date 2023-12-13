import torch.nn as nn

class LossInitializer(object):

    @staticmethod
    def initializeLoss(model_type, **kwargs):
        if model_type == "discriminant":
            criterion = nn.BCEWithLogitsLoss(**kwargs)
        elif model_type == "regressive":
            criterion = nn.MSELoss()
        else:
            raise Exception("undefined loss function type")
        return criterion