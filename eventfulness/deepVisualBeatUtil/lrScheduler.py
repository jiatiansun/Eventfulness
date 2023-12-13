import torch.optim as optim
import torch

class SchedulerInitializer(object):
    @staticmethod
    def initialize_warmRestart(optimizer, T_0=1, T_mult=2, eta_min=1e-06, last_epoch=-1, verbose=True):
        print("Use T_0={}, T_mult={}, eta_min={}, verbose={} for scheduler".format(T_0, T_mult, eta_min, last_epoch,
                                                                                   verbose))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0, T_mult=T_mult, eta_min=eta_min,
                                                                    last_epoch=last_epoch, verbose=verbose)
    @staticmethod
    def initialize_reducePlateu(optimizer, mode='max',
                                factor=0.1, patience=20,
                                threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-08, eps=1e-08, verbose=True):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode,
                                                          factor=factor, patience=patience,
                                                          threshold=threshold, threshold_mode=threshold_mode,
                                                          cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose)

    @staticmethod
    def initialize_scheduler(optimizer, scheduler_name='', **kwargs):
        if scheduler_name == 'warmRestart':
            return SchedulerInitializer.initialize_warmRestart(optimizer, **kwargs)
        elif scheduler_name == 'ReducePlateu':
            return SchedulerInitializer.initialize_reducePlateu(optimizer, **kwargs)
        else:
            return None