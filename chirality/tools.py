import torch
from torch import nn
from torch.optim import lr_scheduler
import os

def get_optimizer(model,
                  optim,
                  learning_rate,
                  momentum,
                  weight_decay,
                  amsgrad=False,):
    if optim == 'sgd':
        optim_module = torch.optim.SGD
        optim_param = {"lr" : learning_rate,
                       "momentum": momentum}
        if weight_decay != None:
            optim_param["weight_decay"] = weight_decay
    elif optim == "adam":
        optim_module = torch.optim.Adam
        optim_param = {"lr": learning_rate,
                       "weight_decay": weight_decay,
                       "amsgrad": amsgrad}
    else:
        print("Not supported")

    optimizer = optim_module(
                    filter(lambda x : x.requires_grad, model.parameters()),
                    **optim_param
                )
    return optimizer

def get_scheduler(optimizer, decay_step, gamma=0.1):
    scheduler = lr_scheduler.StepLR(
                    optimizer,
                    step_size=decay_step,
                    gamma=gamma
                )
    return scheduler

def get_log_name(args):
    logging_details = ['optim', args.optimizer,
                       'lr', str(args.learning_rate),
                       'decaystep', str(args.decay_step),
                       'wd', str(args.weight_decay),
                       'batch', str(args.batch_size),
                       'diff',str(args.name_different)]
    return "_".join(logging_details)

def get_dir_name(out_dir,log_name):
    return os.path.join(out_dir, log_name)