"""
This file is adapted from https://github.com/mahmoodlab/PANTHER/blob/main/src/utils/utils.py
"""
import torch.optim as optim
from transformers import (get_constant_schedule_with_warmup, 
                         get_linear_schedule_with_warmup, 
                         get_cosine_schedule_with_warmup)

def get_optim(args, model=None, parameters=None):
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    if parameters is None:
        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(
            n, p) and p.requires_grad]
        parameters = [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.weight_decay},
        ]

    if args.opt == "adamW":
        optimizer = optim.AdamW(parameters, lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9)
    elif args.opt == 'RAdam':
        optimizer = optim.RAdam(parameters, lr=args.lr)
    else:
        raise NotImplementedError
    return optimizer

def get_lr_scheduler(args, optimizer):
    scheduler_name = args.lr_scheduler
    warmup_steps = args.warmup_steps
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs
    assert not (warmup_steps > 0 and warmup_epochs > 0), "Cannot have both warmup steps and epochs"
    sgd_steps_in_one_epoch = args.sgd_steps_in_one_epoch
    
    if warmup_steps > 0:
        warmup_steps = warmup_steps
    elif warmup_epochs > 0:
        warmup_steps = sgd_steps_in_one_epoch * warmup_epochs
    else:
        warmup_steps = 0

    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=sgd_steps_in_one_epoch * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=sgd_steps_in_one_epoch * epochs,
        )

    return lr_scheduler