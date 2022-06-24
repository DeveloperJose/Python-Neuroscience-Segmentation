"""
Authors: Bibek Aryal, Alex Arnal, Jose Perez
"""
from segmentation.data.data import fetch_loaders
from segmentation.model.frame import Framework
from segmentation.model.losses import unifiedloss
import segmentation.model.functions as fn
import keys

import os
import yaml
import json
import pathlib
import warnings
import pdb
import torch
import logging
import time
from timeit import default_timer as timer
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import numpy as np
from addict import Dict
from twilio.rest import Client
import segmentation_models_pytorch as smp
from ray import tune
from ray.tune.schedulers import ASHAScheduler
warnings.filterwarnings("ignore")


def train(hp, conf, checkpoint_dir=None):
    # Uses code from https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    # %% Dataset loading
    data_dir = pathlib.Path(conf.data_dir)
    batch_size = 12
    if hp['encoder_depth'] > 4 or hp['decoder_pab'] > 32:
        batch_size = 2
    
    loaders = fetch_loaders(processed_dir=data_dir / "processed",
                            batch_size=batch_size,
                            use_channels=conf.use_channels)

    # %% Model loading
    cm = hp['decoder_channels_mult']
    cs = hp['decoder_channels_start']
    if cm == 0:
        dec_channels = [cs for i in range(hp['encoder_depth'])]
    else:
        dec_channels = [cm*cs*(i+1) for i in range(hp['encoder_depth'])]

    print('Net Decoder Channels: ', dec_channels)
    net = smp.MAnet(encoder_name=hp['encoder_name'],
                    encoder_depth=hp['encoder_depth'],
                    encoder_weights=None,
                    decoder_use_batchnorm=hp['decoder_batchnorm'],
                    decoder_channels=dec_channels,
                    decoder_pab_channels=hp['decoder_pab'],
                    in_channels=conf.model_opts.args.inchannels,
                    classes=conf.model_opts.args.outchannels,
                    activation=None,
                    # aux_params=dict(dropout=hp['dropout'], classes=1, activation=None, pooling='max')
                    )

    activation_name = hp['loss_act']
    if activation_name == 'sigmoid':
        act = torch.nn.Sigmoid()
    elif activation_name == 'softmax':
        act = torch.nn.Softmax(dim=1)
    else:
        raise Exception(f'Activation {activation_name} not currently supported')

    loss_fn = unifiedloss(weight=hp['loss_weight'],
                          delta=hp['loss_delta'],
                          gamma=hp['loss_gamma'],
                          act=act,
                          label_smoothing=hp['loss_smooth'],
                          outchannels=conf.model_opts.args.outchannels)
    opt = optim.Adam(net.parameters(), lr=hp['lr'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        opt.load_state_dict(optimizer_state)

    frame = Framework(net, loss_fn, opt, conf)
    tune.report(val_loss=np.inf, val_iou=0, bs=batch_size, epoch_t=np.inf, train_t=np.inf, val_t=np.inf)

    # %% Main epoch loop
    for epoch in range(conf.epochs):
        epoch_start = timer()
        loss = {}

        # Training Loop
        train_start = timer()
        loss["train"], train_metrics = fn.train_epoch(epoch, loaders["train"], frame, conf)
        train_end = timer()

        # Validation loop
        val_start = timer()
        loss["val"], val_metrics = fn.validate(epoch, loaders["val"], frame, conf)
        val_end = timer()

        # Save checkpoint
        if epoch % 10 == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save((frame.model.state_dict(), frame.optimizer.state_dict()), path)

        epoch_duration = timer() - epoch_start
        train_duration = train_end- train_start
        val_duration = val_end - val_start
        # Report to RayTune
        val_iou = val_metrics['IoU'][1].item()
        tune.report(val_loss=loss['val'], val_iou=val_iou, bs=batch_size, epoch_t=epoch_duration, train_t=train_duration, val_t=val_duration)

    print('Finished training')


if __name__ == "__main__":
    client = Client(keys.account_sid, keys.auth_token)
    conf = Dict(yaml.safe_load(open('./conf/hp_tune.yaml')))

    hp_space = {
        'lr': tune.loguniform(1e-4, 1e-1),
        #'batch_size': 10, # tune.choice([8, 16]),
        'encoder_name': 'resnet18', # tune.choice(['resnet18', 'resnet34', 'vgg11', 'vgg16']), # https://github.com/qubvel/segmentation_models.pytorch#encoders
        'encoder_depth': tune.choice([3, 4]),
        'decoder_batchnorm': tune.choice([True, False]),
        'decoder_channels_start': tune.randint(1, 4),
        'decoder_channels_mult': tune.randint(0, 2),
        'decoder_pab': tune.qrandint(16, 64, 16),
        # 'dropout': tune.uniform(0, 0.5),
        'loss_weight': tune.quniform(0, 1, 0.1),
        'loss_delta': tune.quniform(0, 1, 0.1),
        'loss_gamma': tune.quniform(0, 1, 0.1),
        'loss_act': tune.choice(['sigmoid', 'softmax']),
        'loss_smooth': tune.quniform(0, 1, 0.1),
    }

    # hp_test_space = {
    #     'lr': 1e-4,
    #     'batch_size': 16,
    #     'encoder_name': 'resnet18',  # https://github.com/qubvel/segmentation_models.pytorch#encoders
    #     'encoder_depth': 4,
    #     'decoder_batchnorm': True,
    #     'decoder_channels_start': 4,
    #     'decoder_channels_mult': 0,
    #     'decoder_pab': 64,
    #     'dropout': 0,
    #     'loss_weight': 0.5,
    #     'loss_delta': 0.6,
    #     'loss_gamma': 0.5,
    #     'loss_act': tune.choice([torch.nn.Sigmoid(), torch.nn.Softmax(dim=1)]),
    #     'loss_smooth': 0.1,
    # }

    scheduler = ASHAScheduler(metric='val_iou',
                              mode='max',
                              max_t=conf.epochs,
                              grace_period=1,
                              reduction_factor=2
                              )
    reporter = tune.CLIReporter(metric_columns=['training_iteration', 'bs', 'epoch_t', 'train_t', 'val_t', 'val_loss', 'val_iou'])

    result = tune.run(
        partial(train, conf=conf),
        resources_per_trial={"cpu": 4, "gpu": 1},
        max_concurrent_trials=2, # How many GPUs to use
        config = hp_space,
        num_samples=500, # Total times the search space is sampled
        scheduler=scheduler,
        progress_reporter=reporter,
        name=conf.run_name,
        local_dir='runs/'
    )

    best_trial = result.get_best_trial('val_iou', 'max', 'last')
    msg = f'{conf.run_name} has completed, best trial -> Config={best_trial.config} | val_loss={best_trial.last_result["val_loss"]} | val_iou={best_trial.last_debug["val_iou"]}'
    print(msg)

    # %% Send a text message via Twilio
    client.messages.create(
        body=msg,
        from_=keys.src_phone,
        to=keys.dst_phone
    )
