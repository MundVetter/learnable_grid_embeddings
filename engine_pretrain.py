# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.utils import make_grid


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (glimpses, locations, imgs, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        glimpses = glimpses.to(device, non_blocking=True)
        locations = locations.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _ = model(glimpses, locations, imgs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(data_loader: Iterable, model: torch.nn.Module, device: torch.device, epoch: int, log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 20

    with torch.no_grad():
        for data_iter_step, (glimpses, locations, imgs, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            glimpses = glimpses.to(device, non_blocking=True)
            locations = locations.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)


            with torch.cuda.amp.autocast():
                loss, predictions = model(glimpses, locations, imgs)

            
            if data_iter_step == 0 and log_writer is not None:
                img_grid = make_grid(imgs)
                predictions_grid = make_grid(model.unpatchify(predictions))
                log_writer.add_image('images', img_grid, epoch)
                log_writer.add_image('predictions', predictions_grid, epoch)

                input_imgs = []
                for glimpse_item, location_item in zip(glimpses,  locations):
                    img = misc.create_image_from_glimpses(glimpse_item, location_item, args.input_size, args.in_chans)
                    input_imgs.append(img)
                input_imgs = torch.stack(input_imgs)
                input_imgs_grid = make_grid(input_imgs)
                log_writer.add_image('input glimpses', input_imgs_grid, epoch)

            loss_value = loss.item()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}