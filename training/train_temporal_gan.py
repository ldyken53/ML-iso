#!/usr/bin/env python3

# Copyright 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from glob import glob
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import training.optimizer as opt
import training.schedule as sch

from config import *
from dataset import *
from model.discriminator import *
from model.settings import *
from model.common import *
from loss import *
from learning_rate import *
from result import *
from util import *


def main():
    # Parse the command line arguments
    cfg = parse_args(description='Trains a model using preprocessed datasets.')

    # Start the worker(s)
    start_workers(cfg, main_worker)

# Worker function


def main_worker(rank, cfg):
    # Initialize the worker
    distributed = init_worker(rank, cfg)

    # Initialize the random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    # Initialize the PyTorch device
    device = init_device(cfg, id=rank)

    # Initialize the model
    model = get_model(cfg)
    print(model)
    model.to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True)

    # Initialize the loss functions
    criterion_spatial = get_loss_function(cfg, 'spatial_loss')
    criterion_spatial.to(device)

    criterion_temporal = get_loss_function(cfg, 'temporal_loss')
    criterion_temporal.to(device)

    # Initialize discriminator
    discriminator = Discriminator(cfg.model_config, 3)
    discriminator.to(device)
    if distributed:
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator, device_ids=[rank], find_unused_parameters=True)

    # Initialize the transfer function
    transfer = get_transfer_function(cfg)

    # Initialize the optimizer
    optimizer = opt.get_optimizer(cfg.train_config, model.parameters())
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # Check whether the result already exists
    result_dir = get_result_dir(cfg)
    resume = os.path.isdir(result_dir)

    # Sync the workers (required due to the previous isdir check)
    if distributed:
        dist.barrier()

    # Start or resume training
    if resume:
        if rank == 0:
            print('Resuming result:', cfg.result)

        # Load and verify the config
        result_cfg = load_config(result_dir)
        if set(result_cfg.features) != set(cfg.features):
            error('input feature set mismatch')

        # Restore the latest checkpoint
        last_epoch = get_latest_checkpoint_epoch(result_dir)
        checkpoint = load_checkpoint(
            result_dir, device, last_epoch, model, optimizer)
        step = checkpoint['step']
        last_step = step - 1  # will be incremented by the LR scheduler init
    else:
        if rank == 0:
            print('Result:', cfg.result)
            os.makedirs(result_dir)

            # Save the config
            save_config(result_dir, cfg)

            # Save the source code
            src_filenames = glob(os.path.join(os.path.dirname(
                sys.argv[0]), '**/*.py'), recursive=True)
            src_zip_filename = os.path.join(result_dir, 'src.zip')
            save_zip(src_zip_filename, src_filenames)

        last_epoch = 0
        step = 0
        last_step = -1

    # Make sure all workers have loaded the checkpoint
    if distributed:
        dist.barrier()

    start_epoch = last_epoch + 1
    if start_epoch > cfg.epochs:
        exit()  # nothing to do

    # Reset the random seed if resuming result
    if cfg.seed is not None and start_epoch > 1:
        seed = cfg.seed + start_epoch - 1
        torch.manual_seed(seed)

    # Initialize the training dataset
    #train_data = MockTrainingTemporalDataset(cfg, cfg.train_data)
    train_data = TrainingTemporalDataset(cfg, cfg.train_data)

    if len(train_data) > 0:
        if rank == 0:
            print('Training images:', train_data.num_images)
    else:
        error('no training images (forgot to run preprocess?)')
    train_loader, train_sampler = get_data_loader(
        rank, cfg, train_data, shuffle=True)
    train_steps_per_epoch = len(train_loader)

    # Initialize the learning rate scheduler
    lr_scheduler = sch.get_scheduler(cfg, optimizer)

    # Initialize the validation dataset
    #valid_data = MockValidationTemporalDataset(cfg, cfg.valid_data)
    valid_data = ValidationTemporalDataset(cfg, cfg.valid_data)
    if len(valid_data) > 0:
        if rank == 0:
            print('Validation images:', valid_data.num_images)
        valid_loader, valid_sampler = get_data_loader(
            rank, cfg, valid_data, shuffle=False)
        valid_steps_per_epoch = len(valid_loader)

    # Initialize the summary writer
    log_dir = get_result_log_dir(result_dir)
    if rank == 0:
        summary_writer = SummaryWriter(log_dir)
        if step == 0:
            summary_writer.add_scalar(
                'learning_rate', lr_scheduler.get_last_lr()[0], 0)

    # Training and evaluation loops
    if rank == 0:
        print()
        progress_format = '%-5s %' + \
            str(len(str(cfg.epochs))) + 'd/%d: ' % cfg.epochs
        total_start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs+1):
        if rank == 0:
            start_time = time.time()
            progress = ProgressBar(train_steps_per_epoch,
                                   progress_format % ('Train', epoch))

        # Switch to training mode
        model.train()
        train_loss = 0.
        train_spatial_loss = 0.
        train_temp_loss = 0.
        train_adverserial_loss = 0.
        train_crit_fake = 0.
        train_crit_real = 0.
        train_crit = 0.

        # Iterate over the batches
        if distributed:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader, 0):
            # Get the batch
            input, target = batch
            input = input.to(device,  non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()

            # for _ in range(5):
            # Train Discriminator
            optimizer_d.zero_grad()
            fake_outputs = list()
            fake_features = None
            for t in range(cfg.temp_size):
                output, fake_features = model(
                    input[:, t, ...], fake_features)
                fake_outputs.append(output.detach())
            # .permute(0,2,1,3,4).detach()
            fake_outputs = torch.stack(fake_outputs, dim=1)
            true_outputs = target  # .permute(0,2,1,3,4)
            grad_penalty = gradient_penalty(discriminator, true_outputs, fake_outputs)
            loss_fake = torch.mean(discriminator(fake_outputs))
            loss_real = torch.mean(discriminator(true_outputs))
            loss_d = -loss_real + loss_fake + grad_penalty
            loss_d.backward()
            optimizer_d.step()

            # Clip weights
            # TODO Disabled due to gradient penalty
            # for param in discriminator.parameters():
            #     param.data.clamp_(-0.01, 0.01)

            train_crit_fake += loss_fake/5
            train_crit_real += loss_real/5
            train_crit += loss_d/5

            if i%5 != 0:
                if rank == 0:
                    progress.next()
                continue

            # Train Generator

            # Run a training step
            optimizer.zero_grad()

            # Create output buffer
            outputs = list()
            features = None

            loss = 0
            spatial_loss = 0
            temporal_loss = 0
            adverserial_loss = 0
            for t in range(cfg.temp_size):
                output, features = model(input[:, t, ...], features)
                outputs.append(output)
                loss_factor = 1 - np.exp(-0.5*t)
                spatial_loss += loss_factor * \
                    criterion_spatial(outputs[-1], target[:, t, ...])
                if t > 0:
                    temporal_loss += criterion_temporal(
                        outputs[-1] - outputs[-2], target[:, t, ...] - target[:, t-1, ...])

            spatial_loss /= cfg.temp_size
            temporal_loss /= (cfg.temp_size - 1)
            # .permute(0,2,1,3,4)))
            adverserial_loss = -torch.mean(discriminator(torch.stack(outputs, dim=1)))
            if i % 100 == 0 and rank == 0:
                save_image('out_input.png', tensor_to_image(
                    input[0, -1, 0:3, ...].detach()))
                save_image('out_pred.png', tensor_to_image(
                    outputs[-1][0].detach()))
                save_image('out_target.png', tensor_to_image(
                    target[0, -1, ...].detach()))
                save_image('out_diff.png', np.abs(tensor_to_image(
                    target[0, -1, ...].detach()) - tensor_to_image(outputs[-1][0].detach())))
            # spatial_loss = criterion_spatial(outputs[-1], target[:, -1, ...])
            # temporal_loss = criterion_temporal(outputs[-1] - outputs[-2], target[:, -1, ...] - target[:, -2, ...])
            loss = 0.01 * adverserial_loss + 0.2 * spatial_loss + 0.02 * temporal_loss
            loss.backward()

            # Next step
            optimizer.step()
            step += 1
            train_loss += loss
            train_spatial_loss += spatial_loss
            train_temp_loss += temporal_loss
            train_adverserial_loss += adverserial_loss
            if rank == 0:
                progress.next()

        lr_scheduler.step(epoch)
        # Compute the average training loss
        if distributed:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_spatial_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_temp_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_adverserial_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_crit_fake, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_crit_real, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_crit, op=dist.ReduceOp.SUM)
        train_loss = 5*train_loss.item() / (train_steps_per_epoch * cfg.num_devices)
        train_spatial_loss = 5*train_spatial_loss.item(
        ) / (train_steps_per_epoch * cfg.num_devices)
        train_temp_loss = 5*train_temp_loss.item() / (train_steps_per_epoch * cfg.num_devices)
        train_adverserial_loss = 5*train_adverserial_loss.item(
        ) / (train_steps_per_epoch * cfg.num_devices)
        train_crit_fake = 5*train_crit_fake.item() / (train_steps_per_epoch * cfg.num_devices)
        train_crit_real = 5*train_crit_real.item() / (train_steps_per_epoch * cfg.num_devices)
        train_crit = 5*train_crit.item() / (train_steps_per_epoch * cfg.num_devices)

        # Write summary
        if rank == 0:
            # if epoch == 1:
            #   summary_writer.add_graph(unwrap_module(model), [input[:, -1, ...], None])
            summary_writer.add_scalar(
                'learning_rate', lr_scheduler.get_last_lr()[0], epoch)
            summary_writer.add_scalar('loss', train_loss, epoch)
            summary_writer.add_scalar(
                'spatial_loss', train_spatial_loss, epoch)
            summary_writer.add_scalar('temporal_loss', train_temp_loss, epoch)
            summary_writer.add_scalar(
                'adverserial_loss', train_adverserial_loss, epoch)
            summary_writer.add_scalar('crit_real', train_crit_real, epoch)
            summary_writer.add_scalar('crit_fake', train_crit_fake, epoch)
            summary_writer.add_scalar('crit_loss', train_crit, epoch)

        # Print stats
        if rank == 0:
            duration = time.time() - start_time
            total_duration = time.time() - total_start_time
            lr = lr_scheduler.get_last_lr()[0]
            images_per_sec = len(train_data) / duration
            eta = ((cfg.epochs - epoch) * total_duration /
                   (epoch + 1 - start_epoch))
            progress.finish('loss=%.6f spatial_loss=%.6f temp_loss=%.6f adverserial_loss=%.6f crit_loss=%.6f crit_real=%.6f crit_fake=%.6f lr=%.6f  (%.1f images/s, %s, eta %s)'
                            % (train_loss, train_spatial_loss, train_temp_loss, train_adverserial_loss, train_crit, train_crit_real, train_crit_fake, lr, images_per_sec, format_time(duration), format_time(eta, precision=2)))

        if ((cfg.valid_epochs > 0 and epoch % cfg.valid_epochs == 0) or epoch == cfg.epochs) \
                and len(valid_data) > 0:
            # Validation
            if rank == 0:
                start_time = time.time()
                progress = ProgressBar(
                    valid_steps_per_epoch, progress_format % ('Valid', epoch))

            # Switch to evaluation mode
            model.eval()
            valid_loss = 0.

            # Iterate over the batches
            with torch.no_grad():
                for _, batch in enumerate(valid_loader, 0):
                    # Get the batch
                    input, target = batch
                    input = input.to(device,  non_blocking=True).float()
                    target = target.to(device, non_blocking=True).float()

                    # Create output buffer
                    outputs = list()
                    features = None

                    loss = 0
                    spatial_loss = 0
                    temporal_loss = 0
                    # Run a validation step
                    for t in range(cfg.temp_size):
                        output, features = model(input[:, t, ...], features)
                        outputs.append(output)
                        loss_factor = 1 - np.exp(-0.5*t)
                        spatial_loss += loss_factor * \
                            criterion_spatial(outputs[-1], target[:, t, ...])
                        if t > 0:
                            temporal_loss += criterion_temporal(
                                outputs[-1] - outputs[-2], target[:, t, ...] - target[:, t-1, ...])

                    spatial_loss /= cfg.temp_size
                    temporal_loss /= (cfg.temp_size - 1)
                    #   loss += criterion_spatial(output, target[:, t, ...]) # Compute spatial loss for the current frame

                    # loss /= cfg.temp_size
                    # spatial_loss = criterion_spatial(outputs[-1], target[:, -1, ...])
                    # temporal_loss = criterion_temporal(outputs[-1] - outputs[-2], target[:, -1, ...] - target[:, -2, ...])
                    loss = 0.8 * spatial_loss + 0.2 * temporal_loss

                    # Next step
                    valid_loss += loss
                    if rank == 0:
                        progress.next()

            # Compute the average validation loss
            if distributed:
                dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
            valid_loss = valid_loss.item() / (valid_steps_per_epoch * cfg.num_devices)

            # Write summary
            if rank == 0:
                summary_writer.add_scalar('valid_loss', valid_loss, epoch)

            # Print stats
            if rank == 0:
                duration = time.time() - start_time
                images_per_sec = len(valid_data) / duration
                progress.finish('valid_loss=%.6f  (%.1f images/s, %.1fs)'
                                % (valid_loss, images_per_sec, duration))

        if (rank == 0) and ((cfg.save_epochs > 0 and epoch % cfg.save_epochs == 0) or epoch == cfg.epochs):
            # Save a checkpoint
            save_checkpoint(result_dir, epoch, step,
                            model, optimizer, lr_scheduler)

    # Cleanup
    cleanup_worker(cfg)


if __name__ == '__main__':
    main()
