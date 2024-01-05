#!/usr/bin/env python3

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

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
from model.settings import *
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
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

  # Initialize the loss function
  criterion = get_loss_function(cfg)
  criterion.to(device)

  # Initialize the transfer function
  transfer = get_transfer_function(cfg)

  # Initialize the optimizer
  optimizer = opt.get_optimizer(cfg.train_config, model.parameters())

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
    checkpoint = load_checkpoint(result_dir, device, last_epoch, model, optimizer)
    step = checkpoint['step']
    last_step = step - 1 # will be incremented by the LR scheduler init
  else:
    if rank == 0:
      print('Result:', cfg.result)
      os.makedirs(result_dir)

      # Save the config
      save_config(result_dir, cfg)

      # Save the source code
      src_filenames = glob(os.path.join(os.path.dirname(sys.argv[0]), '*.py'))
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
    exit() # nothing to do

  # Reset the random seed if resuming result
  if cfg.seed is not None and start_epoch > 1:
    seed = cfg.seed + start_epoch - 1
    torch.manual_seed(seed)

  # Initialize the training dataset
  train_data = TrainingDataset(cfg, cfg.train_data)
  if len(train_data) > 0:
    if rank == 0:
      print('Training images:', train_data.num_images)
  else:
    error('no training images (forgot to run preprocess?)')
  train_loader, train_sampler = get_data_loader(rank, cfg, train_data, shuffle=True)
  train_steps_per_epoch = len(train_loader)

  # Initialize the learning rate scheduler
  lr_scheduler = sch.get_scheduler(cfg, optimizer)

  # Initialize the validation dataset
  valid_data = ValidationDataset(cfg, cfg.valid_data)
  if len(valid_data) > 0:
    if rank == 0:
      print('Validation images:', valid_data.num_images)
    valid_loader, valid_sampler = get_data_loader(rank, cfg, valid_data, shuffle=False)
    valid_steps_per_epoch = len(valid_loader)

  # Initialize the summary writer
  log_dir = get_result_log_dir(result_dir)
  if rank == 0:
    summary_writer = SummaryWriter(log_dir)
    if step == 0:
      summary_writer.add_scalar('learning_rate', lr_scheduler.get_last_lr()[0], 0)

  # Training and evaluation loops
  if rank == 0:
    print()
    progress_format = '%-5s %' + str(len(str(cfg.epochs))) + 'd/%d: ' % cfg.epochs
    total_start_time = time.time()

  for epoch in range(start_epoch, cfg.epochs+1):
    if rank == 0:
      start_time = time.time()
      progress = ProgressBar(train_steps_per_epoch, progress_format % ('Train', epoch))

    # Switch to training mode
    model.train()
    train_loss = 0.

    # Iterate over the batches
    if distributed:
      train_sampler.set_epoch(epoch)

  
    for i, batch in enumerate(train_loader, 0):
      # Get the batch
      input, target = batch
      input  = input.to(device,  non_blocking=True).float()
      target = target.to(device, non_blocking=True).float()

      # save_image('out'+str(i)+'.png', tensor_to_image(input[0]))

      # Run a training step
      optimizer.zero_grad()
      output = model(input)

      loss = criterion(output, target)
      loss.backward()
      
      # Next step
      optimizer.step()
      step += 1
      train_loss += loss
      if rank == 0:
        progress.next()

    lr_scheduler.step(epoch)
    # Compute the average training loss
    if distributed:
      dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    train_loss = train_loss.item() / (train_steps_per_epoch * cfg.num_devices)

    # Write summary
    if rank == 0:
      if epoch == 1:
        summary_writer.add_graph(unwrap_module(model), input)
      summary_writer.add_scalar('learning_rate', lr_scheduler.get_last_lr()[0], epoch)
      summary_writer.add_scalar('loss', train_loss, epoch)

    # Print stats
    if rank == 0:
      duration = time.time() - start_time
      total_duration = time.time() - total_start_time
      lr = lr_scheduler.get_last_lr()[0]
      images_per_sec = len(train_data) / duration
      eta = ((cfg.epochs - epoch) * total_duration / (epoch + 1 - start_epoch))
      progress.finish('loss=%.6f, lr=%.6f  (%.1f images/s, %s, eta %s)'
                      % (train_loss, lr, images_per_sec, format_time(duration), format_time(eta, precision=2)))

    if ((cfg.valid_epochs > 0 and epoch % cfg.valid_epochs == 0) or epoch == cfg.epochs) \
      and len(valid_data) > 0:
      # Validation
      if rank == 0:
        start_time = time.time()
        progress = ProgressBar(valid_steps_per_epoch, progress_format % ('Valid', epoch))

      # Switch to evaluation mode
      model.eval()
      valid_loss = 0.

      # Iterate over the batches
      with torch.no_grad():
        for _, batch in enumerate(valid_loader, 0):
          # Get the batch
          input, target = batch
          input  = input.to(device,  non_blocking=True).float()
          target = target.to(device, non_blocking=True).float()

          # Run a validation step
          output = model(input)
          loss = criterion(output, target)

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
      save_checkpoint(result_dir, epoch, step, model, optimizer, lr_scheduler)

  # Cleanup
  cleanup_worker(cfg)

if __name__ == '__main__':
  main()