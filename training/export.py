#!/usr/bin/env python3

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

from doctest import Example
import os
import time
import numpy as np
import torch
import torch.quantization
import torch.onnx
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

from config import *
from util import *
from dataset import *
from model.settings import *
from color import *
from result import *
from model.unet import *
from model.wnet import *
from model.temporal_wnet import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Performs inference on a dataset using the specified training result.')

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Open the result
  result_dir = get_result_dir(cfg)
  if not os.path.isdir(result_dir):
    error('result does not exist')
  print('Result:', cfg.result)

  # Load the result config
  result_cfg = load_config(result_dir)
  cfg.temp_size = result_cfg.temp_size
  cfg.features = result_cfg.features
  cfg.transfer = result_cfg.transfer
  cfg.model    = result_cfg.model
  if 'model_config' in result_cfg:
    cfg.model_config = result_cfg.model_config
  target_feature = 'hdr' if 'hdr' in cfg.features else 'ldr'


  # Initialize the model
  model = get_model(cfg)
  print(model)
  model.to(device)

  # Load the trained weights
  load_checkpoint(result_dir, device, cfg.checkpoint, model)

  if cfg.format == 'onnx':
    B, C, H, W = cfg.input_dimensions
    dummy_input = torch.randn(B, C, H, W, device=device)
    _, dummy_recurrent = model(dummy_input, None)

    # enable_onnx_checker needs to be disabled. See notes below.
    torch.onnx.export(model, (dummy_input, dummy_recurrent), os.path.join(result_dir, f"{cfg.result}.onnx"), verbose=True, opset_version=11)

  if cfg.format == 'torch':
    traced_model = torch.jit.script(model)
    traced_model.save(os.path.join(result_dir, cfg.result+'_traced.pt'))



if __name__ == '__main__':
  main()
