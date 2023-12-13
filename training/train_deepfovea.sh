#!/bin/bash

./train_deepfovea.py ldr --filter RT --train_data /media/data/mlready/rt_train --valid_data /media/data/mlready/rt_valid --save_epochs 5 --device cuda --config config/train_temporal_unet.json "$@"