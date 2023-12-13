#!/bin/bash

./train_temporal_gan.py ldr --filter RT --train_data /media/data/mlready/greencloud_temp_train --valid_data /media/data/mlready/greencloud_temp_valid --save_epochs 10 --device cuda --config config/train_temporal_wnet_gan_small.json "$@"