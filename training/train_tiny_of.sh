#!/bin/bash

#./train_temporal.py ldr --filter RT --train_data /media/data/mlready/greencloud_temp_train --valid_data /media/data/mlready/greencloud_temp_valid --save_epochs 10 --device cuda --config config/train_temporal_wnet_small.json "$@"
./train_temporal_opticalflow.py ldr --filter RT --train_data /media/data/mlready/gc8_train --valid_data /media/data/mlready/gc8_valid --save_epochs 5 --device cuda --config config/train_temporal_wnet_tiny.json "$@"