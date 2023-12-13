#!/bin/bash

#./train_temporal.py ldr --filter RT --train_data /media/data/mlready/greencloud_temp_train --valid_data /media/data/mlready/greencloud_temp_valid --save_epochs 10 --device cuda --config config/train_temporal_wnet_small.json "$@"
#./train_temporal.py ldr --filter RT --train_data /media/storage1/davbauer/gc_train --valid_data /media/storage1/davbauer/gc_valid --save_epochs 10 --device cuda --config config/train_temporal_wnet_small.json "$@"
./train_temporal.py ldr --filter RT --train_data /media/data/mlready/rt_train --valid_data /media/data/mlready/rt_valid --save_epochs 5 --device cuda --config config/train_temporal_wnet_small.json "$@"