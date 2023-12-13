#!/bin/bash

docker run \
    --gpus all \
    -it \
    --rm \
    --ipc=host \
    --user `id -u`:`id -g` \
    -v `pwd`:/workspace/fovolnet \
    -v /media:/media \
    -w /workspace/fovolnet/training \
    "$@" \
    pytorch-dl