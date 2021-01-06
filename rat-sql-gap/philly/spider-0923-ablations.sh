#!/bin/bash -x

source ./setup.sh
python3 train.py \
    --config configs/spider-20190205/nl2code-0923-ablations.jsonnet \
    --config-args "{align_mat: $1, align_loss: $2, att: $3, data_path: '$PT_DATA_DIR/'}" \
    --logdir "$PT_OUTPUT_DIR/logdirs"
