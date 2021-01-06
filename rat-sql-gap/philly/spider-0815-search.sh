#!/bin/bash -x

source ./setup.sh
python3 train.py \
    --config configs/spider-20190205/nl2code-0715-align-philly.jsonnet \
    --config-args "{bs: $1, lr: $2, top_k_learnable: $3, decoder_recurrent_size: $4, decoder_dropout: $5, num_layers: $6, end_lr: 0, att: 0, setting: 'basic', loss_type: 'softmax', data_path: '$PT_DATA_DIR/'}" \
    --logdir "$PT_OUTPUT_DIR/logdirs"
