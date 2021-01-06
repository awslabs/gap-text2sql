#!/bin/bash -x

source ./setup.sh
python3 bert_train.py \
    --config configs/spider-20190205/nl2code-0822-bert-philly.jsonnet \
    --config-args "{bs: $6, lr: 1e-3, bert_lr: $1, encoder_recurrent_size: $2, decoder_recurrent_size: $3, decoder_dropout: $4, num_layers: $5, end_lr: 0, att: 0, setting: 'bert', loss_type: 'softmax', data_path: '$PT_DATA_DIR/'}" \
    --logdir "$PT_OUTPUT_DIR/logdirs"
