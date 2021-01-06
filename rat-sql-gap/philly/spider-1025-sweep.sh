#!/bin/bash -x

source ./setup.sh
python3 -u train.py \
    --config configs/spider-20190205/nl2code-1017-bert-philly.jsonnet \
    --config-args "{bs: $1, num_batch_accumulated: $2, lr: $3, bert_lr: $4, att: 0, end_lr: 0, sc_link: true, use_align_mat: true, use_align_loss: true, bert_token_type: $5,  decoder_hidden_size: 512, clause_order: '$6', end_with_from: true, data_path: '$PT_DATA_DIR/'}" \
    --logdir "$PT_OUTPUT_DIR/logdirs"
