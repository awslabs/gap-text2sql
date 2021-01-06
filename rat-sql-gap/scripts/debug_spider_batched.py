import argparse
import collections
import datetime
import json
import os

import _jsonnet
import attr
import asdl
import numpy as np
import torch

from seq2struct import ast_util
from seq2struct import datasets
from seq2struct import models
from seq2struct import optimizers

from seq2struct.utils import registry
from seq2struct.utils import saver as saver_mod
from seq2struct.utils import vocab


def check_close(a, b):
    assert (a - b).abs().max() < 1e-5


def test_enc_equal(input0, inputb, sequential):
    input1 = input0
    inputc = inputb
    inputb0 = [inputb[0]]

    input0_history = [input0]
    inputb_history = [inputb]
    inputb0_history = [inputb0]

    for m in sequential:
        input0 = m.forward_unbatched(input0)
        input1 = m.forward_unbatched(input1)
        inputb0 = m.forward(inputb0)
        inputb = m.forward(inputb)
        inputc = m.forward(inputc)

        input0_history.append(input0)
        inputb_history.append(inputb)
        inputb0_history.append(inputb0)

        input0_enc, input0_bounds = input0
        input1_enc, input1_bounds = input1
        inputb0_enc, inputb0_bounds = inputb0
        inputb_enc, inputb_bounds = inputb
        inputc_enc, inputc_bounds = inputc

        check_close(input0_enc.squeeze(1), inputb0_enc.select(0))
        check_close(input0_enc.squeeze(1), inputb_enc.select(0))
        check_close(input0_enc.squeeze(1), input1_enc.squeeze(1))
        check_close(inputb_enc.select(0), inputc_enc.select(0))
        assert np.array_equal(input0_bounds, inputb_bounds[0])

    return input0, inputb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    # 0. Construct preprocessors
    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'],
        unused_keys=('name',))
    model_preproc.load()

    # 1. Construct model
    model = registry.construct('model', config['model'],
            unused_keys=('encoder_preproc', 'decoder_preproc'), preproc=model_preproc, device=device)
    model.to(device)
    model.eval()

    # 3. Get training data somewhere
    train_data = model_preproc.dataset('train')
    train_eval_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=10,
            collate_fn=lambda x: x)

    batch = next(iter(train_eval_data_loader))
    descs = [x for x, y in batch]

    q0, qb = test_enc_equal([descs[0]['question']], [[desc['question']] for desc in descs], model.encoder.question_encoder)

    c0, cb = test_enc_equal(descs[0]['columns'], [desc['columns'] for desc in descs], model.encoder.column_encoder)

    t0, tb = test_enc_equal(descs[0]['tables'], [desc['tables'] for desc in descs], model.encoder.table_encoder)

    q0_enc, c0_enc, t0_enc = model.encoder.encs_update.forward_unbatched(
            descs[0], q0[0], c0[0], c0[1], t0[0], t0[1])
    qb_enc, cb_enc, tb_enc = model.encoder.encs_update.forward(
            descs, qb[0], cb[0], cb[1], tb[0], tb[1])

    check_close(q0_enc.squeeze(1), qb_enc.select(0))
    check_close(c0_enc.squeeze(1), cb_enc.select(0))
    check_close(t0_enc.squeeze(1), tb_enc.select(0))


if __name__ == '__main__':
    main()
