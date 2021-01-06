import argparse
import itertools
import json
import os
import sys

import _jsonnet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam-size', type=int, default=1)
    script_args = parser.parse_args()

    for (glove, upd_type, num_layers), att in itertools.product((
        (False, 'full', 4),
        (True, 'no_subtypes', 4),
        (True, 'merge_types', 4),
        (True, 'full', 2),
        (True, 'full', 0),
    ), (0, 1, 2)):
        steps = list(range(1100, 40000, 1000)) + [40000]
        args = '{{glove: {glove}, upd_type: \'{upd_type}\', num_layers: {num_layers}, att: {att}}}'.format(
                glove='true' if glove else 'false',
                upd_type=upd_type,
                num_layers=num_layers,
                att=att)
        config = json.loads(
            _jsonnet.evaluate_file(
                'configs/spider-20190205/nl2code-0521-ablations.jsonnet', tla_codes={'args': args}))
        logdir = os.path.join(
                'logdirs/20190521-ablations',
                config['model_name'])

        for step in steps:
            if not os.path.exists(os.path.join(
                logdir,
                'model_checkpoint-{:08d}'.format(step))):
                continue

            if os.path.exists(os.path.join(
                logdir,
                'eval-val-step{:05d}-bs{}.jsonl'.format(step, script_args.beam_size))):
                continue

            infer_command = ((
                'python infer.py '
                '--config configs/spider-20190205/nl2code-0521-ablations.jsonnet '
                '--logdir logdirs/20190521-ablations '
                '--config-args "{args}" ' +
                '--output __LOGDIR__/infer-val-step{step:05d}-bs{beam_size}.jsonl ' +
                '--step {step} --section val --beam-size {beam_size}').format(
                    args=args,
                    step=step,
                    beam_size=script_args.beam_size,
                    ))

            eval_command = ((
                'python eval.py --config configs/spider-20190205/nl2code-0521-ablations.jsonnet ' +
                '--logdir logdirs/20190521-ablations ' +
                '--config-args "{args}" ' +
                '--inferred __LOGDIR__/infer-val-step{step:05d}-bs{beam_size}.jsonl ' +
                '--output __LOGDIR__/eval-val-step{step:05d}-bs{beam_size}.jsonl ' +
                '--section val').format(
                    args=args,
                    step=step,
                    beam_size=script_args.beam_size,
                    ))

            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
