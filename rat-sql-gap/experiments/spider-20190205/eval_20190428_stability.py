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

    for (bs, lr, end_lr), att in itertools.product((
        (50, 1e-3, 0),
        #(100, 1e-3, 0),
        #(10, 5e-4, 0),
        #(10, 2.5e-4, 0),
        #(10, 1e-3, 5e-4),
        #(10, 1e-3, 2.5e-4),
    ), (0, 1, 2)):
        steps = list(range(1100, 40000, 1000)) + [40000]
        args = "{{bs: {bs}, lr: {lr}, end_lr: {end_lr}, att: {att}}}".format(
                    bs=bs,
                    lr=lr,
                    end_lr=end_lr,
                    att=att,
                    )
        config = json.loads(
            _jsonnet.evaluate_file(
                'configs/spider-20190205/nl2code-0428-stability.jsonnet', tla_codes={'args': args}))
        logdir = os.path.join(
                'logdirs/20190428-stability',
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
                'python infer.py --config configs/spider-20190205/nl2code-0428-stability.jsonnet '
                '--logdir logdirs/20190428-stability '
                '--config-args "{args}" '
                '--output __LOGDIR__/infer-val-step{step:05d}-bs{beam_size}.jsonl ' +
                '--step {step} --section val --beam-size {beam_size}').format(
                    step=step,
                    args=args,
                    beam_size=script_args.beam_size,
                    ))

            eval_command = ((
                'python eval.py --config configs/spider-20190205/nl2code-0428-stability.jsonnet '
                '--logdir logdirs/20190428-stability '
                '--config-args "{args}" '
                '--inferred __LOGDIR__/infer-val-step{step:05d}-bs{beam_size}.jsonl ' +
                '--output __LOGDIR__/eval-val-step{step:05d}-bs{beam_size}.jsonl ' +
                '--section val').format(
                    step=step,
                    args=args,
                    beam_size=script_args.beam_size,
                    ))

            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
