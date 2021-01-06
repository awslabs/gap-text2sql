import itertools
import json
import os
import sys

import _jsonnet

def main():
    for att in (0, 1):
        steps = list(range(100, 2600, 100))
        logdir = os.path.join(
                'logdirs/20181231-nl2code-hearthstone-fef2c5b',
                'att{}'.format(att))

        for step in steps:
            if not os.path.exists(os.path.join(
                logdir,
                'model_checkpoint-{:08d}'.format(step))):
                continue

            if os.path.exists(os.path.join(
                logdir,
                'infer-val-step{:05d}-bs1.jsonl'.format(step))):
                continue

            infer_command = ((
                'python infer.py '
                '--config configs/hearthstone/nl2code.jsonnet '
                '--logdir {logdir} '
                '--output __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl ' +
                '--step {step} --section val --beam-size 1').format(
                    logdir=logdir,
                    step=step,
                    ))

            #eval_command = ((
            #    'python eval.py --config configs/spider-20190205/nl2code-0521-ablations.jsonnet ' +
            #    '--logdir logdirs/20190521-ablations ' +
            #    '--config-args "{args}" ' +
            #    '--inferred __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl ' +
            #    '--output __LOGDIR__/eval-val-step{step:05d}-bs1.jsonl ' +
            #    '--section val').format(
            #        args=args,
            #        step=step,
            #        ))

            #print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            print(infer_command)


if __name__ == '__main__':
    main()
