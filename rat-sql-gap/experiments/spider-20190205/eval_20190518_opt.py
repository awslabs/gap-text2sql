import itertools
import os
import sys


def main():
    all_commands = []
    all_eval_commands = []

    for lr, wd, decay, att in itertools.product(
        (1e-3, 1e-4),
        (0, 1e-2),
        ('cosine', 'linear'),
        (0, 1),
    ):
        steps = list(range(1100, 20000, 500)) + [20000]
        for step in steps:
            infer_command = ((
                'python infer.py --config configs/spider-20190205/nl2code-0518-opt.jsonnet ' +
                '--logdir logdirs/20190518-opt ' +
                '--config-args "{{lr: {lr}, wd: {wd}, decay: \'{decay}\', att: {att}}}" ' +
                '--output __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl '
                '--step {step} --section val --beam-size 1').format(
                    step=step,
                    lr=lr,
                    wd=wd,
                    decay=decay,
                    att=att,
                    ))

            eval_command = ((
                'python eval.py --config configs/spider-20190205/nl2code-0518-opt.jsonnet '
                '--logdir logdirs/20190518-opt '
                '--config-args "{{lr: {lr}, wd: {wd}, decay: \'{decay}\', att: {att}}}" '
                '--inferred __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl '
                '--output __LOGDIR__/eval-val-step{step:05d}-bs1.jsonl '
                '--section val').format(
                    step=step,
                    lr=lr,
                    wd=wd,
                    decay=decay,
                    att=att,
                    ))

            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
