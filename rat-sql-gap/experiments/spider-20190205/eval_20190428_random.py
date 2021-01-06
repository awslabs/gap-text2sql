import itertools
import os
import sys


def main():
    all_commands = []
    all_eval_commands = []

    for att, fixed in itertools.product(
        (0, 1, 2, 3), (['init'], ['data', 'model'])):
        steps = list(range(1100, 40000, 1000)) + [40000]
        for step in steps:
            infer_command = ((
                'python infer.py --config configs/spider-20190205/nl2code-0428-random.jsonnet ' +
                '--logdir logdirs/20190428-random ' +
                '--config-args "{{fixed: {fixed}, att: {att}}}" ' +
                '--output __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl ' +
                '--step {step} --section val --beam-size 1').format(
                    step=step,
                    fixed=fixed,
                    att=att))

            eval_command = ((
                'python eval.py --config configs/spider-20190205/nl2code-0428-random.jsonnet ' +
                '--logdir logdirs/20190428-random ' +
                '--config-args "{{fixed: {fixed}, att: {att}}}" ' +
                '--inferred __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl ' +
                '--output __LOGDIR__/eval-val-step{step:05d}-bs1.jsonl ' +
                '--section val').format(
                    step=step,
                    fixed=fixed,
                    att=att))

            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
