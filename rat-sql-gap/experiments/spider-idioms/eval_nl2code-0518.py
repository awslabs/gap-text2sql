import itertools
import json
import os
import sys

import _jsonnet

def main():
    for st, nt, att in itertools.product(
        ('cov-xent', 'cov-examples'),
        (40, 80),
        (0,),
    ):
        steps = list(range(1100, 40000, 1000)) + [40000]
        args = '{{st: \'{st}\', nt: {nt}, att: {att}}}'.format(
                st=st,
                nt=nt,
                att=att)

        config = json.loads(
            _jsonnet.evaluate_file(
                'configs/spider-idioms/nl2code-0518.jsonnet', tla_codes={'args': args}))
        logdir = os.path.join(
                'logdirs/spider-idioms/nl2code-0518',
                config['model_name'])

        for step in steps:
            if not os.path.exists(os.path.join(
                logdir,
                'model_checkpoint-{:08d}'.format(step))):
                continue

            if os.path.exists(os.path.join(
                logdir,
                'eval-val-step{:05d}-bs1.jsonl'.format(step))):
                continue

            infer_command = ((
                'python infer.py --config configs/spider-idioms/nl2code-0518.jsonnet '
                '--logdir logdirs/spider-idioms/nl2code-0518 '
                '--config-args "{args}" '
                '--output __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl '
                '--step {step} --section val --beam-size 1').format(
                    args=args,
                    step=step,
                    ))

            eval_command = ((
                'python eval.py --config configs/spider-idioms/nl2code-0518.jsonnet ' 
                '--logdir logdirs/spider-idioms/nl2code-0518 ' 
                '--config-args "{args}" ' 
                '--inferred __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl ' 
                '--output __LOGDIR__/eval-val-step{step:05d}-bs1.jsonl ' 
                '--section val').format(
                    args=args,
                    step=step,
                    ))
            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
