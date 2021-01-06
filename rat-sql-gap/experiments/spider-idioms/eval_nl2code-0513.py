import itertools
import os
import sys


def main():
    all_commands = []
    all_eval_commands = []

    for st, nt, att in itertools.product(
        ('cov-xent', 'cov-examples'),
        (10, 20, 40, 80),
        (0,),
    ):
        steps = list(range(1100, 40000, 1000)) + [40000]
        for step in steps:
            infer_command = ((
                'python infer.py --config configs/spider-idioms/nl2code-0513.jsonnet '
                '--logdir logdirs/spider-idioms/nl2code-0513 '
                '--config-args "{{st: \'{st}\', nt: {nt}, att: {att}}}" '
                '--output __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl '
                '--step {step} --section val --beam-size 1').format(
                    step=step,
                    st=st,
                    nt=nt,
                    att=att,
                    ))

            eval_command = ((
                'python eval.py --config configs/spider-idioms/nl2code-0513.jsonnet ' 
                '--logdir logdirs/spider-idioms/nl2code-0513 ' 
                '--config-args "{{st: \'{st}\', nt: {nt}, att: {att}}}" ' 
                '--inferred __LOGDIR__/infer-val-step{step:05d}-bs1.jsonl ' 
                '--output __LOGDIR__/eval-val-step{step:05d}-bs1.jsonl ' 
                '--section val').format(
                    step=step,
                    st=st,
                    nt=nt,
                    att=att,
                    ))
            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
