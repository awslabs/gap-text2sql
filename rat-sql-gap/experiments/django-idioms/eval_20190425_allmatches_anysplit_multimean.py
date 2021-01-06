import itertools
import os
import sys


def main():
    all_commands = []
    all_eval_commands = []

    for filt, st, nt in itertools.product(
        ('none', 'contains-hole'),
        ('cov-xent', 'cov-examples'),
        (10, 20, 40, 80),
    ):
        steps = list(range(1100, 40000, 1000)) + [40000]
        logdir = (
            'logdirs/20190425-django-allmatches-anysplit-multimean'
            '/filt-{filt}_st-{st}_nt-{nt}').format(filt=filt, st=st, nt=nt)

        for step in steps:
            infer_command = (
                'python infer.py --config configs/django-idioms/nl2code-0425-allmatches-anysplit-multimean.jsonnet '
                '--logdir {logdir} '
                '--config-args "{{filt: \'{filt}\', st: \'{st}\', nt: {nt}}}" '
                '--output {logdir}/infer-val-step{step:05d}-bs1.jsonl '
                '--step {step} --section val --beam-size 1').format(
                    logdir=logdir,
                    step=step,
                    filt=filt,
                    st=st,
                    nt=nt,
                    )
            print(infer_command)


if __name__ == '__main__':
    main()
