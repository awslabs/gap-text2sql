import itertools
import os
import sys


def main():
    all_commands = []
    all_eval_commands = []

    for output_from, emb, min_freq in itertools.product(
        ('false', 'true'), ('glove-42B', 'bpemb-10k', 'bpemb-100k'), (3, 50)):
        if min_freq == 50 and emb != 'glove-42B':
            continue
        steps = list(range(1100, 40000, 1000)) + [40000]
        for step in steps:
            logdir = 'logdirs/20190401/output_from=%(output_from)s,emb=%(emb)s,min_freq=%(min_freq)d,att=0' % dict(
                output_from=output_from,
                emb=emb,
                min_freq=min_freq,
            )

            infer_command = ((
                'python infer.py --config configs/spider-20190205/nl2code-0401.jsonnet '
                + '--config-args "{{output_from: {output_from}, ' +
                'emb: \'{emb}\', min_freq: {min_freq}, att: 0}}" ' + '--logdir logdirs/20190401 ' +
                '--output {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
                '--step {step} --section val --beam-size 1').format(
                    step=step, output_from=output_from, emb=emb, min_freq=min_freq,
                    logdir=logdir))

            eval_command = ((
                'python eval.py --config configs/spider-20190205/nl2code-0401.jsonnet '
                + '--config-args "{{output_from: {output_from}, ' +
                'emb: \'{emb}\', min_freq: {min_freq}, att: 0}}" ' + 
                '--inferred {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
                '--output {logdir}/eval-val-step{step:05d}-bs1.jsonl ' +
                '--section val').format(
                    step=step, output_from=output_from, emb=emb, min_freq=min_freq,
                    logdir=logdir))

            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
