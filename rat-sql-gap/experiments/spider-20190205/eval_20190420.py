import itertools
import os
import sys


def main():
    all_commands = []
    all_eval_commands = []

    for att, enc_size, dec_size in itertools.product(
        (0, 1), (256, 512), (256, 512)):
        steps = list(range(1100, 40000, 1000)) + [40000]
        for step in steps:
            logdir = 'logdirs/20190420/output_from=false,enc_size=%(enc_size)d,dec_size=%(dec_size)d,att=%(att)d' % dict(
                att=att,
                enc_size=enc_size,
                dec_size=dec_size
            )

            infer_command = ((
                'python infer.py --config configs/spider-20190205/nl2code-0420.jsonnet '
                + '--config-args "{{output_from: false, ' +
                'enc_size: {enc_size}, dec_size: {dec_size}, att: {att}}}" ' + '--logdir logdirs/20190420 ' +
                '--output {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
                '--step {step} --section val --beam-size 1').format(
                    step=step,
                    att=att,
                    enc_size=enc_size,
                    dec_size=dec_size,
                    logdir=logdir))

            eval_command = ((
                'python eval.py --config configs/spider-20190205/nl2code-0420.jsonnet '
                + '--config-args "{{output_from: false, ' +
                'enc_size: {enc_size}, dec_size: {dec_size}, att: {att}}}" ' +
                '--inferred {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
                '--output {logdir}/eval-val-step{step:05d}-bs1.jsonl ' +
                '--section val').format(
                    step=step,
                    att=att,
                    enc_size=enc_size,
                    dec_size=dec_size,
                    logdir=logdir))

            print('{} && {}'.format(infer_command, eval_command))
            #print(eval_command)
            #print(infer_command)


if __name__ == '__main__':
    main()
