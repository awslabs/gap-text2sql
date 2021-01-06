import itertools
import os
import sys

def main():
  all_commands = []
  all_eval_commands = []

  for output_from, att in itertools.product(
      ('false', 'true'), (0, 1, 2, 3)):
    steps = list(range(1100, 40000, 1000)) + [40000]
    for step in steps:
      logdir = 'logdirs/20190327/rerun,output_from=%(output_from)s,att=%(att)d' % dict(
          output_from=output_from,
          att=att,
      )

      infer_command = (
        ('python infer.py --config configs/spider-20190205/nl2code-0327.jsonnet ' +
        '--config-args "{{output_from: {output_from}, ' +
        'att: {att}}}" ' +
        '--logdir logdirs/20190327 ' +
        '--output {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
        '--step {step} --section val --beam-size 1').format(
          step=step,
          output_from=output_from,
          att=att,
          logdir=logdir))

      eval_command = (
        ('python eval.py --config configs/spider-20190205/nl2code-0327.jsonnet ' +
        '--config-args "{{output_from: {output_from}, ' +
        'att: {att}}}" ' +
        '--inferred {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
        '--output {logdir}/eval-val-step{step:05d}-bs1.jsonl ' +
        '--section val').format(
          step=step,
          output_from=output_from,
          att=att,
          logdir=logdir))

      print('{} && {}'.format(infer_command, eval_command))
      #print(eval_command)
      #print(infer_command)


if __name__ == '__main__':
  main()

