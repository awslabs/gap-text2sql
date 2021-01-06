import itertools
import os
import sys


TEMPLATE = '''#!/bin/bash
#PBS -N {job_name}
#PBS -m n

echo "Activating environment {env_name}"
source /export/vcl-nfs1-data2/shared/euichuls/miniconda3/bin/activate {env_name}

echo "PBS_ARRAY_INDEX: $PBS_ARRAY_INDEX"
echo "Hostname: $HOSTNAME"
source /export/vcl-nfs2/shared/common/jobs/gpu_select.sh
echo {job_name}

cd {base_dir}

'''

def main():
  all_commands = []
  all_eval_commands = []

  for output_from, max_steps, batch_size in itertools.product(
      ('true', 'false'), (40000, 80000), (10, 20)):
    if max_steps == 40000:
      steps = list(range(2100, 40000, 2000)) + [40000]
    elif max_steps == 80000:
      steps = list(range(2100, 80000, 2000)) + [80000]
    else:
      raise ValueError(max_steps)

    for upd_tied, upd_type in (
        ('false', 'no_subtypes'),
        ('false', 'merge_types'),
        ('true', 'full')):

      for step in steps:

        logdir = 'logdirs/20190302/output_from=%(output_from)s,upd_steps=%(upd_steps)d,upd_type=%(upd_type)s,upd_tied=%(upd_tied)s,max_steps=%(max_steps)d,batch_size=%(batch_size)d' % dict(
            output_from=output_from,
            upd_steps=4,
            upd_type=upd_type,
            upd_tied=upd_tied,
            max_steps=max_steps,
            batch_size=batch_size,
        )

        infer_command = (
          ('python infer.py --config configs/spider-20190205/nl2code-0302.jsonnet ' +
          '--config-args "{{output_from: {output_from}, ' +
          'upd_steps: 4, upd_type: \'{upd_type}\', upd_tied: {upd_tied}, max_steps: {max_steps}, ' +
          'batch_size: {batch_size}}}" ' +
          '--logdir logdirs/20190302 ' +
          '--output {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
          '--step {step} --section val --beam-size 1').format(
            step=step,
            output_from=output_from,
            upd_type=upd_type,
            upd_tied=upd_tied,
            max_steps=max_steps,
            batch_size=batch_size,
            logdir=logdir))

        eval_command = (
          ('python eval.py --config configs/spider-20190205/nl2code-0302.jsonnet ' +
          '--config-args "{{output_from: {output_from}, ' +
          'upd_steps: 4, upd_type: \'{upd_type}\', upd_tied: {upd_tied}, max_steps: {max_steps}, ' +
          'batch_size: {batch_size}}}" ' +
          '--inferred {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
          '--output {logdir}/eval-val-step{step:05d}-bs1.jsonl ' +
          '--section val').format(
            step=step,
            output_from=output_from,
            upd_type=upd_type,
            upd_tied=upd_tied,
            max_steps=max_steps,
            batch_size=batch_size,
            logdir=logdir))

        print('{} && {}'.format(infer_command, eval_command))
        #print(eval_command)


if __name__ == '__main__':
  main()

