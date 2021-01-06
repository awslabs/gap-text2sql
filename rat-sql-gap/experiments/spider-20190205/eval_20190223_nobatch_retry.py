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

  for output_from, upd_steps in itertools.product(('true', 'false'), (2, 3, 4, 5, 6)):
    job_name = 'e0223,{},{}'.format(output_from, upd_steps)
    commands = []

    total = 2 * 2 * (20 + 40)
    i = 0

    for (qenc, ctenc), max_steps, batch_size in itertools.product(
        (('e', 'e'), ('eb', 'ebs')),
        ('40000', '80000'),
        ('10', '20')):
      if max_steps == '40000':
          steps = list(range(2100, 40000, 2000)) + [40000]
      elif max_steps == '80000':
          steps = list(range(2100, 80000, 2000)) + [80000]
      for step in steps:
          logdir = 'logdirs/20190223/output_from={output_from},qenc={qenc},ctenc={ctenc},upd_steps={upd_steps},max_steps={max_steps},batch_size={batch_size}'.format(
              output_from=output_from,
              qenc=qenc,
              ctenc=ctenc,
              upd_steps=upd_steps,
              max_steps=max_steps,
              batch_size=batch_size)

          model_commands = []
          model_commands.append(
            ('python infer.py --config configs/spider-20190205/nl2code-0220.jsonnet ' +
            '--config-args "{{output_from: {output_from}, ' +
            'qenc: \'{qenc}\', ctenc: \'{ctenc}\', ' +
            'upd_steps: {upd_steps}, max_steps: {max_steps}, ' +
            'batch_size: {batch_size}}}" ' +
            '--logdir logdirs/20190223 ' +
            '--output {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
            '--step {step} --section val --beam-size 1').format(
              step=step,
              output_from=output_from,
              qenc=qenc,
              ctenc=ctenc,
              upd_steps=upd_steps,
              max_steps=max_steps,
              batch_size=batch_size,
              logdir=logdir))

          eval_command = (
            ('python eval.py --config configs/spider-20190205/nl2code-0220.jsonnet ' +
            '--config-args "{{output_from: {output_from}, ' +
            'qenc: \'{qenc}\', ctenc: \'{ctenc}\', ' +
            'upd_steps: {upd_steps}, max_steps: {max_steps}, ' +
            'batch_size: {batch_size}}}" ' +
            '--inferred {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
            '--output {logdir}/eval-val-step{step:05d}-bs1.jsonl ' +
            '--section val').format(
              step=step,
              output_from=output_from,
              qenc=qenc,
              ctenc=ctenc,
              upd_steps=upd_steps,
              max_steps=max_steps,
              batch_size=batch_size,
              logdir=logdir))
          model_commands.append(eval_command)
          all_eval_commands.append(eval_command)

          commands.append(' && '.join(model_commands))
          commands.append('echo Finished {}/{}'.format(i, total))
          all_commands.append(' && '.join(model_commands))
          i += 1

    f = open('experiments/PBS_eval_20190223_nobatch_retry_{}.sh'.format(job_name), 'w')
    f.write(TEMPLATE.format(
      job_name=job_name,
      env_name='seq2s',
      base_dir=os.path.realpath(os.getcwd())))
    for i, cmd in enumerate(commands):
      f.write('{cmd}\n'.format(cmd=cmd))
    f.close()

    with open('experiments/eval_20190223_nobatch_retry_all_eval_commands.txt',
        'w') as f:
      f.write('\n'.join(all_eval_commands))
    with open('experiments/eval_20190223_nobatch_retry_all_commands.txt',
        'w') as f:
      f.write('\n'.join(all_commands) + '\n')
      

if __name__ == '__main__':
  main()

