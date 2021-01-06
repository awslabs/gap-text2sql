import itertools
import os
import sys


TEMPLATE = '''#!/bin/bash
#PBS -N {job_name}
#PBS -J 1-{num_jobs:d}

echo "Activating environment {env_name}"
source /export/vcl-nfs1-data2/shared/euichuls/miniconda3/bin/activate {env_name}

echo "PBS_ARRAY_INDEX: $PBS_ARRAY_INDEX"
echo "Hostname: $HOSTNAME"
source /export/vcl-nfs2/shared/common/jobs/gpu_select.sh
echo {job_name}

cd {base_dir}
'''

def main():
  for output_from, upd_steps in itertools.product(('true', 'false'), (0, 1, 2)):
    job_name = 'output_from={},upd_steps={}'.format(output_from, upd_steps)
    commands = []
    for qenc, ctenc, tinc, step in itertools.product(('e', 'eb'), ('e', 'eb', 'ebs'),
        ('true', 'false'), list(range(2100, 40000, 2000)) + [40000]):
      model_commands = []

      logdir = 'logdirs/20190214/output_from={output_from},qenc={qenc},ctenc={ctenc},tinc={tinc},upd_steps={upd_steps}'.format(
          output_from=output_from,
          qenc=qenc,
          ctenc=ctenc,
          tinc=tinc,
          upd_steps=upd_steps)

      model_commands.append(
        ('python infer.py --config configs/spider-20190205/nl2code-0214.jsonnet ' +
        '--config-args "{{output_from: {output_from}, ' +
        'qenc: \'{qenc}\', ctenc: \'{ctenc}\', ' +
        'tinc: {tinc}, upd_steps: {upd_steps}}}" ' +
        '--logdir logdirs/20190214 ' +
        '--output {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
        '--step {step} --section val --beam-size 1').format(
          step=step,
          output_from=output_from,
          qenc=qenc,
          ctenc=ctenc,
          tinc=tinc,
          upd_steps=upd_steps,
          logdir=logdir))
      model_commands.append(
        ('python eval.py --config configs/spider-20190205/nl2code-0214.jsonnet ' +
        '--config-args "{{output_from: {output_from}, ' +
        'qenc: \'{qenc}\', ctenc: \'{ctenc}\', ' +
        'tinc: {tinc}, upd_steps: {upd_steps}}}" ' +
        '--inferred {logdir}/infer-val-step{step:05d}-bs1.jsonl ' +
        '--output {logdir}/eval-val-step{step:05d}-bs1.jsonl ' +
        '--section val').format(
          step=step,
          output_from=output_from,
          qenc=qenc,
          ctenc=ctenc,
          tinc=tinc,
          upd_steps=upd_steps,
          logdir=logdir))

      commands.append(' && '.join(model_commands))

    f = open('experiments/PBS_20190214_eval_{}.sh'.format(job_name), 'w')
    f.write(TEMPLATE.format(
      job_name=job_name,
      env_name='seq2s',
      num_jobs=len(commands),
      base_dir=os.path.realpath(os.getcwd())))
    for i, cmd in enumerate(commands):
      f.write('''if [[ $PBS_ARRAY_INDEX == {i} ]]; then {cmd}; fi\n'''.format(i=i+1,
        cmd=cmd))
    f.close()

if __name__ == '__main__':
  main()

