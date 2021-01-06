import itertools
import os
import sys


TEMPLATE = '''#!/bin/bash
#PBS -N {job_name}
#PBS -J 1-{num_jobs:d}
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
  for output_from, upd_steps in itertools.product(('true', 'false'), (2, 3, 4, 5, 6)):
    job_name = 'output_from={},upd_steps={}'.format(output_from, upd_steps)
    commands = []
    for (qenc, ctenc), max_steps, batch_size in itertools.product(
        (('e', 'e'), ('eb', 'ebs')),
        ('40000', '80000'),
        ('10', '20')):
      commands.append(
        ('python train.py --config configs/spider-20190205/nl2code-0220.jsonnet '
        + '--config-args "{{output_from: {}, qenc: \'{}\', ctenc: \'{}\', '
        + 'upd_steps: {}, max_steps: {}, batch_size: {}}}" --logdir logdirs/20190223').format(
        output_from, qenc, ctenc, upd_steps, max_steps, batch_size))

    f = open('experiments/PBS_20190223_nobatch_retry_{}.sh'.format(job_name), 'w')
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

