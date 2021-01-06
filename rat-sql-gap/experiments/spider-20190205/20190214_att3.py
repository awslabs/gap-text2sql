import itertools
import os
import sys


TEMPLATE = '''
#!/bin/bash
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
  job_name = 'spider_att3'
  commands = []
  for output_from, upd_steps in itertools.product(('true', 'false'), (0, 1, 2)):
    for qenc, ctenc, tinc in itertools.product(('e', 'eb'), ('e', 'eb', 'ebs'),
        ('true', 'false')):
      if (output_from, qenc, ctenc, tinc, upd_steps) not in (
          ('false', 'eb', 'eb', 'true', 2),
          ('true', 'eb', 'ebs', 'false', 0),
          ('true', 'eb', 'ebs', 'false', 2),
          ('true', 'eb', 'ebs', 'true', 2),
          ('true', 'e', 'e', 'true', 1)):
        continue
      commands.append('python train.py --config configs/spider-20190205/nl2code-0214.jsonnet --config-args "{{output_from: {}, qenc: \'{}\', ctenc: \'{}\', tinc: {}, upd_steps: {}}}" --logdir logdirs/20190214'.format(
        output_from, qenc, ctenc, tinc, upd_steps))

  f = open('experiments/PBS_20190214_{}.sh'.format(job_name), 'w')
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

