### Configure blobfuse
'''
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt-get update
sudo apt-get install blobfuse
sudo mkdir /mnt/blobfusetmp && sudo chown richard:richard /mnt/blobfusetmp
https://github.com/Azure/azure-storage-fuse/wiki/2.-Configuring-and-Running
echo -n "Key for Azure blob account 'wuphillyblob': "
read AZURE_KEY
cat <<EOF >connection.cfg
accountName wuphillyblob
accountKey $AZURE_KEY
containerName phillytools
EOF
sudo mkdir /mnt/wuphillyblob_blob; sudo chown richard:richard /mnt/wuphillyblob_blob
blobfuse /mnt/wuphillyblob_blob  --tmp-path=/mnt/blobfusetmp --config-file=connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 --log-level=LOG_WARNING --file-cache-timeout-in-seconds=120
'''
# Also mount the file share at wuphillyblob_share using commands from Azure

### Other setup (including downloading this repository)
'''
sudo apt install -y parallel unzip
git clone git@github.com:namisan/struct_embed
function gdrive_download () {
        COOKIES=$(mktemp)
        CONFIRM=$(wget --quiet --save-cookies ${COOKIES} --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
        wget --content-disposition --load-cookies ${COOKIES} "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1"
        rm -rf ${COOKIES}
        }
gdrive_download 11icoH_EA-NYb0OrPTdehRWm_d7-DIzWX
unzip spider.zip
cd struct_embed; git checkout dev/bawang/sp_align
# Setup virtualenv so that Python ≥3.6 is at `python`
pip install -e .
pip install pyyaml
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
bash data/spider-20190205/generate.sh ../spider
cp -R ../spider/database data/spider-20190205/
ln -s /mnt/wuphillyblob_blob/spider/bawang/* data/spider-20190205
'''

### Running on wikisql-1128v1-wu2-try0
'''
python experiments/spider-20190205/eval_20191128_wikisql.py --philly-yaml /mnt/wuphillyblob_share/spider/wikisql-1128v1-wu2-try0/philly.yaml --blob-root /mnt/wuphillyblob_blob --output /mnt/wuphillyblob_share/spider/wikisql-1128v1-wu2-try0/results > eval_20191128_wikisql.txt
cat eval_20191128_wikisql.txt | nice -n20 parallel --eta
'''



import argparse
import collections
import glob
import itertools
import json
import os
import sys

import _jsonnet
import attr
import yaml


CONFIG_PATH = 'configs/wikisql/nl2code-1128v1.jsonnet'


def create_exp_info(name, commands, data_dir):
    args_str = f"{{att: 0, data_path: '{data_dir}/wikisql/'}}"
    return {}, args_str


@attr.s
class Experiment:
    infer_output = attr.ib()
    infer_cmd = attr.ib()
    eval_output = attr.ib()
    eval_cmd = attr.ib()
    ckpt_path = attr.ib()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--philly-yaml', required=True)
    parser.add_argument('--blob-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--use-heuristic', action='store_true')

    # Limit output
    parser.add_argument('--snapshots')

    args = parser.parse_args()

    experiments = yaml.load(open(args.philly_yaml))
    data_dir = os.path.join(args.blob_root, experiments['data']['remote_dir'])
    os.makedirs(args.output, exist_ok=True)

    heuristic_postfix = '-heuristic' if args.use_heuristic else ''

    processed_experiments = collections.OrderedDict()
    for job in experiments['jobs']:
        exp_info_dict, args_str = create_exp_info(job['name'], job['command'], data_dir)
        config = json.loads(_jsonnet.evaluate_file(CONFIG_PATH, tla_codes={'args': args_str}))
        model_logdir = base_logdir = os.path.join(args.blob_root, job['results_dir'])

        job_output_root = os.path.join(args.output, job['id'])
        os.makedirs(job_output_root, exist_ok=True)
        #with open(os.path.join(job_output_root, 'exp_info.json'), 'w') as f:
        #    json.dump(exp_info_dict, f)

        # All these default args are here because of https://eev.ee/blog/2011/04/24/gotcha-python-scoping-closures/
        infer_output = lambda step, job_output_root=job_output_root: (
            f'{job_output_root}/infer-val-step{step:05d}-bs{args.beam_size}{heuristic_postfix}.jsonl'
        )
        infer_cmd = lambda step, base_logdir=base_logdir, args_str=args_str, infer_output=infer_output: (
            f'CACHE_DIR={data_dir} python infer.py '
            f'--config {CONFIG_PATH} '
            f'--logdir {base_logdir} '
            f'--config-args "{args_str}" ' 
            f'--output {infer_output(step)} ' 
            f'--step {step} --section val --beam-size {args.beam_size}'
            + (' --use-heuristic' if args.use_heuristic else '')
        )

        eval_output = lambda step, job_output_root=job_output_root: (
            f'{job_output_root}/eval-val-step{step:05d}-bs{args.beam_size}{heuristic_postfix}.jsonl'
        )
        eval_cmd = lambda step, args_str=args_str, infer_output=infer_output, eval_output=eval_output: (
            f'CACHE_DIR={data_dir} python eval.py '
            f'--config {CONFIG_PATH} '
            f'--config-args "{args_str}" ' 
            f'--inferred {infer_output(step)} '
            f'--output {eval_output(step)} ' 
            f'--section val'
        )

        ckpt_path = lambda step, model_logdir=model_logdir: os.path.join(model_logdir, f'model_checkpoint-{step:08d}')

        processed_experiments[job['id']] =  Experiment(
                infer_output, infer_cmd, eval_output, eval_cmd, ckpt_path)

    if args.snapshots:
        exp_id_step_pairs = json.loads(args.snapshots)
        for exp_id, step in exp_id_step_pairs:
            if exp_id not in processed_experiments:
                continue
            exp = processed_experiments[exp_id]
            if not os.path.exists(exp.ckpt_path(step)):
                continue
            infer_exists = os.path.exists(exp.infer_output(step))
            eval_exists = os.path.exists(exp.eval_output(step))
            if infer_exists and eval_exists:
                continue
            elif infer_exists:
                print(f'{exp.eval_cmd(step)}')
            else:
                print(f'{exp.infer_cmd(step)} && {exp.eval_cmd(step)}')
        return

    steps = [40000] + list(reversed(range(1100, 40000, 1000)))
    for step in steps:
        for exp in processed_experiments.values():
            if not os.path.exists(exp.ckpt_path(step)):
                continue
            infer_exists = os.path.exists(exp.infer_output(step))
            eval_exists = os.path.exists(exp.eval_output(step))
            if infer_exists and eval_exists:
                continue
            elif infer_exists:
                print(f'{exp.eval_cmd(step)}')
            else:
                print(f'{exp.infer_cmd(step)} && {exp.eval_cmd(step)}')


if __name__ == '__main__':
    main()

