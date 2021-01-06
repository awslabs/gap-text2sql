import argparse
import collections
import glob
import itertools
import json
import os
import shlex
import sys

import _jsonnet
import attr
import yaml

@attr.s
class Experiment:
    infer_output = attr.ib()
    infer_cmd = attr.ib()
    eval_output = attr.ib()
    eval_cmd = attr.ib()
    ckpt_path = attr.ib()


def single_command(exp, step):
    infer_exists = os.path.exists(exp.infer_output(step))
    eval_exists = os.path.exists(exp.eval_output(step))
    if infer_exists and eval_exists:
        return None
    elif infer_exists:
        return f'{exp.eval_cmd(step)}'
    else:
        return f'{exp.infer_cmd(step)} && {exp.eval_cmd(step)}'

def generate_commands(
    create_exp_info_fn,
    config_path,
    steps,

    philly_yaml,
    blob_root,
    output_path,
    beam_size,
    use_heuristic,
    snapshots,
):
    experiments = yaml.load(open(philly_yaml))
    data_dir = os.path.join(blob_root, experiments['data']['remote_dir'])
    os.makedirs(output_path, exist_ok=True)

    heuristic_postfix = '-heuristic' if use_heuristic else ''

    processed_experiments = collections.OrderedDict()
    for job in experiments['jobs']:
        exp_info_dict, args_str = create_exp_info_fn(job['name'], job['command'], data_dir)
        config = json.loads(_jsonnet.evaluate_file(config_path, tla_codes={'args': args_str}))
        base_logdir = os.path.join(blob_root, job['results_dir'], 'logdirs')
        if 'model_name' in config:
            model_logdir = os.path.join(base_logdir, config['model_name'])
        else:
            model_logdir = base_logdir

        job_output_root = os.path.join(output_path, job['id'])
        os.makedirs(job_output_root, exist_ok=True)
        if exp_info_dict:
            with open(os.path.join(job_output_root, 'exp_info.json'), 'w') as f:
                json.dump(exp_info_dict, f)

        # All these default args are here because of https://eev.ee/blog/2011/04/24/gotcha-python-scoping-closures/
        infer_output = lambda step, job_output_root=job_output_root: (
            f'{job_output_root}/infer-val-step{step:05d}-bs{beam_size}{heuristic_postfix}.jsonl'
        )
        infer_cmd = lambda step, base_logdir=base_logdir, args_str=args_str, infer_output=infer_output: (
            f'CACHE_DIR={data_dir} python infer.py '
            f'--config {config_path} '
            f'--logdir {base_logdir} '
            f'--config-args {shlex.quote(args_str)} ' 
            f'--output {infer_output(step)} ' 
            f'--step {step} --section val --beam-size {beam_size}'
            + (' --use-heuristic' if use_heuristic else '')
        )

        eval_output = lambda step, job_output_root=job_output_root: (
            f'{job_output_root}/eval-val-step{step:05d}-bs{beam_size}{heuristic_postfix}.jsonl'
        )
        eval_cmd = lambda step, args_str=args_str, infer_output=infer_output, eval_output=eval_output: (
            f'CACHE_DIR={data_dir} python eval.py '
            f'--config {config_path} '
            f'--config-args {shlex.quote(args_str)} ' 
            f'--inferred {infer_output(step)} '
            f'--output {eval_output(step)} ' 
            f'--section val'
        )

        ckpt_path = lambda step, model_logdir=model_logdir: os.path.join(model_logdir, f'model_checkpoint-{step:08d}')

        processed_experiments[job['id']] =  Experiment(
                infer_output, infer_cmd, eval_output, eval_cmd, ckpt_path)

    if snapshots:
        exp_id_step_pairs = json.loads(snapshots)
        for exp_id, step in exp_id_step_pairs:
            if exp_id not in processed_experiments:
                continue
            command = single_command(processed_experiments[exp_id], step)
            if command is not None:
                yield command
        return

    for step in steps:
        for exp in processed_experiments.values():
            if not os.path.exists(exp.ckpt_path(step)):
                continue
            command = single_command(exp, step)
            if command is not None:
                yield command

def main(create_exp_info_fn, config_path, steps=[40000] + list(reversed(range(1100, 40000, 1000)))):
    parser = argparse.ArgumentParser()
    parser.add_argument('--philly-yaml', required=True)
    parser.add_argument('--blob-root', required=True)
    parser.add_argument('--output', required=True, dest='output_path')
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--use-heuristic', action='store_true')
    # Used to limit output
    parser.add_argument('--snapshots')
    args = parser.parse_args()

    for line in generate_commands(create_exp_info_fn, config_path, steps, **vars(args)):
        print(line)


