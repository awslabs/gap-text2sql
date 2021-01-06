import argparse
import json

import _jsonnet
import attr

from seq2struct.utils import evaluation

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--section', required=True)
    parser.add_argument('--inferred', required=True)
    parser.add_argument('--output')
    parser.add_argument('--logdir')
    args = parser.parse_args()
    return args

def main(args):
    real_logdir, metrics = evaluation.compute_metrics(args.config, args.config_args, args.section, args.inferred, args.logdir)

    if args.output:
        if real_logdir:
            output_path = args.output.replace('__LOGDIR__', real_logdir)
        else:
            output_path = args.output
        with open(output_path, 'w') as f:
            json.dump(metrics, f)
        print('Wrote eval results to {}'.format(output_path))
    else:
        print(metrics)


if __name__ == '__main__':
    args = add_parser()
    main(args)
