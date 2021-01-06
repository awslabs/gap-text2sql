import argparse
import json
import os

import _jsonnet
import attr

from seq2struct import datasets
from seq2struct.utils import registry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--section', required=True)
    parser.add_argument('--inferred', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    os.makedirs(args.output, exist_ok=True)
    gold = open(os.path.join(args.output, 'gold.txt'), 'w')
    predicted = open(os.path.join(args.output, 'predicted.txt'), 'w')

    inferred = open(args.inferred)
    data = registry.construct('dataset', config['data'][args.section])

    for line in inferred:
        infer_results = json.loads(line)
        if infer_results['beams']:
            inferred_code = infer_results['beams'][0]['inferred_code']
        else:
            inferred_code = 'SELECT a FROM b'
        item = data[infer_results['index']]
        gold.write('{}\t{}\n'.format(item.orig['query'].replace('\t', ' '), item.schema.db_id))
        predicted.write('{}\n'.format(inferred_code))

if __name__ == '__main__':
    main()
