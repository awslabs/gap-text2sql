import ast
import argparse
import json
import os
import pprint

import astor
import tqdm
import _jsonnet

from seq2struct import datasets
from seq2struct import grammars

from seq2struct.utils import registry
from third_party.spider import evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    os.makedirs(args.output, exist_ok=True)
    gold = open(os.path.join(args.output, 'gold.txt'), 'w')
    predicted = open(os.path.join(args.output, 'predicted.txt'), 'w')

    train_data = registry.construct('dataset', config['data']['train'])
    grammar = registry.construct('grammar', config['model']['decoder_preproc']['grammar'])

    evaluator = evaluation.Evaluator(
            'data/spider-20190205/database',
            evaluation.build_foreign_key_map_from_json('data/spider-20190205/tables.json'),
            'match')

    for i, item in enumerate(tqdm.tqdm(train_data, dynamic_ncols=True)):
        parsed = grammar.parse(item.code, 'train')
        sql = grammar.unparse(parsed, item)

        evaluator.evaluate_one(
                item.schema.db_id,
                item.orig['query'].replace('\t', ' '),
                sql)

        gold.write('{}\t{}\n'.format(item.orig['query'].replace('\t', ' '), item.schema.db_id))
        predicted.write('{}\n'.format(sql))

if __name__ == '__main__':
    main()
