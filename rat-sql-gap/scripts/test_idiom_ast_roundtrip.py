import ast
import argparse
import json
import pprint

import astor
import tqdm
import _jsonnet

from seq2struct import datasets
from seq2struct import grammars

from seq2struct.utils import registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    train_data = registry.construct('dataset', config['data']['train'])
    grammar = registry.construct('grammar', config['model']['decoder_preproc']['grammar'])
    base_grammar = registry.construct('grammar', config['model']['decoder_preproc']['grammar']['base_grammar'])

    for i, item in enumerate(tqdm.tqdm(train_data, dynamic_ncols=True)):
        parsed = grammar.parse(item.code, 'train')
        orig_parsed = base_grammar.parse(item.orig['orig'], 'train')

        canonicalized_orig_code = base_grammar.unparse(
            base_grammar.parse(item.orig['orig'], 'train'), item)
        unparsed = grammar.unparse(parsed, item)
        if canonicalized_orig_code != unparsed:
            print('Original tree:')
            pprint.pprint(orig_parsed)
            print('Rewritten tree:')
            pprint.pprint(parsed)
            print('Reconstructed tree:')
            pprint.pprint(grammar._expand_templates(parsed))
            print('Original code:')
            print(canonicalized_orig_code)
            print('Reconstructed code:')
            print(unparsed)
            
            import IPython; IPython.embed()
            break


if __name__ == '__main__':
    main()
