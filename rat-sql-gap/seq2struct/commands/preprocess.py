import argparse
import json
import os

import _jsonnet
import tqdm

from seq2struct import datasets
from seq2struct import models
from seq2struct.utils import registry
from seq2struct.utils import vocab

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])

    def preprocess(self):
        self.model_preproc.clear_items()
        for section in self.config['data']:
            data = registry.construct('dataset', self.config['data'][section])
            for item in tqdm.tqdm(data, desc=section, dynamic_ncols=True):
                to_add, validation_info = self.model_preproc.validate_item(item, section)
                if to_add:
                    self.model_preproc.add_item(item, section, validation_info)
        self.model_preproc.save()

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args

def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    preprocessor = Preprocessor(config)
    preprocessor.preprocess()

if __name__ == '__main__':
    args = add_parser()
    main(args)
