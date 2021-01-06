import itertools
import json

import attr
import torch

from seq2struct.utils import registry


@attr.s
class IdiomAstItem:
    text = attr.ib()
    code = attr.ib()
    orig = attr.ib()
    # TODO: Don't hardcode str_map here, which is only used for Django
    str_map = attr.ib()


@registry.register('dataset', 'idiom_ast')
class IdiomAstDataset(torch.utils.data.Dataset): 
    def __init__(self, path, limit=None):
        self.path = path
        self.examples = []
        for line in itertools.islice(open(self.path), limit):
            example = json.loads(line)
            if isinstance(example['text'], dict):
                self.examples.append(IdiomAstItem(
                    text=example['text']['tokens'],
                    code=example['rewritten_ast'],
                    orig=example,
                    str_map=example['text']['str_map']))
            else:
                self.examples.append(IdiomAstItem(
                    text=example['text'],
                    code=example['rewritten_ast'],
                    orig=example,
                    str_map=None))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
