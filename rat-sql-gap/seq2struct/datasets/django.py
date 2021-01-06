import ast
import collections
import itertools
import json

import attr
import astor
import torch.utils.data

from seq2struct.utils import registry


@attr.s
class DjangoItem:
    text = attr.ib()
    code = attr.ib()
    str_map = attr.ib()


@registry.register('dataset', 'django')
class DjangoDataset(torch.utils.data.Dataset): 
    def __init__(self, path, limit=None):
        self.path = path
        self.examples = []
        for line in itertools.islice(open(self.path), limit):
            example = json.loads(line)
            self.examples.append(DjangoItem(
                text=example['text']['tokens'],
                code=example['orig'],
                str_map=example['text']['str_map']))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @attr.s
    class Metrics:
        dataset = attr.ib()
        exact_match = attr.ib(factory=list)

        def add(self, item, inferred_code, obsolete_gold_code=None):
            if obsolete_gold_code is None:
                try:
                    gold_code = astor.to_source(ast.parse(item.code))
                except ParseError:
                    return
            else:
                gold_code = obsolete_gold_code

            # Both of these should be canonicalized
            exact_match = gold_code == inferred_code

            self.exact_match.append(exact_match)

        def finalize(self):
            return collections.OrderedDict((
                ('exact match', sum(self.exact_match) / len(self.exact_match)),
            ))
