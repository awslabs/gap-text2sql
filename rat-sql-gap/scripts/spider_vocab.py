import argparse
import collections
import functools
import itertools

import bpemb
import corenlp
import torchtext
import tqdm

from seq2struct.datasets import spider
from seq2struct.models import pretrained_embeddings


#class Tokenizer:
#    def __init__(self):
#        self.client = corenlp.CoreNLPClient(annotators="tokenize ssplit")
#
#    @functools.lru_cache(maxsize=1024)
#    def tokenize(self, text):
#        ann = self.client.annotate(text)
#        return [tok.word for sent in ann.sentence for tok in sent.token]


def count_glove(data, embedder):
    present = collections.Counter()
    missing = collections.Counter()
    counted_db_ids = set()

    for item in tqdm.tqdm(data):
        question_tokens = embedder.tokenize(item.orig['question'])
        for token in question_tokens:
            if embedder.lookup(token) is None:
                missing[token] += 1
            else:
                present[token] += 1

        if item.orig['db_id'] in counted_db_ids:
            continue
        column_names = [
            embedder.tokenize(column.unsplit_name) for column in item.schema.columns
        ]
        table_names = [
            embedder.tokenize(table.unsplit_name) for table in item.schema.tables
        ]

        for token in itertools.chain(*column_names, *table_names):
            if embedder.lookup(token) is None:
                missing[token] += 1
            else:
                present[token] += 1
        counted_db_ids.add(item.orig['db_id'])
    return present, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='glove')
    parser.add_argument('--name')
    args = parser.parse_args()

    args = parser.parse_args()
    train = spider.SpiderDataset(
        paths=(
            'data/spider-20190205/train_spider.json',
            # 'data/spider-20190205/train_others.json',
        ),
        tables_paths=(
            'data/spider-20190205/tables.json',
        ),
        db_path='data/spider-20190205/database')
    
    dev = spider.SpiderDataset(
        paths=(
            'data/spider-20190205/dev.json',
        ),
        tables_paths=(
            'data/spider-20190205/tables.json',
        ),
        db_path='data/spider-20190205/database')

    if args.mode == 'glove':
        embedder = pretrained_embeddings.GloVe(args.name or '42B')
    elif args.mode == 'bpemb':
        embedder = pretrained_embeddings.BPEmb(dim=100, vocab_size=int(args.name or 10000))
    t_g_present, t_g_missing = count_glove(train, embedder)
    d_g_present, d_g_missing = count_glove(dev, embedder)

    import IPython
    IPython.embed()

if __name__ == '__main__':
    main()
