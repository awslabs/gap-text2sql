import json
import os

import attr
import torch
import torchtext

from seq2struct.models import abstract_preproc
from seq2struct.models import variational_lstm
from seq2struct.models import spider_enc
from seq2struct.utils import registry
from seq2struct.utils import vocab
import collections


@registry.register('encoder', 'spider')
class SpiderEncoder(torch.nn.Module):

    batchd = False

    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                save_path,
                min_freq=3,
                max_count=5000):
            self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
            self.data_dir = os.path.join(save_path, 'enc')

            self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
            self.texts = collections.defaultdict(list)
            self.vocab = None

        def validate_item(self, item, section):
            return True, None
        
        def add_item(self, item, section, validation_info):
            if section == 'train':
                for token in item.text:
                    self.vocab_builder.add_word(token)
                for column in item.schema.columns:
                    for token in column.name:
                        self.vocab_builder.add_word(token)

            self.texts[section].append(self.preprocess_item(item, validation_info))

        def clear_items(self):
            self.texts = collections.defaultdict(list)

        def preprocess_item(self, item, validation_info):
            column_names = []

            last_table_id = None
            table_bounds = []

            for i, column in enumerate(item.schema.columns):
                table_name = ['all'] if column.table is None else column.table.name
                column_names.append([column.type] + table_name + column.name)

                table_id = None if column.table is None else column.table.id
                if last_table_id != table_id:
                    table_bounds.append(i)
                    last_table_id = table_id
            table_bounds.append(len(item.schema.columns))
            assert len(table_bounds) == len(item.schema.tables) + 1

            return {
                'question': item.text,
                'columns': column_names,
                'table_bounds': table_bounds,
            }

        def save(self):
            os.makedirs(self.data_dir, exist_ok=True)
            self.vocab = self.vocab_builder.finish()
            self.vocab.save(self.vocab_path)

            for section, texts in self.texts.items():
                with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                    for text in texts:
                        f.write(json.dumps(text) + '\n')

        def load(self):
            self.vocab = vocab.Vocab.load(self.vocab_path)

        def dataset(self, section):
            return [
                json.loads(line)
                for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

    def __init__(
            self,
            device,
            preproc: Preproc,
            word_emb_type='random',
            word_emb_size=128,
            recurrent_size=256,
            dropout=0.,
            table_enc='none'):
        super().__init__()
        self._device = device
        self.vocab = preproc.vocab

        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        assert self.recurrent_size % 2 == 0

        if word_emb_type == 'random':
            self.embedding = torch.nn.Embedding(
                num_embeddings=len(self.vocab),
                embedding_dim=self.word_emb_size)
            self._embed_words = self._embed_words_learned
        elif word_emb_type == 'glove.42B-fixed':
            cache = os.path.join(os.environ.get('CACHE_DIR', os.getcwd()), '.vector_cache')
            self.embedding = torchtext.vocab.GloVe(name='42B', cache=cache)
            assert word_emb_size == self.embedding.dim
            self._embed_words = self._embed_words_fixed

        self.question_encoder = variational_lstm.LSTM(
                input_size=self.word_emb_size,
                hidden_size=self.recurrent_size // 2,
                bidirectional=True,
                dropout=dropout)

        self.column_name_encoder = variational_lstm.LSTM(
                input_size=self.word_emb_size,
                hidden_size=self.recurrent_size // 2,
                bidirectional=True,
                dropout=dropout)

        if table_enc == 'none':
            self._table_enc = self._table_enc_none
        elif table_enc == 'mean_columns':
            self._table_enc = self._table_enc_mean_columns
        else:
            raise ValueError(table_enc)

        #self.column_set_encoder = lstm.LSTM(
        #        input_size=self.recurrent_size,
        #        hidden_size=self.recurrent_size // 2,
        #        bidirectional=True,
        #        dropout=dropout)

    def forward(self, desc):
        # emb shape: desc length x batch (=1) x word_emb_size
        question_emb = self._embed_words(desc['question'])

        # outputs shape: desc length x batch (=1) x recurrent_size
        # state shape:
        # - h: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
        # - c: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
        question_outputs, question_state = self.question_encoder(question_emb)

        # column_embs: list of batch (=1) x recurrent size
        column_embs = []
        for column in desc['columns']:
            column_name_embs = self._embed_words(column)

            # outputs shape: desc length x batch (=1) x recurrent_size
            # state shape:
            # - h: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            # - c: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            _, (h, c) = self.column_name_encoder(column_name_embs)
            column_embs.append(torch.cat((h[0], h[1]), dim=-1))

        #columns_outputs, columns_state = self.column_set_encoder(torch.stack(column_embs, dim=0))
        #columns_outputs = columns_outputs.transpose(0, 1)
        columns_outputs = torch.stack(column_embs, dim=1)

        return spider_enc.SpiderEncoderState(
            state=question_state,
            memory=question_outputs.transpose(0, 1),
            words=desc['question'],
            pointer_memories={
                'column': columns_outputs,
                'table': self._table_enc(desc, columns_outputs)},
            pointer_maps={})
    
    def _embed_words_learned(self, tokens):
        # token_indices shape: batch (=1) x length
        token_indices = torch.tensor(
                self.vocab.indices(tokens),
                device=self._device).unsqueeze(0)

        # emb shape: batch (=1) x length x word_emb_size
        emb = self.embedding(token_indices)

        # return value shape: desc length x batch (=1) x word_emb_size
        return emb.transpose(0, 1)

    def _embed_words_fixed(self, tokens):
        # return value shape: desc length x batch (=1) x word_emb_size
        return torch.stack(
                [self.embedding[token] for token in tokens],
                dim=0).unsqueeze(1).to(self._device)

    def _table_enc_mean_columns(self, desc, columns_outputs):
        # columns_outputs: batch (=1) x number of columns x recurrent size
        # TODO batching
        table_bounds = desc['table_bounds']
        return torch.stack(
           [columns_outputs[:, a:b].mean(dim=1)
              for a, b in zip(table_bounds, table_bounds[1:])],
           dim=1)

    def _table_enc_none(self, desc, column_embs):
        return None
