import json

import attr
import torch
import networkx as nx

from seq2struct.utils import registry
from seq2struct.datasets.spider_lib import evaluation


@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()


def load_tables(paths):
    schemas = {}
    eval_foreign_key_maps = {}

    for path in paths:
        schema_dicts  = json.load(open(path))
        for schema_dict in schema_dicts:
            tables = tuple(
                Table(
                    id=i,
                    name=name.split(),
                    unsplit_name=name,
                    orig_name=orig_name,
                )
                for i, (name, orig_name) in enumerate(zip(
                    schema_dict['table_names'], schema_dict['table_names_original']))
            )
            columns = tuple(
                Column(
                    id=i,
                    table=tables[table_id] if table_id >= 0 else None,
                    name=col_name.split(),
                    unsplit_name=col_name,
                    orig_name=orig_col_name,
                    type=col_type,
                )
                for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                    schema_dict['column_names'],
                    schema_dict['column_names_original'],
                    schema_dict['column_types']))
            )

            # Link columns to tables
            for column in columns:
                if column.table:
                    column.table.columns.append(column)

            for column_id in schema_dict['primary_keys']:
                # Register primary keys
                column = columns[column_id]
                column.table.primary_keys.append(column)

            foreign_key_graph = nx.DiGraph()
            for source_column_id, dest_column_id in schema_dict['foreign_keys']:
                # Register foreign keys
                source_column = columns[source_column_id]
                dest_column = columns[dest_column_id]
                source_column.foreign_key_for = dest_column
                foreign_key_graph.add_edge(
                    source_column.table.id,
                    dest_column.table.id,
                    columns=(source_column_id, dest_column_id))
                foreign_key_graph.add_edge(
                    dest_column.table.id,
                    source_column.table.id,
                    columns=(dest_column_id, source_column_id))

            db_id = schema_dict['db_id']
            assert db_id not in schemas
            schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
            eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps

def load_tables_from_schema_dict(schema_dict):
    schemas = {}
    eval_foreign_key_maps = {}
    schema_dicts = [schema_dict]

    for schema_dict in schema_dicts:
        tables = tuple(
            Table(
                id=i,
                name=name.split(),
                unsplit_name=name,
                orig_name=orig_name,
            )
            for i, (name, orig_name) in enumerate(zip(
                schema_dict['table_names'], schema_dict['table_names_original']))
        )
        columns = tuple(
            Column(
                id=i,
                table=tables[table_id] if table_id >= 0 else None,
                name=col_name.split(),
                unsplit_name=col_name,
                orig_name=orig_col_name,
                type=col_type,
            )
            for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                schema_dict['column_names'],
                schema_dict['column_names_original'],
                schema_dict['column_types']))
        )

        # Link columns to tables
        for column in columns:
            if column.table:
                column.table.columns.append(column)

        for column_id in schema_dict['primary_keys']:
            # Register primary keys
            column = columns[column_id]
            column.table.primary_keys.append(column)

        foreign_key_graph = nx.DiGraph()
        for source_column_id, dest_column_id in schema_dict['foreign_keys']:
            # Register foreign keys
            source_column = columns[source_column_id]
            dest_column = columns[dest_column_id]
            source_column.foreign_key_for = dest_column
            foreign_key_graph.add_edge(
                source_column.table.id,
                dest_column.table.id,
                columns=(source_column_id, dest_column_id))
            foreign_key_graph.add_edge(
                dest_column.table.id,
                source_column.table.id,
                columns=(dest_column_id, source_column_id))

        db_id = schema_dict['db_id']
        assert db_id not in schemas
        schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
        eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps

@registry.register('dataset_infer', 'spider')
class SpiderDatasetInfer(torch.utils.data.Dataset):
    def __init__(self, schemas, eval_foreign_key_maps, db_path):
        self.db_path = db_path
        self.examples = []
        self.schemas, self.eval_foreign_key_maps = schemas, eval_foreign_key_maps

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

@registry.register('dataset', 'spider')
class SpiderDataset(torch.utils.data.Dataset): 
    def __init__(self, paths, tables_paths, db_path, limit=None):
        self.paths = paths
        self.db_path = db_path
        self.examples = []

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)

        for path in paths:
            raw_data = json.load(open(path))
            for entry in raw_data:
                item = SpiderItem(
                    text=entry['question_toks'],
                    code=entry['sql'],
                    schema=self.schemas[entry['db_id']],
                    orig=entry,
                    orig_schema=self.schemas[entry['db_id']].orig)
                self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    class Metrics:
        def __init__(self, dataset):
          self.dataset = dataset
          self.foreign_key_maps = {
              db_id: evaluation.build_foreign_key_map(schema.orig)
              for db_id, schema in self.dataset.schemas.items()
          }
          self.evaluator = evaluation.Evaluator(
              self.dataset.db_path,
              self.foreign_key_maps,
              'match')
          self.results = []

        def add(self, item, inferred_code, orig_question=None):
            ret_dict = self.evaluator.evaluate_one(
                item.schema.db_id, item.orig['query'], inferred_code)
            if orig_question:
                ret_dict["orig_question"] = orig_question
            self.results.append(ret_dict)
        
        def add_beams(self, item, inferred_codes, orig_question=None):
            beam_dict = {}
            if orig_question:
                beam_dict["orig_question"] = orig_question
            for i, code in enumerate(inferred_codes):
                ret_dict = self.evaluator.evaluate_one(
                    item.schema.db_id, item.orig['query'], code)
                beam_dict[i] = ret_dict
                if ret_dict["exact"] is True:
                    break
            self.results.append(beam_dict)

        def finalize(self):
            self.evaluator.finalize()
            return {
                'per_item': self.results,
                'total_scores': self.evaluator.scores
            }


@registry.register('dataset', 'spider_idiom_ast')
class SpiderIdiomAstDataset(torch.utils.data.Dataset):

    def __init__(self, paths, tables_paths, db_path, limit=None):
        self.paths = paths
        self.db_path = db_path
        self.examples = []

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)

        for path in paths:
            for line in open(path):
                entry = json.loads(line)
                item = SpiderItem(
                    text=entry['orig']['question_toks'],
                    code=entry['rewritten_ast'],
                    schema=self.schemas[entry['orig']['db_id']],
                    orig=entry['orig'],
                    orig_schema=self.schemas[entry['orig']['db_id']].orig)
                self.examples.append(item)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    Metrics = SpiderDataset.Metrics