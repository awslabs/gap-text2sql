# coding: utf-8
import argparse
import collections
import copy
import io
import itertools
import json

import _jsonnet
import asdl
import attr
import networkx
import tqdm

from seq2struct import datasets
from seq2struct import models
from seq2struct.utils import registry

# Initial units: node and its required product-type fields, recursively
# Eligible triples: node, field_name then
# - opt: None, or type of field
# - seq: None, or if len > 0:
#   - sum type: type of first element
#   - product type: type of first element (indicates there's more than one element)
#   - constant: value
# - neither: type of field, or constant value
#
# Merging
# 1. Replace type of node with something else
# 2. Promote fields of the involved field
#   - None: nothing to promote
#   - seq, type of first element: all fields of that type
#   - neither: type of field


class IdentitySet(collections.abc.MutableSet):
    def __init__(self, iterable=()):
        self.map = {id(x): x for x in iterable}

    def __contains__(self, value):
        return id(value) in self.map
    
    def __iter__(self):
        return self.map.values()
    
    def __len__(self):
        return len(self.map)
    
    def add(self, value):
        self.map[id(value)] = value
    
    def discard(self, value):
        self.map.pop(id(value))


@attr.s
class TypeInfo:
    name = attr.ib()
    base_name = attr.ib()
    predecessor_name = attr.ib()
    predecessor_triple = attr.ib()
    # Fields whose values need to be specified in a node of this type.
    # OrderedDict or dict. Keys: str if single element, tuple if more than one element
    unset_fields = attr.ib()
    # Fields whose values are already known.
    # dict. Keys: always tuple (even if only one element)
    preset_fields = attr.ib()
    ## Sequential fields which have been ended.
    ## set. Elements: always tuple (even if only one element)
    #depleted_fields = attr.ib()
    preset_seq_elem_counts = attr.ib(factory=lambda: collections.Counter())


@attr.s(frozen=True)
class Primitive:
    value = attr.ib()


class TreeBPE:
    def __init__(self, grammar):
        self.grammar = grammar
        self.ast_wrapper = grammar.ast_wrapper

        self.type_infos = {
            k: TypeInfo(
                name=k,
                base_name=k,
                predecessor_name=k,
                predecessor_triple=None,
                unset_fields=collections.OrderedDict((field.name, field) for field in v.fields),
                preset_fields={},
            ) for k, v in self.ast_wrapper.singular_types.items()
        }
        self.type_graph = networkx.DiGraph()
        for k in self.ast_wrapper.singular_types:
            self.type_graph.add_node(k)
        
        self.created_types = []
        self.pre_iteration_counts = []
        self.iterations_finished = 0
    
    def run_iteration(self, trees):
        triple_occurrences, node_type_counts = self.count_triples(trees)
        self.pre_iteration_counts.append(node_type_counts)

        # Most frequent
        most_freq_triple, most_freq_occurrences = max(
                triple_occurrences.items(), key=lambda kv: len(kv[1]))
        if len(most_freq_occurrences) == 1:
            raise Exception('No more work to do!')

        existing_type_name, field_name, field_info = most_freq_triple
        tuple_name = field_name if isinstance(field_name, tuple) else (field_name,)
        existing_type = self.type_infos[existing_type_name]
        existing_field = existing_type.unset_fields[field_name]

        promoted_fields = []
        promoted_seq_elem_counts = collections.Counter()
        promoted_preset_fields = {}
        if isinstance(field_info, Primitive) or field_info is None:
            pass
        else:
            # Figure out which fields of type `field_info` should be promoted
            # Example:
            #   most_freq_triple = ('Call', 'func', 'Name')
            #   field_info = 'Name'
            #   type_infos['Name'].unset_fields = {'id': Field(identifier, id)}
            for is_preset, (field_field_name, field_field) in itertools.chain(
                    zip(itertools.repeat(False), self.type_infos[field_info].unset_fields.items()),
                    zip(itertools.repeat(True), self.type_infos[field_info].preset_fields.items())):
                if isinstance(field_field_name, tuple):
                    field_field_tuple_name = field_field_name
                else:
                    field_field_tuple_name = (field_field_name,)
                if existing_field.seq:
                    suffix = (existing_type.preset_seq_elem_counts[tuple_name],) + field_field_tuple_name
                else:
                    suffix = field_field_tuple_name
                new_name = tuple_name + suffix
                if isinstance(field_field, asdl.Field):
                    new_field = asdl.Field(
                            type=field_field.type,
                            name=new_name,
                            seq=field_field.seq,
                            opt=field_field.opt)
                else:
                    new_field = field_field

                if is_preset:
                    promoted_preset_fields[new_name] = new_field
                else:
                    promoted_fields.append((field_field, new_field))

                seq_elem_count = self.type_infos[field_info].preset_seq_elem_counts[field_field_tuple_name]
                if seq_elem_count:
                    promoted_seq_elem_counts[new_name] = seq_elem_count

        # Create a new type
        new_preset_fields = {**existing_type.preset_fields, **promoted_preset_fields}
        new_preset_seq_elem_counts = existing_type.preset_seq_elem_counts + promoted_seq_elem_counts
        if existing_field.seq and field_info is not None:
            new_preset_fields[
                tuple_name + (new_preset_seq_elem_counts[tuple_name],)] = field_info
            new_preset_seq_elem_counts[tuple_name] += 1
        else:
            new_preset_fields[tuple_name] = field_info

        new_unset_fields = {
            **{f.name: f for old_field, f in promoted_fields},
            **existing_type.unset_fields
        }
        if field_info is None or not existing_field.seq:
            # Only unset if...
            # - field is not sequential
            # - field has been set to None, meaning the end of a sequence
            del new_unset_fields[field_name]

        new_type = TypeInfo(
            name='Type{:04d}_{}'.format(self.iterations_finished, existing_type.base_name),
            base_name=existing_type.base_name,
            predecessor_name=existing_type.name,
            predecessor_triple=most_freq_triple,
            unset_fields=new_unset_fields,
            preset_fields=new_preset_fields,
            preset_seq_elem_counts = new_preset_seq_elem_counts
        )
        self.type_infos[new_type.name] = new_type
        self.created_types.append(new_type)
        self.type_graph.add_edge(new_type.name, existing_type.name)
        self.iterations_finished += 1

        # Tracks which occurrences have been removed due to promotion.
        discarded = IdentitySet()
        for occ in most_freq_occurrences:
            if occ in discarded:
                continue

            occ['_type'] = new_type.name
            def delete_obsoleted_field():
                if existing_field.seq:
                    # todo: change 0 if we can promote other elements
                    del occ[field_name][0]
                    if not occ[field_name]:
                        del occ[field_name]
                else:
                    del occ[field_name]

            if isinstance(field_info, Primitive):
                delete_obsoleted_field()
            elif field_info is None:
                pass
            else:
                if existing_field.seq:
                    # todo: change 0 if we can promote other elements
                    value_to_promote = occ[field_name][0]
                else:
                    value_to_promote = occ[field_name]
                delete_obsoleted_field()
                discarded.add(value_to_promote)

                for old_field, new_field in promoted_fields:
                    if old_field.name not in value_to_promote:
                        assert old_field.opt or old_field.seq
                        continue
                    occ[new_field.name] = value_to_promote[old_field.name]
                    assert occ[new_field.name]
    
    def finish(self, trees):
        _, node_type_counts = self.count_triples(trees)
        self.pre_iteration_counts.append(node_type_counts)

    def count_triples(self, trees):
        triple_occurrences = collections.defaultdict(list)
        node_type_counts = collections.Counter()

        for tree in trees:
            queue = collections.deque([tree])
            while queue:
                node = queue.pop()
                node_type_counts[node['_type']] += 1
                for field_name, field in self.type_infos[node['_type']].unset_fields.items():
                    if field_name in node:
                        field_value = node[field_name]
                        is_primitive = field.type in self.ast_wrapper.primitive_types

                        if field.seq:
                            relevant_value = field_value[0]
                            if not is_primitive:
                                queue.extend(field_value)
                        else:
                            relevant_value = field_value
                            if not is_primitive:
                                queue.append(field_value)

                        if is_primitive:
                            field_info = Primitive(relevant_value)
                        else:
                            field_info = relevant_value['_type']
                    else:
                        assert field.seq or field.opt
                        field_info = None

                    triple_occurrences[node['_type'], field_name, field_info].append(node)

                for field_name in self.type_infos[node['_type']].preset_fields:
                    assert field_name not in node
        
        return triple_occurrences, node_type_counts
    
    def visualize(self, root_type: TypeInfo):
        result = io.StringIO()
        def print_type(this_type, parent_lasts, field_prefix):
            def print_child(s, last, parent_lasts):
                for parent_last in parent_lasts:
                    if parent_last:
                        result.write('  ')
                    else:
                        result.write('│ ')
                
                if last:
                    result.write('└─')
                else:
                    result.write('├─')
                print(s, file=result)

            if parent_lasts:
                print_child(this_type.base_name, parent_lasts[-1], parent_lasts[:-1])
            else:
                print(this_type.base_name, file=result)
            fields = self.type_infos[this_type.base_name].unset_fields

            for i, field in enumerate(fields.values()):
                last_field = i + 1 == len(fields)
                # Print the name of the field
                print_child(
                    '{} [{}]{}'.format(
                        field.name, field.type, '?' if field.opt else '*' if field.seq else ''),
                    last_field,
                    parent_lasts)
                
                field_path = field_prefix + (field.name,)
                parent_lasts_for_field = parent_lasts + (last_field,)

                if field.opt and field_path in root_type.preset_fields and root_type.preset_fields[field_path] is None:
                    # Don't print '??' because we've already determined that the field should be unset
                    pass
                elif field.seq:
                    # Print all the elements
                    if field_path in root_type.preset_fields:
                        assert root_type.preset_fields[field_path] is None
                        seq_complete = True
                    else:
                        seq_complete = False

                    preset_count = root_type.preset_seq_elem_counts[field_path]

                    for i in range(preset_count):
                        last_seq_elem = seq_complete and i + 1 == preset_count
                        seq_elem_path = field_path + (i,)
                        field_value = root_type.preset_fields[seq_elem_path]
                        if isinstance(field_value, Primitive):
                            print_child(
                                repr(field_value.value),
                                last_seq_elem,
                                parent_lasts_for_seq)
                        else:
                            print_type(
                                self.type_infos[field_value],
                                parent_lasts_for_field + (last_seq_elem,),
                                seq_elem_path)

                    if not seq_complete:
                        print_child('??', True, parent_lasts_for_field)
                else:
                    if field_path not in root_type.preset_fields:
                        print_child('??', True, parent_lasts_for_field)
                    else:
                        field_value = root_type.preset_fields[field_path]
                        if isinstance(field_value, Primitive):
                            print_child(repr(field_value.value), True, parent_lasts_for_field)
                        else:
                            print_type(
                                self.type_infos[field_value],
                                parent_lasts_for_field + (True,), 
                                field_path)

        print_type(root_type, (), ())
        return result.getvalue()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--section', default='train')
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--vis-out')
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    # 0. Construct preprocessors
    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'])
    model_preproc.load()

    # 3. Get training data somewhere
    preproc_data = model_preproc.dataset(args.section)
    all_trees = [dec.tree for enc, dec in preproc_data]
    tree_bpe = TreeBPE(model_preproc.dec_preproc.grammar)
    for i in tqdm.tqdm(range(args.num_iters), dynamic_ncols=True):
        tree_bpe.run_iteration(all_trees)
    tree_bpe.finish(all_trees)
    print('Finished')

    if args.vis_out:
        f = open(args.vis_out, 'w')
        f.write('''# Documentation
#
# Idiom trees are printed like this:
#   NodeType
#   ├─field1 [field1_type]
#   ├─field2 [field2_type]?
#   └─field3 [field3_type]*
# ? indicates the field is optional.
# * indicates the field is sequential.
#
# If a field has a known primitive value, it is written like this:
#   └─field3 [str]
#     └─'value'
#
# If a field has a known type for its value, it is written like this:
#   └─field3 [field3_type]
#     └─Field3NodeType
#       └─...
#
# If a field:
# - does not have a known value, or
# - is sequential and the idiom allows for further entries at the end
# it is written like this:
#   └─field3 [field3_type]
#     └─??
# 
# If a field:
# - is optional and known to lack a value, or
# - is sequential and the idiom does not allow for further entries at the end
# then there is no ??.

Initial node type frequency:
''')

        for k, v in tree_bpe.pre_iteration_counts[0].most_common():
            print('- {}: {}'.format(k, v), file=f)
        print(file=f)

        for i, type_info in enumerate(tree_bpe.created_types):
            print('# Idiom {} [{}]'.format(i, type_info.name), file=f)
            print('# Descended from {} by setting {} to {}'.format(*type_info.predecessor_triple), file=f)
            print('# Frequency at creation: {}'.format(tree_bpe.pre_iteration_counts[i + 1][type_info.name]), file=f)
            print(tree_bpe.visualize(type_info), file=f)
        f.close()
    else:
        import IPython; IPython.embed()

if __name__ == '__main__':
    main()

#    ast_wrapper = grammar.ast_wrapper
#
#    # TODO: Revive the following
#    ## Preprocess the grammar
#    ## Create initial units: node and its required product-type fields, recursively
#    #units = {name: {} for name in ast_wrapper.singular_types}
#    #for name, cons in ast_wrapper.singular_types.items():
#    #    unit_fields = units[name]
#    #    for field in cons.fields:
#    #        if not field.seq and not field.opt and field.type in ast_wrapper.singular_types:
#    #            unit_fields[field.name] = units[field.type]
#
#    # field name syntax:
#    # (field_name{1}, ..., field_name{k}, i, field_name{k+1}, ..., field_name{n})
#    
#    type_infos = {
#        k: TypeInfo(
#            name=k,
#            base_name=k,
#            predecessor_name=k,
#            unset_fields={field.name: field for field in v.fields},
#            preset_fields={}
#        ) for k, v in ast_wrapper.singular_types.items()
#    }
#
#    # Count types
#    for iteration in range(100):