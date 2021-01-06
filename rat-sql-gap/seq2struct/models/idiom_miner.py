import collections
import functools
import json
import os

from seq2struct.models import abstract_preproc
from seq2struct.utils import registry


TreeNode = collections.namedtuple('TreeNode', ['name', 'refs', 'children'])


class IdiomPreproc(abstract_preproc.AbstractPreproc):

    def __init__(self, grammar, save_path, censor_pointers):
        self.save_path = save_path
        self.censor_pointers = censor_pointers
        self.grammar = registry.construct('grammar', grammar)
        self.ast_wrapper = self.grammar.ast_wrapper

        self.items = collections.defaultdict(list)

    def validate_item(self, item, section):
        parsed = self.grammar.parse(item.code, section)
        if parsed:
            self.ast_wrapper.verify_ast(parsed)
            return True, parsed
        return section != 'train', None

    def add_item(self, item, section, validation_info):
        converted = AstConverter(self.grammar, self.censor_pointers).convert(
            validation_info)
        self.items[section].append({
            'text': item.text,
            'ast': converted,
            'orig': item.orig,
        })
    
    def clear_items(self):
        self.items.clear()

    def save(self):
        os.makedirs(self.save_path, exist_ok=True)
        for section in self.items:
            with open(os.path.join(self.save_path, '{}.jsonl'.format(section)), 'w') as f:
                for item in self.items[section]:
                    f.write(json.dumps(item) + '\n')
        
        # Output the grammar too
        expected_children = {'Null': [], 'End': []}
        field_name_nodes = []
        binarizers = []
        literals = []
        single_child = []
        for name, type_ in self.ast_wrapper.singular_types.items():
            expected_children[name] = [
                '{}-{}'.format(name, field.name) for field in type_.fields]
            field_name_nodes.extend(
                '{}-{}'.format(name, field.name) for field in type_.fields)
            for field in type_.fields:
                if len(type_.fields) == 1:
                    field_name = name
                else:
                    field_name = '{}-{}'.format(name, field.name)

                if field.seq:
                    binarizers.append(field_name)
                else:
                    single_child.append(field_name)
                if field.type in {'identifier', 'int', 'string', 'bytes', 'object', 'singleton'}:
                    literals.append(field_name)
                if field.type in self.grammar.pointers:
                    literals.append(field_name)
        with open(os.path.join(self.save_path, 'grammar.json'), 'w') as f:
            json.dump({
                'expected_children': expected_children,
                'field_name_nodes': field_name_nodes,
                'binarizers': binarizers,
                'literals': literals,
                'single_child': single_child,
            }, f, indent=2, sort_keys=True)
    
    def load(self):
        raise NotImplementedError
    
    def dataset(self, section):
        raise NotImplementedError


class AstConverter:
    def __init__(self, grammar, censor_pointers):
        self.grammar = grammar
        self.ast_wrapper = grammar.ast_wrapper
        self.symbols = {}

        self.split_constants = False
        self.preserve_terminal_types = True
        self.censor_pointers = censor_pointers

    def convert(self, node):
        if not isinstance(node, dict):
            if self.split_constants and isinstance(node, str):
                return [TreeNode(piece, [], []) for piece in node.split(' ')]
            if self.preserve_terminal_types:
                return TreeNode(node, [], [])
            return TreeNode(repr(node), [], [])

        node_type = node['_type']
        children = []
        fields_for_type = self.ast_wrapper.singular_types[node_type].fields

        for field in fields_for_type:
            field_node = node.get(field.name)
            if field.type in self.grammar.pointers:
                ref_getter = functools.partial(self.pointer_ref_getter, field.type)
            else:
                ref_getter = lambda value: []

            if len(fields_for_type) == 1:
                field_tree_node_name = node_type
            else:
                field_tree_node_name = '{}-{}'.format(node_type, field.name)

            if field.seq:
                field_tree_node = self.make_binarized_list(
                    ref_getter, field_tree_node_name, field.type, field_node or [])
            else:
                if field.opt and field_node is None:
                    child = TreeNode('Null', [], [])
                    refs = []
                else:
                    if self.censor_pointers and field.type in self.grammar.pointers:
                        child = None
                    else:
                        child = self.convert(field_node)
                    refs = ref_getter(field_node)
                
                if child is None:
                    child_list = []
                elif isinstance(child, list):
                    child_list = child
                else:
                    child_list = [child]

                field_tree_node = TreeNode(field_tree_node_name, refs, child_list)
            children.append(field_tree_node)

        if len(children) == 1:
            return children[0]
        else:
            return TreeNode(node_type, [], children)

    def pointer_ref_getter(self, pointer_type, pointer_value):
        symbol = (pointer_type, pointer_value)
        symbol_id = self.symbols.get(symbol)
        if symbol_id is None:
            symbol_id = self.symbols[symbol] = len(self.symbols)
        return [symbol_id]

    def make_binarized_list(self, ref_getter, node_name, elem_type, elems):
        # TODO: Change binarizer node names to be the sum/product type of the elements in the list, rather than the name of the field
        #       This is so that idioms containing binarizer fragments can be generalized.
        root = tree_node = TreeNode(node_name, [], [])
        for elem in elems:
            new_tree_node = TreeNode(node_name, [], [])
            if self.censor_pointers and elem_type in self.grammar.pointers:
                raise NotImplementedError
            else:
                elem_tree_node = self.convert(elem)
                tree_node.children.extend([elem_tree_node, new_tree_node])
            tree_node.refs.extend(ref_getter(elem))
            tree_node = new_tree_node
        tree_node.children.append(TreeNode('End', [], []))
        return root


@registry.register('model', 'IdiomMiner')
class IdiomMinerModel:
    '''A dummy model for housing IdiomPreproc.'''
    Preproc = IdiomPreproc
