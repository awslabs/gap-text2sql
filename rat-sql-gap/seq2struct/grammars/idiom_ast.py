import collections
import copy
import enum
import functools
import json
import re

import asdl
import attr

from seq2struct import ast_util
from seq2struct.utils import registry


class HoleType(enum.Enum):
    ReplaceSelf = 1
    AddChild = 2


class MissingValue:
    pass


@attr.s
class SeqField:
    type_name = attr.ib()
    field = attr.ib()


@registry.register('grammar', 'idiom_ast')
class IdiomAstGrammar:

    def __init__(self, base_grammar, template_file, root_type=None,
        all_sections_rewritten=False):
        self.base_grammar = registry.construct('grammar', base_grammar)
        self.templates = json.load(open(template_file))
        self.all_sections_rewritten = all_sections_rewritten

        self.pointers = self.base_grammar.pointers
        self.ast_wrapper = copy.deepcopy(self.base_grammar.ast_wrapper)
        self.base_ast_wrapper = self.base_grammar.ast_wrapper
        # TODO: Override root_type more intelligently
        self.root_type = self.base_grammar.root_type
        if base_grammar['name'] == 'python':
            self.root_type = 'mod'

        singular_types_with_single_seq_field = set(
            name for name, type_info in self.ast_wrapper.singular_types.items()
            if len(type_info.fields) == 1 and type_info.fields[0].seq)
        seq_fields = {
            '{}-{}'.format(name, field.name): SeqField(name, field)
            for name, type_info in self.ast_wrapper.singular_types.items()
            for field in type_info.fields
            if field.seq
        }

        templates_by_head_type = collections.defaultdict(list)
        for template in self.templates:
            head_type = template['idiom'][0]
            # head_type can be one of the following:
            # 1. name of a constructor/product with a single seq field. 
            # 2. name of any other constructor/product
            # 3. name of a seq field (e.g. 'Dict-keys'),
            #    when the containing constructor/product contains more than one field
            #    (not yet implemented)
            # For 1 and 3, the template should be treated as a 'seq fragment'
            # which can occur in any seq field of the corresponding sum/product type.
            # However, the NL2Code model has no such notion currently.
            if head_type in singular_types_with_single_seq_field:
                # field.type could be sum type or product type, but not constructor
                field = self.ast_wrapper.singular_types[head_type].fields[0]
                templates_by_head_type[field.type].append((template, SeqField(head_type, field)))
                templates_by_head_type[head_type].append((template, None))
            elif head_type in seq_fields:
                seq_field = seq_fields[head_type]
                templates_by_head_type[seq_field.field.type].append((template, seq_field))
            else:
                templates_by_head_type[head_type].append((template, None))
        
        types_to_replace = {}

        for head_type, templates in templates_by_head_type.items():
            constructors, seq_fragment_constructors = [], []
            for template, seq_field in templates:
                if seq_field:
                    if head_type in self.ast_wrapper.product_types:
                        seq_type = '{}_plus_templates'.format(head_type)
                    else:
                        seq_type = head_type

                    seq_fragment_constructors.append(
                        self._template_to_constructor(template, '_{}_seq'.format(seq_type), seq_field))
                else:
                    constructors.append(self._template_to_constructor(template, '', seq_field))

            # head type can be:
            # constructor (member of sum type)
            if head_type in self.ast_wrapper.constructors:
                assert constructors
                assert not seq_fragment_constructors

                self.ast_wrapper.add_constructors_to_sum_type(
                    self.ast_wrapper.constructor_to_sum_type[head_type],
                    constructors)
            
            # sum type
            elif head_type in self.ast_wrapper.sum_types:
                assert not constructors
                assert seq_fragment_constructors
                self.ast_wrapper.add_seq_fragment_type(head_type, seq_fragment_constructors)

            # product type
            elif head_type in self.ast_wrapper.product_types:
                # Replace Product with Constructor
                # - make a Constructor
                orig_prod_type = self.ast_wrapper.product_types[head_type]
                new_constructor_for_prod_type = asdl.Constructor(
                    name=head_type, fields=orig_prod_type.fields)
                # - remove Product in ast_wrapper
                self.ast_wrapper.remove_product_type(head_type)

                # Define a new sum type
                # Add the original product type and template as constructors
                name = '{}_plus_templates'.format(head_type)
                self.ast_wrapper.add_sum_type(
                    name,
                    asdl.Sum(types=constructors + [new_constructor_for_prod_type]))
                # Add seq fragment constructors
                self.ast_wrapper.add_seq_fragment_type(name, seq_fragment_constructors)

                # Replace every occurrence of the product type in the grammar
                types_to_replace[head_type] = name
            
            # built-in type
            elif head_type in self.ast_wrapper.primitive_types:
                raise NotImplementedError(
                    'built-in type as head type of idiom unsupported: {}'.format(head_type))
                # Define a new sum type
                # Add the original built-in type and template as constructors
                # Replace every occurrence of the product type in the grammar
            
            else:
                raise NotImplementedError('Unable to handle head type of idiom: {}'.format(head_type))
            
        # Replace occurrences of product types which have been used as idiom head types
        for constructor_or_product in self.ast_wrapper.singular_types.values():
            for field in constructor_or_product.fields:
                if field.type in types_to_replace:
                    field.type = types_to_replace[field.type]
        
        self.templates_containing_placeholders = {}
        for name, constructor in self.ast_wrapper.singular_types.items():
            if not hasattr(constructor, 'template'):
                continue
            hole_values = {}
            for field in constructor.fields:
                hole_id = self.get_hole_id(field.name)
                placeholder = ast_util.HoleValuePlaceholder(id=hole_id, is_seq=field.seq, is_opt=field.opt)
                if field.seq:
                    hole_values[hole_id] = [placeholder]
                else:
                    hole_values[hole_id] = placeholder

            self.templates_containing_placeholders[name] = constructor.template(hole_values)

        if root_type is not None:
            if isinstance(root_type, (list, tuple)):
                for choice in root_type:
                    if (choice in self.ast_wrapper.singular_types or
                        choice in self.ast_wrapper.sum_types):
                        self.root_type = choice
                        break
            else:
                self.root_type = root_type

    def parse(self, code, section):
        if self.all_sections_rewritten or section == 'train':
            return self.convert_idiom_ast(code, template_id=None)()
        else:
            return self.base_grammar.parse(code, section)

    def unparse(self, tree, item):
        expanded_tree = self._expand_templates(tree)
        self.base_ast_wrapper.verify_ast(expanded_tree)
        return self.base_grammar.unparse(expanded_tree, item)

    def tokenize_field_value(self, field_value):
        return self.base_grammar.tokenize_field_value(field_value)

    #
    #
    #

    @classmethod
    def get_hole_id(cls, field):
        m = re.match('^hole(\d+)$', field)
        if not m:
            raise ValueError('Unexpected field name: {}'.format(field))
        return int(m.group(1))

    def _expand_templates(self, tree):
        if not isinstance(tree, dict):
            return tree

        node_type = tree['_type']
        constructor = self.ast_wrapper.constructors.get(node_type)

        expanded_fields = {}
        for field, value in tree.items():
            if field == '_type':
                continue
            if isinstance(value, (list, tuple)):
                result = []
                for item in value:
                    converted = self._expand_templates(item)
                    if isinstance(item, dict) and re.match('^Template\d+_.*_seq$', item['_type']):
                        # TODO: Handle seq fragment fields here
                        item_type_info = self.ast_wrapper.constructors[converted['_type']]

                        assert len(item_type_info.fields) == 1
                        assert item_type_info.fields[0].seq
                        result += converted.get(item_type_info.fields[0].name, [])
                    else:
                        result.append(converted)
                expanded_fields[field] = result
            else:
                expanded_fields[field] = self._expand_templates(value)

        if constructor is None or not hasattr(constructor, 'template'):
            return {'_type': node_type, **expanded_fields}
 
        template = constructor.template
        hole_values = {}
        for field, expanded_value in expanded_fields.items():
            hole_id = self.get_hole_id(field)

            # Do something special if we have a seq fragment
            hole_values[hole_id] = expanded_value
        return template(hole_values)
    
    def _template_to_constructor(self, template_dict, suffix, seq_field):
        hole_node_types = {}

        # Find where the holes occur
        stack = [(None, template_dict['idiom'], None)]
        while stack:
            parent, node, child_index = stack.pop()
            node_type, ref_symbols, hole_id, children = node
            if hole_id is not None:
                assert hole_id not in hole_node_types
                # node_type could be:
                # - name of field
                #   => hole type is same as field's type
                # - name of type, if it only has one child
                # - binarizer
                hyphenated_node_type = None
                unhyphenated_node_type = None

                hole_type_str = template_dict['holes'][hole_id]['type']
                if hole_type_str == 'AddChild':
                    node_type_for_field_type = node_type

                elif hole_type_str == 'ReplaceSelf':
                    # Two types of ReplaceSelf
                    # 1. Node has a hyphen: should be a repeated field
                    # 2. Node lacks a hyphen, and
                    #    2a. node is same as parent: a repeated field
                    #    2b. node is not the same as parent: an elem
                    if '-' in node_type:
                        node_type_for_field_type = node_type
                    else:
                        node_type_for_field_type = parent[0]
                        #if '-' in parent_type:
                        #    hyphenated_node_type = parent_type
                        #else:
                        #    unhyphenated_node_type = parent_type

                field_info = self._get_field_info_from_name(node_type_for_field_type)

                # Check for situations like 
                # Call-args
                # |       \
                # List[0] Call-args[1]
                #
                # Tuple
                # |       \
                # Tuple[0] Tuple[1]
                # where hole 0 should not be a sequence.

                if field_info.seq and hole_type_str == 'ReplaceSelf' and '-' not in node_type:
                    assert child_index in (0, 1)
                    seq = child_index == 1
                else:
                    seq = field_info.seq
                hole_node_types[hole_id] = (field_info.type, seq, field_info.opt)
            stack += [(node, child, i) for i, child in enumerate(children)]
        
        # Create fields for the holes
        fields = []
        for hole in template_dict['holes']:
            i = hole['id']
            field_type, seq, opt = hole_node_types[i]
            field = asdl.Field(type=field_type, name='hole{}'.format(i), seq=seq, opt=opt)
            field.hole_type = HoleType[hole['type']]
            fields.append(field)

        constructor = asdl.Constructor('Template{}{}'.format(template_dict['id'], suffix), fields)
        constructor.template = self.convert_idiom_ast(template_dict['idiom'], template_id=template_dict['id'], seq_field=seq_field)

        return constructor

    def _get_field_info_from_name(self, node_type):
        if '-' in node_type:
            type_name, field_name = node_type.split('-') 
            type_info = self.ast_wrapper.singular_types[type_name]
            field_info, = [field for field in type_info.fields if field.name == field_name]
        else:
            type_info = self.ast_wrapper.singular_types[node_type]
            assert len(type_info.fields) == 1
            field_info = type_info.fields[0]
        return field_info
    
    @classmethod
    def _node_type(cls, node):
        if isinstance(node[0], dict):
            if 'nt' in node[0]:
                return node[0]['nt']
            elif 'template_id' in node[0]:
                return 'Template{}'.format(node[0]['template_id'])
        else:
            return node[0]

    def convert_idiom_ast(self, idiom_ast, template_id=None, seq_fragment_type=None, seq_field=None):
        if template_id is not None:
            node_type, ref_symbols, hole, children = idiom_ast
        else:
            node_type, ref_symbols, children = idiom_ast

        is_template_node = False
        extra_types = []
        if isinstance(node_type, dict):
            if seq_fragment_type:
                suffix = '_{}_seq'.format(seq_fragment_type)
            else:
                suffix = '' 
            if 'template_id' in node_type:
                node_type = 'Template{}{}'.format(node_type['template_id'], suffix)
                is_template_node = True
            elif 'nt' in node_type and 'mt' in node_type:
                extra_types = ['Template{}{}'.format(i, suffix) for i in node_type['mt']]
                node_type = node_type['nt']

        if seq_field is None:
            field_infos = self.ast_wrapper.singular_types[node_type].fields
        else:
            field_infos = [seq_field.field]

        # Each element of this list is a tuple (field, child)
        # - field: asdl.Field object
        # - child: an idiom_ast node
        #   If field.seq then child will be a binarizer node, or a template headed by a binarizer
        #   Otherwise, child will be a node whose type indicates the field's name (e.g. Call-func),
        #   and with a single child that contains the content of the field
        children_to_convert = []
        if is_template_node:
            assert len(children) == len(field_infos)
            for field, child in zip(field_infos, children):
                if field.hole_type == HoleType.ReplaceSelf and  field.seq:
                    children_to_convert.append((field, child))
                else:
                    assert not field.seq
                    dummy_node = list(idiom_ast)
                    dummy_node[0] = '{}-{}'.format(node_type, field.name)
                    dummy_node[-1] = [child]
                    children_to_convert.append((field, dummy_node))
               # else:
               #     raise ValueError('Unexpected hole_type: {}'.format(field.hole_type))
        else:
            fields_by_name = {f.name: f for f in field_infos}
            if len(field_infos) == 0:
                pass
            elif len(field_infos) == 1:
                children_to_convert.append((field_infos[0], idiom_ast))
            else:
                prefix_len = len(node_type) + 1
                for child in children:
                    field_name = self._node_type(child)[prefix_len:]
                    children_to_convert.append((fields_by_name[field_name], child))
        assert set(field.name for field, _ in children_to_convert) == set(field.name for field in field_infos)

        def result_creator(hole_values={}):
            if template_id is not None and hole is not None and self.templates[template_id]['holes'][hole]['type'] == 'ReplaceSelf':
                return hole_values.get(hole, MissingValue)

            result = {}
            for field, child_node in children_to_convert:
                # field: ast.Field object representing the field in the ASDL Constructor/Product
                # child_node:
                #   the idiom_ast nodes which specify the field
                #   len(child_children)
                #   - 0: should never happen
                #   - 1: for regular fields, opt fields, seq fields if length is 0 or represented by a template node
                #   - 2: for seq fields of length >= 1
                if field.type in self.ast_wrapper.primitive_types:
                    convert = lambda node: (lambda hole_values: self.convert_builtin_type(field.type, self._node_type(node)))
                else:
                    convert = functools.partial(
                        self.convert_idiom_ast, template_id=template_id)

                if field.seq:
                    value = []
                    while True:
                        # child_node[2]: ID of hole
                        if template_id is not None and child_node[2] is not None:
                            hole_value =  hole_values.get(child_node[2], [])
                            assert isinstance(hole_value, list)
                            value += hole_value
                            #if value:
                            #    assert isinstance(hole_value, list)
                            #    value += hole_value
                            #else:
                            #    return hole_value
                            assert len(child_node[-1]) == 0
                            break

                        child_type, child_children = child_node[0], child_node[-1]
                        if isinstance(child_type, dict) and 'template_id' in child_type:
                            # Another template
                            value.append(convert(child_node, seq_fragment_type=field.type if field.seq else None)(hole_values))
                            break
                        # If control reaches here, child_node is a binarizer node
                        if len(child_children) == 1:
                            assert self._node_type(child_children[0]) == 'End'
                            break
                        elif len(child_children) == 2:
                            # TODO: Sometimes we need to value.extend?
                            value.append(convert(child_children[0])(hole_values))
                            child_node = child_children[1]
                        else:
                            raise ValueError('Unexpected number of children: {}'.format(len(child_children)))
                    present = bool(value)
                elif field.opt:
                    # child_node[2]: ID of hole
                    if template_id is not None and child_node[2] is not None:
                        assert len(child_node[-1]) == 0
                        present = child_node[2] in hole_values
                        value = hole_values.get(child_node[2])
                    else:
                        assert len(child_node[-1]) == 1
                        # type of first (and only) child of child_node
                        if self._node_type(child_node[-1][0]) == 'Null':
                            value = None
                            present = False
                        else:
                            value = convert(child_node[-1][0])(hole_values)
                            present = value is not MissingValue
                else:
                    if template_id is not None and child_node[2] is not None:
                        assert len(child_node[-1]) == 0
                        value = hole_values[child_node[2]]
                        present = True
                    else:
                        assert len(child_node[-1]) == 1
                        value = convert(child_node[-1][0])(hole_values)
                        present = True
                if present:
                    result[field.name] = value

            result['_type'] = node_type
            result['_extra_types'] = extra_types
            return result

        return result_creator

    def convert_builtin_type(self, field_type, value):
        if field_type == 'singleton' and value == 'Null':
            return None
        return value
