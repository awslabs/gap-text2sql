import ast
import os
import re

import asdl
import astor

from seq2struct import ast_util
from seq2struct.utils import registry


PYTHON_AST_FIELD_BLACKLIST = {
    'Attribute': {'ctx'},
    'Subscript': {'ctx'},
    'Starred': {'ctx'},
    'Name': {'ctx'},
    'List': {'ctx'},
    'Tuple': {'ctx'},
}


BUILTIN_TYPE_TO_PYTHON_TYPES = {
    'identifier': (str,),
    'int': (int,),
    'string': (str,),
    'bytes': (bytes,),
    'object': (int, float),
    'singleton': (bool, type(None))
}


def split_string_whitespace_and_camelcase(s):
    split_space = s.split(' ')
    result = []
    for token in split_space:
        if token: # \uE012 is an arbitrary glue character, from the Private Use Area.
            camelcase_split_token = re.sub('([a-z])([A-Z])', '\\1\uE012\\2', token).split('\uE012')
            result.extend(camelcase_split_token)
        result.append(' ')
    return result[:-1]




@registry.register('grammar', 'python')
class PythonGrammar:

    ast_wrapper = ast_util.ASTWrapper(
            asdl.parse(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'Python.asdl')))

    root_type = 'Module'

    pointers = set()

    @classmethod
    def parse(cls, code, section):
        try:
            py_ast = ast.parse(code)
            return cls.from_native_ast(py_ast)
        except SyntaxError:
            return  None

    @classmethod
    def unparse(cls, tree, item):
        ast_tree = cls.to_native_ast(tree)
        return astor.to_source(ast_tree)

    @classmethod
    def tokenize_field_value(cls, field_value):
        if isinstance(field_value, bytes):
            field_value = field_value.encode('latin1')
        else:
            field_value = str(field_value)
        return split_string_whitespace_and_camelcase(field_value)

    @classmethod
    def from_native_ast(cls, node):
        if not isinstance(node, ast.AST):
            return node
        # type: (ast.AST) -> Dict[str, Any]
        node_type = node.__class__.__name__
        field_infos = {
            f.name: f
            for f in cls.ast_wrapper.singular_types[node_type].fields
        }
        result = {'_type': node_type}  # type: Dict[str, Any]
        for field, value in ast.iter_fields(node):
            if field in PYTHON_AST_FIELD_BLACKLIST.get(node_type, set()):
                continue
            field_info = field_infos[field]
            if field_info.opt and value is None:
                continue
            if isinstance(value, (list, tuple)):
                assert field_info.seq
                if value:
                    result[field] = [cls.from_native_ast(v) for v in value]
            else:
                result[field] = cls.from_native_ast(value)
        return result

    @classmethod
    def to_native_ast(cls, node):
        if isinstance(node, (list, tuple)):
            return [cls.to_native_ast(item) for item in node]
        elif not isinstance(node, dict):
            return node

        result = getattr(ast, node['_type'])()
        # Add any missing fields
        type_info = cls.ast_wrapper.singular_types[node['_type']]
        for field_info in type_info.fields:
            if field_info.seq:
                value = node.get(field_info.name, [])
            elif field_info.opt:
                value = node.get(field_info.name, None)
            else:
                value = node[field_info.name]
            setattr(result, field_info.name, cls.to_native_ast(value))
        return result
