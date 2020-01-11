#!/usr/bin/python

import json as json
import ast

from numpy import unicode


def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s


def parse_file(filename, file_id):
    # global c, d
    tree = ast.parse(read_file_to_string(filename))
    if len(tree.body) != 0:
        first_lineno = tree.body[0].lineno
    else:
        first_lineno = 0
    json_tree = []

    def gen_identifier(identifier, parent_lineno, file_id, node_type='identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        json_node["location"] = (file_id, parent_lineno, pos)
        return pos
    
    def traverse_list(l, parent_lineno, file_id, node_type='list'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        children = []
        json_node["location"] = (file_id, parent_lineno, pos)

        for item in l:
            children.append(traverse(item, parent_lineno, file_id))
        if len(children) != 0:
            json_node['children'] = children
        return pos
        
    def traverse(node, parent_lineno, file_id):
        # Fixing the location information (lineno and col_offset of nodes recursively)
        ast.fix_missing_locations(node)

        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__
        children = []
        lineno = parent_lineno

        if "lineno" in node._attributes:
            lineno = node.lineno
        json_node["location"] = (file_id, lineno, pos)

        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = unicode(node.n)
        elif isinstance(node, ast.Str):
            json_node['value'] = node.s.decode('utf-8')
            # json_node['value'] = node.s
            # json_node["location"] = (file_id, parent_lineno, lineno, pos)
        elif isinstance(node, ast.alias):
            json_node['value'] = unicode(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname, lineno, file_id))
        elif isinstance(node, ast.FunctionDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = unicode(node.module)
        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n, lineno, file_id))
        elif isinstance(node, ast.keyword):
            json_node['value'] = unicode(node.arg)

        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target, lineno, file_id))
            children.append(traverse(node.iter, lineno, file_id))
            children.append(traverse_list(node.body, lineno, file_id, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse,  lineno, file_id, 'orelse'))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test, lineno, file_id))
            children.append(traverse_list(node.body, lineno, file_id, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, lineno, file_id, 'orelse'))
        elif isinstance(node, ast.With):
            children.append(traverse(node.context_expr, lineno, file_id))
            if node.optional_vars:
                children.append(traverse(node.optional_vars, lineno, file_id))
            children.append(traverse_list(node.body, lineno, file_id, 'body'))
        elif isinstance(node, ast.TryExcept):
            children.append(traverse_list(node.body, lineno, file_id, 'body'))
            children.append(traverse_list(node.handlers, lineno, file_id, 'handlers'))
            if node.orelse:
                children.append(traverse_list(node.orelse, lineno, file_id, 'orelse'))
        elif isinstance(node, ast.TryFinally):
            children.append(traverse_list(node.body, lineno, file_id, 'body'))
            children.append(traverse_list(node.finalbody, lineno, file_id, 'finalbody'))
        elif isinstance(node, ast.arguments):
            children.append(traverse_list(node.args, lineno, file_id, 'args'))
            children.append(traverse_list(node.defaults, lineno, file_id, 'defaults'))
            if node.vararg:
                children.append(gen_identifier(node.vararg, lineno, file_id, 'vararg'))
            if node.kwarg:
                children.append(gen_identifier(node.kwarg, lineno, file_id, 'kwarg'))
        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], lineno, file_id, 'type'))
            if node.name:
                children.append(traverse_list([node.name], lineno, file_id, 'name'))
            children.append(traverse_list(node.body, lineno, file_id, 'body'))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, lineno, file_id, 'bases'))
            children.append(traverse_list(node.body, lineno, file_id, 'body'))
            children.append(traverse_list(node.decorator_list, lineno, file_id, 'decorator_list'))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args, lineno, file_id))
            children.append(traverse_list(node.body, lineno, file_id, 'body'))
            children.append(traverse_list(node.decorator_list, lineno, file_id, 'decorator_list'))
        else:
            # Default handling: iterate over children.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) \
                        or isinstance(child, ast.operator) \
                        or isinstance(child, ast.boolop) \
                        or isinstance(child, ast.unaryop) \
                        or isinstance(child, ast.cmpop):
                    # Directly include expr_context,
                    # and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + type(child).__name__
                else:
                    children.append(traverse(child, lineno, file_id))
                
        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, lineno, file_id, 'attr'))
                
        if len(children) != 0:
            json_node['children'] = children
        return pos
    
    traverse(tree, first_lineno, file_id)
    return json.dumps(json_tree, separators=(',', ':'), ensure_ascii=False)


if __name__ == "__main__":
    try:
        filename = "data/pimutils/khal/tests/aux_fake.py"
        print(parse_file(filename, 0))
        print(type(parse_file(filename,0)))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
