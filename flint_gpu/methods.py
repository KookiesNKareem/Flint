"""Lift direct module attributes into live operator arguments without changing user code."""
import ast
import inspect
import linecache
import textwrap
import hashlib


class MethodFast:
    def __init__(self, fn, factory):
        self.original = fn
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        node.decorator_list = []
        if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            raise TypeError('Decorated methods currently require explicit positional parameters')
        names = []
        class Lift(ast.NodeTransformer):
            def visit_Attribute(self, value):
                if isinstance(value.value, ast.Name) and value.value.id == 'self':
                    if value.attr not in names: names.append(value.attr)
                    return ast.copy_location(ast.Name(id='_muse_'+value.attr, ctx=value.ctx), value)
                return self.generic_visit(value)
        node = Lift().visit(node)
        node.args.args = node.args.args[1:] + [ast.arg(arg='_muse_'+name) for name in names]
        if node.args.defaults:
            raise TypeError('Decorated method defaults are not supported yet')
        ast.fix_missing_locations(node)
        source = ast.unparse(node)+'\n'
        filename = '<muse-method-'+hashlib.sha256(source.encode()).hexdigest()[:20]+'>'
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = dict(fn.__globals__)
        exec(compile(source, filename, 'exec'), namespace)
        lifted = namespace[node.name]
        lifted.__module__ = fn.__module__
        self.runtime = factory(lifted)
        parameters = [a.arg for a in node.args.args[:-len(names)]] if names else [a.arg for a in node.args.args]
        arguments = parameters + ['self.'+name for name in names]
        wrapper = 'def forward(self, '+', '.join(parameters)+'):\n    return _runtime('+', '.join(arguments)+')\n'
        env = {'_runtime': self.runtime}
        exec(compile(wrapper, filename+'-bound', 'exec'), env)
        self.bound = env['forward']
        self.bound.__name__ = fn.__name__
        self.bound.__qualname__ = fn.__qualname__
        self.bound.runtime = self.runtime
        self.bound.status = self.runtime.status

    def __get__(self, instance, owner=None):
        return self if instance is None else self.bound.__get__(instance, owner)

    def status(self):
        return self.runtime.status()
