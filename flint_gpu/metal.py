"""Diagnostics for Metal API failures; never modifies a candidate or its verdict."""
import ast


def argument_hint(source):
    try:tree=ast.parse(source)
    except SyntaxError:return ''
    libraries=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
            call=node.value.func
            if isinstance(call,ast.Attribute) and call.attr=='compile_shader':
                libraries.update(t.id for t in node.targets if isinstance(t,ast.Name))
    for node in ast.walk(tree):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id in libraries
            and len(node.args)>=2 and all(isinstance(a,(ast.List,ast.Tuple)) for a in node.args[:2])):
            return ('Metal API diagnostic: this launch groups tensors and scalars into two lists. '
                    'torch.mps.compile_shader binds each positional argument to a shader buffer slot. '
                    'For separate tensor/scalar parameters, pass them individually in declaration order, '
                    'for example lib.scan(x, h, c, out, h_out, c_out, B, T, D, threads=B*D). '
                    'Do not pass lib.scan([x,h,c,out,h_out,c_out], [B,T,D], ...). '
                    'Keep the equations and validation requirements unchanged.')
    return ''
