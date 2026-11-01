"""Deterministic, reviewable edits. Discovery never rewrites a reference body."""
import ast
import difflib
import hashlib
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def functions(root):
    from .qualification import source_files
    root=Path(root).resolve();result={}
    for relative in source_files(root):
        if not relative.endswith('.py'):continue
        path=root/relative
        if path.is_symlink():continue
        try:source=path.read_text();tree=ast.parse(source)
        except (OSError,SyntaxError,UnicodeError):continue
        for node in ast.walk(tree):
            if not isinstance(node,ast.FunctionDef):continue
            key=f'{relative}:{node.lineno}:{node.name}'
            result[key]={'file':relative,'line':node.lineno,'end_line':node.end_lineno,
                         'function':node.name,'node':node,'source':source,'tree':tree}
    return result


def eligibility(item):
    node=item['node'];args=node.args
    if node.decorator_list:return 'already decorated; inspect manually'
    if args.vararg or args.kwarg or args.kwonlyargs:return 'variadic/keyword-only interface needs review'
    if any(isinstance(n,(ast.Yield,ast.YieldFrom,ast.Await,ast.Global,ast.Nonlocal)) for n in ast.walk(node)):
        return 'stateful or suspended function needs review'
    if any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Attribute) and n.func.attr in
        {'backward','step','zero_grad','save','load_state_dict'}) or
        (isinstance(n.func,ast.Name) and n.func.id in {'print','open','exec','eval'})) for n in ast.walk(node)):
        return 'training control or external side effects are outside a kernel boundary'
    if any(isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(
        isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='self'
        for target in (n.targets if isinstance(n,ast.Assign) else [n.target]) for x in ast.walk(target))
        for n in ast.walk(node)):
        return 'module mutation needs review'
    if args.args and args.args[0].arg=='self':
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and isinstance(n.func.value,ast.Name) and n.func.value.id=='self' for n in ast.walk(node)):
            return 'module composition needs a smaller tensor-function boundary'
    # Nested closures cannot be reconstructed reliably by the existing reference capturer.
    if any(isinstance(n,ast.FunctionDef) and n is not node and node in list(ast.walk(n))
           for n in ast.walk(item['tree'])):
        return 'nested closure needs review'
    return None


def propose(root,item,evidence):
    reason=eligibility(item)
    if reason:raise ValueError(reason)
    source=item['source'];tree=item['tree'];node=item['node']
    names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    names|={n.arg for n in ast.walk(tree) if isinstance(n,ast.arg)}
    names|={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    names|={n.asname or n.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.alias)}
    existing=next((a.asname or a.name for n in tree.body if isinstance(n,ast.Import)
                   for a in n.names if a.name in ('flint', 'flint_gpu', 'muse_kernel')),None)
    alias=existing or 'flint'
    while not existing and alias in names:alias='_'+alias
    lines=source.splitlines(keepends=True)
    indent=lines[node.lineno-1][:node.col_offset]
    lines.insert(node.lineno-1,indent+'@'+alias+'.fast\n')
    if not existing:
        after=0
        for n in tree.body:
            if (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) or (
                isinstance(n,ast.ImportFrom) and n.module=='__future__'):
                after=n.end_lineno
            else:break
        # Keep shebang and encoding/comment header before the import.
        if after==0:
            while after<len(lines) and (lines[after].startswith('#') or not lines[after].strip()):after+=1
        lines.insert(after,'import flint\n' if alias=='flint' else f'import flint as {alias}\n')
    after=''.join(lines);ast.parse(after)
    patch=''.join(difflib.unified_diff(source.splitlines(True),after.splitlines(True),
                                      fromfile='a/'+item['file'],tofile='b/'+item['file']))
    return {'schema':1,'root':str(Path(root).resolve()),'file':item['file'],'function':item['function'],
            'line':node.lineno,'before_sha256':digest(source),'after_sha256':digest(after),
            'before':source,'after':after,'diff':patch,'evidence':evidence,
            'id':digest(item['file']+digest(source)+str(node.lineno))[:16]}


def enclosing_region(item):
    """Find a conservative pure statement block around a profiled helper call.

    Only an entire existing if/else block is extracted, with one live output and no
    self access, mutation, returns, loops, or training control. This preserves constraints
    established by nearby tensor transformations instead of guessing input value domains.
    """
    tree=item['tree'];global_names=set()
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):global_names.add(n.name)
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            global_names.update(a.asname or a.name.split('.')[0] for a in n.names)
    global_names|={'None','True','False'}
    target=item['function'];candidates=[]
    for fn in ast.walk(tree):
        if not isinstance(fn,ast.FunctionDef) or fn is item['node']:continue
        for conditional in ast.walk(fn):
            if not isinstance(conditional,ast.If):continue
            for block in (conditional.body,conditional.orelse):
                if len(block)<3:continue
                nodes=[x for statement in block for x in ast.walk(statement)]
                if not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id==target for x in nodes):continue
                if any(isinstance(x,(ast.For,ast.While,ast.Return,ast.Raise,ast.Try,ast.With,ast.Yield,ast.AugAssign,ast.NamedExpr)) for x in nodes):continue
                if any(isinstance(x,ast.Name) and x.id=='self' for x in nodes):continue
                if any(isinstance(x,ast.Assign) and any(not isinstance(t,ast.Name) for t in x.targets) for x in nodes):continue
                calls=[x for x in nodes if isinstance(x,ast.Call)]
                if any((isinstance(x.func,ast.Attribute) and (x.func.attr.endswith('_') or x.func.attr in {'backward','step','zero_grad'}))
                       or (isinstance(x.func,ast.Name) and x.func.id not in global_names) for x in calls):continue
                defined=set();inputs=[]
                for statement in block:
                    for x in ast.walk(statement):
                        if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load) and x.id not in defined|global_names and x.id not in inputs:inputs.append(x.id)
                    defined|={x.id for x in ast.walk(statement) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)}
                later={x.id for x in ast.walk(fn) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load) and x.lineno>block[-1].end_lineno}
                outputs=sorted(defined&later)
                if len(outputs)!=1 or not inputs:continue
                candidates.append({'start':block[0].lineno,'end':block[-1].end_lineno,'indent':block[0].col_offset,
                                   'inputs':inputs,'output':outputs[0],'caller':fn.name})
    return min(candidates,key=lambda x:x['end']-x['start']) if candidates else None


def propose_region(root,item,evidence,region=None):
    region=region or enclosing_region(item)
    if region is None:return propose(root,item,evidence)
    source=item['source'];tree=item['tree'];lines=source.splitlines(True)
    names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}|{n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    helper=item['function']+'_region'
    while helper in names:helper='_'+helper
    start,end,indent=region['start'],region['end'],region['indent']
    body=''.join('    '+line[indent:] if line.strip() else line for line in lines[start-1:end])
    definition=f'\ndef {helper}('+', '.join(region['inputs'])+'):\n'+body+'    return '+region['output']+'\n\n'
    call=' '*indent+region['output']+' = '+helper+'('+', '.join(region['inputs'])+')\n'
    lines[start-1:end]=[call]
    # Insert before the first top-level class/function containing the original block.
    enclosing=next(n for n in tree.body if n.lineno<=start<=n.end_lineno)
    insertion=min([enclosing.lineno,*[x.lineno for x in getattr(enclosing,'decorator_list',[])]])-1
    lines.insert(insertion,definition);extracted=''.join(lines)
    updated=ast.parse(extracted);node=next(n for n in updated.body if isinstance(n,ast.FunctionDef) and n.name==helper)
    helper_item={**item,'source':extracted,'tree':updated,'node':node,'function':helper,'line':node.lineno}
    proposal=propose(root,helper_item,evidence)
    proposal.update(kind='extract_region',region=region,profiled_function=item['function'],profiled_line=item['line'],
                    before=source,before_sha256=digest(source),
                    id=digest(item['file']+digest(source)+str(start)+'region')[:16])
    proposal['diff']=''.join(difflib.unified_diff(source.splitlines(True),proposal['after'].splitlines(True),
                                               fromfile='a/'+item['file'],tofile='b/'+item['file']))
    proposal['evidence']={**evidence,'region_note':'The profiled helper lies inside this extracted block. Its measured share is a lower bound on region work, not a separately measured region total.'}
    return proposal


def apply_proposal(proposal,root=None):
    root=Path(root or proposal['root']).resolve();path=root/proposal['file']
    if path.is_symlink() or not path.resolve().is_relative_to(root) or path.suffix!='.py':
        raise ValueError('Proposal must name a regular Python source inside the project')
    before=path.read_text()
    if digest(before)!=proposal['before_sha256']:raise ValueError('Source changed; rediscover before applying')
    for relative,expected_hash in proposal.get('workload_sources',{}).items():
        dependency=(root/relative).resolve()
        if not dependency.is_relative_to(root) or digest(dependency.read_text())!=expected_hash:
            raise ValueError('Profiled workload source changed; rediscover before applying')
    # Reconstruct the only permitted edit, rather than trusting an arbitrary supplied patch.
    region=proposal.get('kind')=='extract_region'
    item=next((i for i in functions(root).values() if i['file']==proposal['file']
               and i['line']==proposal.get('profiled_line',proposal['line'])
               and i['function']==proposal.get('profiled_function',proposal['function'])),None)
    if item is None:raise ValueError('Proposed function is no longer present')
    expected=propose_region(root,item,proposal.get('evidence',{})) if region else propose(root,item,proposal.get('evidence',{}))
    if expected['after']!=proposal['after'] or expected['after_sha256']!=proposal['after_sha256']:
        raise ValueError('Proposal changes more than the reviewed import and decorator')
    from .optimize_worker import atomic
    folder=root/'.flint';folder.mkdir(exist_ok=True)
    if (folder/'optimization-owner.json').exists():raise RuntimeError('An optimization session owns this project')
    atomic(folder/'qualification.json',{'accepted':False,'reason':'source proposal applied; requalification required'})
    backup=folder/'edits'/expected['id'];backup.mkdir(parents=True,exist_ok=False)
    (backup/'before.py').write_text(before);atomic(backup/'proposal.json',expected)
    temp=path.with_name(path.name+'.flint-tmp')
    with temp.open('x') as f:f.write(expected['after'])
    temp.chmod(path.stat().st_mode);temp.replace(path)
    atomic(folder/'accepted-proposal.json',{'id':expected['id'],'file':expected['file'],
                                          'after_sha256':expected['after_sha256']})
    return {'applied':expected['id'],'file':str(path),'backup':str(backup)}
