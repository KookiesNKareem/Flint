"""Work boundaries independent of profiling, replay and code generation."""
import functools
import threading
import time

_LISTENERS=[]
_HOOK=None
_LOCAL=threading.local()


def subscribe(callback, kind='training'):
    global _HOOK
    entry=(kind,callback);_LISTENERS.append(entry)
    if kind=='training' and _HOOK is None:
        from torch.optim.optimizer import register_optimizer_step_post_hook
        _HOOK=register_optimizer_step_post_hook(lambda *args:emit('training'))
    def remove():
        global _HOOK
        if entry in _LISTENERS:_LISTENERS.remove(entry)
        if _HOOK is not None and not any(k=='training' for k,_ in _LISTENERS):
            _HOOK.remove();_HOOK=None
    return remove


def emit(kind, seconds=None):
    for selected,callback in tuple(_LISTENERS):
        if selected==kind:callback(kind,seconds)


def workload(fn=None, *, kind='batch'):
    """Optional boundary for a Python batch/request function around its existing compiled model.

    This identifies completion; it does not authorize replay or declare side effects safe.
    Nested wrappers emit only the outer boundary. Training uses optimizer hooks automatically.
    """
    if fn is None:return lambda f:workload(f,kind=kind)
    if kind!='batch':raise ValueError('Only batch work-unit wrappers are currently supported')
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        depth=getattr(_LOCAL,'depth',0);_LOCAL.depth=depth+1;start=time.monotonic()
        try:result=fn(*args,**kwargs)
        finally:_LOCAL.depth=depth
        if depth==0:emit(kind,time.monotonic()-start)
        return result
    return wrapped


class Measurements:
    """Synchronized work-unit timings only inside controlled qualification replays."""
    def __init__(self,kind):
        self.kind=kind;self.last=None;self.seconds=[];self.remove=None

    def start(self):
        self.remove=subscribe(self.boundary,self.kind);return self

    def boundary(self,kind,seconds):
        import torch
        from .device import synchronize
        synchronize()
        now=time.perf_counter()
        if self.last is not None:self.seconds.append(now-self.last)
        self.last=now

    def stop(self):
        if self.remove:self.remove();self.remove=None

    def report(self):
        import statistics
        # Drop the first completed unit plus two intervals for warmup/recompilation.
        steady=self.seconds[2:]
        return {'kind':self.kind,'seconds':steady,'median_seconds':statistics.median(steady) if len(steady)>=5 else None,
                'synchronized':True,'discarded_initial_units':3}
