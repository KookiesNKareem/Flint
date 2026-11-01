"""Backend primitives used outside accepted kernel calls."""
import contextlib
import threading
import time
import torch


def available():
    if torch.cuda.is_available():return 'cuda'
    if torch.backends.mps.is_available():return 'mps'
    return 'cpu'


def synchronize(device=None):
    kind=torch.device(device or available()).type
    if kind=='cuda':
        if device is None:torch.cuda.synchronize()
        else:torch.cuda.synchronize(device)
    elif kind=='mps':torch.mps.synchronize()


def timeit(fn,reps=5,warm=2,device=None):
    for _ in range(warm):fn()
    synchronize(device);start=time.perf_counter()
    for _ in range(reps):fn()
    synchronize(device)
    return (time.perf_counter()-start)*1000/reps


@contextlib.contextmanager
def preserve_rng():
    mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):yield
    finally:
        if mps is not None:torch.mps.set_rng_state(mps)


class Memory:
    """CUDA allocator peak or explicitly sampled MPS live allocation (not an exact peak).

    Sampling is confined to qualification processes, never steady-state kernel dispatch.
    MPS samples may miss transient allocations, so they require explicit admission review.
    """
    def __init__(self,device=None,interval=.001):
        self.device=torch.device(device or available()).type
        self.interval=interval;self.maximum=0;self.samples=0;self.thread=None
        self.stop=threading.Event()

    def sample(self):
        if self.device=='mps':
            self.maximum=max(self.maximum,torch.mps.current_allocated_memory());self.samples+=1

    def start(self):
        synchronize(self.device)
        if self.device=='cuda':
            for index in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(index)
        elif self.device=='mps':
            self.sample()
            def monitor():
                while not self.stop.wait(self.interval):self.sample()
            self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
        return self

    def finish(self):
        if self.thread:self.stop.set();self.thread.join()
        synchronize(self.device);self.sample()
        if self.device=='cuda':
            return {'peak_allocated_bytes':sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())),
                    'memory_measurement':{'kind':'cuda_allocator_peak','exact':True}}
        if self.device=='mps':
            return {'peak_allocated_bytes':self.maximum,'memory_measurement':{
                'kind':'mps_sampled_allocated','exact':False,'samples':self.samples,'interval_seconds':self.interval}}
        return {'peak_allocated_bytes':None,'memory_measurement':{'kind':'unavailable','exact':False}}
