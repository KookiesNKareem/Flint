"""Conservative cache identity. Different hardware/software requires a new check."""
import importlib.metadata
import platform
import torch

CACHE_VERSION = 6


def metal_device_name():
    # No subprocess per decorator; query the host without initializing another runtime.
    import ctypes
    try:
        libc=ctypes.CDLL(None);size=ctypes.c_size_t()
        key=b'machdep.cpu.brand_string'
        if libc.sysctlbyname(key,None,ctypes.byref(size),None,0)!=0:return platform.machine()
        buffer=ctypes.create_string_buffer(size.value)
        if libc.sysctlbyname(key,buffer,ctypes.byref(size),None,0)!=0:return platform.machine()
        return buffer.value.decode()
    except (AttributeError,OSError):return platform.machine()


def environment():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpu.append({'name': p.name, 'capability': [p.major, p.minor], 'memory': p.total_memory})
    backend='cuda' if gpu else 'cpu'
    if not gpu and torch.backends.mps.is_available():
        model=metal_device_name()
        gpu.append({'name':model or platform.machine(),'backend':'metal',
                    'memory':torch.mps.recommended_max_memory(),'macos':platform.mac_ver()[0]})
        backend='mps'
    driver = None
    if backend=='cuda':
        import subprocess
        try:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                   capture_output=True, text=True, check=True, timeout=3)
            driver = sorted(set(probe.stdout.strip().splitlines()))
        except (OSError, subprocess.SubprocessError):
            pass
    return {'backend':backend,'driver': driver, 'cache_version': CACHE_VERSION, 'python': platform.python_version(),
            'torch': str(torch.__version__), 'triton': version('triton'), 'cuda': torch.version.cuda,
            'platform': platform.system(), 'gpu': gpu}


def signature(args):
    def describe(a):
        if isinstance(a, torch.Tensor):
            return {'shape': list(a.shape), 'dtype': str(a.dtype), 'device': str(a.device),
                    'stride': list(a.stride()), 'offset': a.storage_offset(), 'requires_grad': a.requires_grad}
        if isinstance(a, tuple):
            return {'tuple': [describe(x) for x in a]}
        if isinstance(a, list):
            return {'list': [describe(x) for x in a]}
        if isinstance(a, dict):
            return {'dict': {k: describe(v) for k, v in a.items()}}
        return {'scalar': a}
    return [describe(a) for a in args]
