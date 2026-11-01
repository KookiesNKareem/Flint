"""Small terminal UI; no rendering or threads inside the kernel execution path."""
import os
import shutil
import sys
import threading
import time


def terminal_output():
    return sys.stderr.isatty() and os.environ.get('TERM') != 'dumb' and os.environ.get('FLINT_INTERACTIVE') != '0'


def interactive():
    return terminal_output() and sys.stdin.isatty()


class Progress:
    """An activity bar, not an estimated percentage of model/GPU work."""
    def __init__(self, label):
        self.label=label;self.enabled=terminal_output();self.stop=threading.Event();self.thread=None

    def render(self):
        elapsed=max(0,int(time.monotonic()-self.started));width=16
        position=int((time.monotonic()-self.started)*5)%(2*(width-3))
        position=min(position,2*(width-3)-position)
        bar=' '*position+'==='+' '*(width-position-3)
        line=f'[{bar}] {self.label}  {elapsed//60:02d}:{elapsed%60:02d}'
        columns=shutil.get_terminal_size((80,24)).columns
        sys.stderr.write('\r\033[2K'+line[:max(1,columns-1)]);sys.stderr.flush()

    def __enter__(self):
        if self.enabled:
            self.started=time.monotonic();self.render()
            def animate():
                while not self.stop.wait(.15):self.render()
            self.thread=threading.Thread(target=animate,daemon=True);self.thread.start()
        return self

    def __exit__(self,*exc):
        if self.thread:
            self.stop.set();self.thread.join()
            sys.stderr.write('\r\033[2K');sys.stderr.flush()


def show_tradeoff(view):
    speed=(f"{view['unit_speedup']:.2f}× faster per work unit · " if view.get('unit_speedup') else '')
    extra=view['extra_bytes']
    memory=f'{extra/1024:+.1f} KiB' if abs(extra)<1024**2 else f'{extra/1024**2:+.1f} MiB'
    label='sampled MPS memory; exact peak unknown' if view.get('memory_measurement')=='mps_sampled_allocated' else 'peak memory'
    print(f"{speed}{view['command_speedup']:.2f}× faster command · {memory} {label} ({view['memory_growth']:.2%})",file=sys.stderr,flush=True)


def confirm_tradeoff(view):
    """None means no interactive review; EOF is never an affirmative answer."""
    if not interactive():return None
    show_tradeoff(view)
    while True:
        sys.stderr.write('Use this kernel? [Y/n] ');sys.stderr.flush()
        answer=sys.stdin.readline()
        if answer=='':
            sys.stderr.write('\n');sys.stderr.flush();return None
        answer=answer.strip().lower()
        if answer in ('','y','yes'):return True
        if answer in ('n','no'):return False
        print('Enter y or n.',file=sys.stderr,flush=True)
