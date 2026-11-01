"""Bounded evidence for a particular execution phase, never universal workload confidence."""
from collections import Counter, deque
import math
import statistics
import time


class Observation:
    def __init__(self, window=4, stable_windows=2, max_units=64, max_seconds=30., variation=.35):
        if window < 2 or stable_windows < 2 or max_units < window*stable_windows:
            raise ValueError('Observation needs at least two complete windows')
        self.window=window;self.required=stable_windows;self.max_units=max_units
        self.max_seconds=max_seconds;self.variation=variation;self.started=time.monotonic()
        self.units=0;self.samples=[];self.previous=None;self.stable=0;self.coverage=set();self.state='observing'
        self.history=deque(maxlen=16)

    def add(self, seconds, signatures, now=None):
        if self.state!='observing':return self.state
        now=time.monotonic() if now is None else now
        if self.units==0:self.started=now  # initialization/first compilation is not steady-state observation
        self.units+=1
        keys=set(signatures);new=keys-self.coverage;self.coverage.update(keys)
        if not math.isfinite(seconds) or seconds<=0:
            self.stable=0
        else:self.samples.append(seconds)
        if new:self.stable=0;self.previous=None;self.samples=[seconds] if seconds>0 else []
        if len(self.samples)>=self.window:
            median=statistics.median(self.samples)
            spread=(max(self.samples)-min(self.samples))/max(median,1e-12)
            changed=self.previous is None or abs(median-self.previous)/max(self.previous,1e-12)>self.variation
            self.stable=0 if spread>self.variation else (1 if changed else self.stable+1)
            self.history.append({'units':self.units,'median_host_seconds':median,'spread':spread,'stable_windows':self.stable})
            self.previous=median;self.samples=[]
            if self.stable>=self.required and self.coverage:self.state='sufficient'
        if self.state=='observing' and (self.units>=self.max_units or now-self.started>=self.max_seconds):
            self.state='inconclusive'
        return self.state

    def report(self):
        return {'state':self.state,'units':self.units,'signature_count':len(self.coverage),
                'windows':list(self.history),'basis':'host work-boundary intervals and metadata coverage',
                'scope':'observed signatures and phase only; not a GPU speedup or correctness claim'}
