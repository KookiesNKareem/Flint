"""Bound a rehearsal at completed optimizer updates without editing its training loop."""
class RehearsalComplete(BaseException):
    pass


class Rehearsal:
    def __init__(self,steps=None):
        self.requested=int(steps or 0);self.completed=0;self.hook=None
        if not 0<=self.requested<=10000:raise ValueError('Rehearsal must use 1–10000 optimizer steps')

    def start(self):
        if self.requested:
            from .units import subscribe
            from .config import read_config
            def step(kind,seconds):
                self.completed+=1
                if self.completed>=self.requested:raise RehearsalComplete()
            self.hook=subscribe(step,read_config()['workload_kind'])
        return self

    def stop(self):
        if self.hook:self.hook()

    def report(self):
        return {'requested_steps':self.requested,'completed_steps':self.completed,
                'complete':not self.requested or self.completed==self.requested}
