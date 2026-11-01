"""Run an unchanged training script and record complete-process GPU allocation/dispatch."""
import atexit
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    script,*args=sys.argv[1:]
    sys.argv=[script,*args]
    sys.path.insert(0,str(Path(script).resolve().parent))
    import torch
    from .rehearsal import Rehearsal,RehearsalComplete
    from .compilation import REGISTRY,write_report
    atexit.unregister(write_report)
    from .config import read_config
    from .live import Live
    live=Live(Path(os.environ['FLINT_LAUNCH_REPORT']).parent,os.environ['FLINT_WORKFLOW_OWNER'],read_config()).start() if os.environ.get('FLINT_LIVE')=='1' else None
    from .units import Measurements
    measurements=Measurements(read_config()['workload_kind']).start() if os.environ.get('FLINT_MEASURE_UNITS')=='1' else None
    from .device import Memory
    memory_monitor=Memory().start()
    ok=False
    rehearsal=Rehearsal(os.environ.get('FLINT_REHEARSAL_STEPS')).start()
    try:
        runpy.run_path(script,run_name='__main__')
        ok=True
    except RehearsalComplete:
        ok=True
    except SystemExit as error:
        ok=error.code in (None,0)
        if not ok:raise
    finally:
        rehearsal.stop()
        if measurements:measurements.stop()
        if live:live.stop()
        ok=ok and rehearsal.report()['complete']
        for runtime in REGISTRY.values():
            try:runtime.finish_capture()
            except Exception as error:
                ok=False;runtime.disable('capture finalization: '+str(error))
        artifacts={}
        for runtime in REGISTRY.values():
            counts=runtime.stats
            if counts.get('kernel',0) or counts.get('compiled kernel graphs',0):
                status=runtime.status()
                artifacts[runtime.job.name]=status['kernel_sha256']
        memory=memory_monitor.finish()
        result={'success':ok,'work_units':measurements.report() if measurements else None,'live':live.report() if live else None,'rehearsal':rehearsal.report(),**memory,'artifacts':artifacts,
                'dispatch':[r.status() | {'name':r.fn.__qualname__,'job':str(r.job)} for r in REGISTRY.values()]}
        Path(os.environ['FLINT_LAUNCH_REPORT']).write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
