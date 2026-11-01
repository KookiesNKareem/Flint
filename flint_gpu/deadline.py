"""Portable process-group deadline for local macOS and remote Linux workers."""
import os
import signal
import subprocess
import sys


def main():
    seconds=float(sys.argv[1]);command=sys.argv[2:]
    def terminated(signum,frame):raise KeyboardInterrupt('worker deadline interrupted')
    signal.signal(signal.SIGTERM,terminated)
    child=subprocess.Popen(command,start_new_session=True)
    try:code=child.wait(timeout=seconds)
    except BaseException:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait();raise
    raise SystemExit(code)


if __name__=='__main__':main()
