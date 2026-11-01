import io
import os
import sys
import pytest
from flint_gpu import terminal


class TTY(io.StringIO):
    def isatty(self):return True


VIEW={'unit_speedup':3.79,'command_speedup':2.56,'extra_bytes':131584,'memory_growth':.0045}


@pytest.mark.parametrize('answer,expected',[('\n',True),('Y\n',True),('n\n',False),('',None),('maybe\nno\n',False)])
def test_default_yes_requires_actual_input(monkeypatch,answer,expected):
    stream=TTY();monkeypatch.setattr(sys,'stderr',stream);monkeypatch.setattr(sys,'stdin',TTY(answer))
    monkeypatch.setenv('TERM','xterm');monkeypatch.delenv('FLINT_INTERACTIVE',raising=False)
    assert terminal.confirm_tradeoff(VIEW) is expected
    assert 'Use this kernel? [Y/n]' in stream.getvalue()
    assert '+128.5 KiB' in stream.getvalue()


def test_pipe_never_prompts_or_approves(monkeypatch):
    stream=TTY();monkeypatch.setattr(sys,'stderr',stream);monkeypatch.setattr(sys,'stdin',io.StringIO('\n'))
    assert terminal.confirm_tradeoff(VIEW) is None
    assert not stream.getvalue()


def test_progress_cleans_up_on_failure_and_is_not_a_fake_percentage(monkeypatch):
    stream=TTY();monkeypatch.setattr(sys,'stderr',stream)
    monkeypatch.setenv('TERM','xterm');monkeypatch.delenv('FLINT_INTERACTIVE',raising=False)
    with pytest.raises(ValueError):
        with terminal.Progress('Checking') as bar:raise ValueError('test')
    assert not bar.thread.is_alive()
    assert 'Checking' in stream.getvalue() and '%' not in stream.getvalue()
    assert stream.getvalue().endswith('\r\x1b[2K')


def test_nonterminal_progress_has_no_animation_or_thread(monkeypatch):
    stream=io.StringIO();monkeypatch.setattr(sys,'stderr',stream)
    with terminal.Progress('Checking') as bar:pass
    assert bar.thread is None and stream.getvalue()==''


@pytest.mark.skipif(os.name!='posix',reason='PTY integration is POSIX-only')
def test_real_pty_progress_then_enter_approval():
    import pty,select,subprocess,time
    master,slave=pty.openpty()
    code="from flint_gpu.terminal import Progress,confirm_tradeoff; import time;\nwith Progress('Checking'): time.sleep(.2)\nprint('CHOICE',confirm_tradeoff("+repr(VIEW)+"))"
    proc=subprocess.Popen([sys.executable,'-c',code],stdin=slave,stdout=slave,stderr=slave,env={**os.environ,'TERM':'xterm','FLINT_INTERACTIVE':'1'})
    os.close(slave);output=b'';sent=False;deadline=time.monotonic()+20
    try:
        while time.monotonic()<deadline:
            if select.select([master],[],[],.1)[0]:
                try:chunk=os.read(master,4096)
                except OSError:break
                if not chunk:break
                output+=chunk
                if b'[Y/n]' in output and not sent:os.write(master,b'\n');sent=True
            if proc.poll() is not None and b'CHOICE True' in output:break
        assert proc.wait(timeout=5)==0
        assert b'Checking' in output and b'CHOICE True' in output and sent
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        os.close(master)
