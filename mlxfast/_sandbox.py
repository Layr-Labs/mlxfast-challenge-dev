"""Sandboxed re-run of transform.py for provenance verification.

The harness re-runs the participant's `transform.py` in a clean
sandbox to verify that the `weights/` directory is a deterministic
function of `reference_weights/`. The sandbox:

  - Forbids network access (socket.connect / socket.bind)
  - Forbids clock reads (time.*, datetime.datetime.now/utcnow)
  - Forbids environment variable reads (env is cleared before the script)
  - Forbids reads from anything outside `reference_weights/` OR the
    Python installation (stdlib + site-packages). Import reads are
    permitted so that `transform.py` can use numpy, mlx, etc. — these
    libraries must already be installed and are not under participant
    control, so allowing their reads does not compromise provenance.
  - Forbids writes to anything outside `weights/`
  - Forbids subprocess spawning (subprocess.Popen, os.system)
  - Forbids process forking (os.fork)
  - Forbids loading new native extensions (ctypes.dlopen) — ctypes can
    bypass open() audit events entirely, so we block it at the dlopen
    level. Extensions already loaded at hook-install time are unaffected
    because dlopen is not called again for cached modules.
  - Captures the byte-hash of everything written to `weights/`

If the re-run produces a different `weights/` than the participant
submitted, the submission fails the provenance check.

This is implemented via Python's `audit` events (PEP 578), which
work on CPython 3.8+ and are not bypassable from pure-Python user
code. ctypes/cffi can bypass open() events; they are blocked
separately via their own audit events.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


def content_hash(path: Path) -> str:
    """SHA-256 of every file under `path`, sorted by relative path."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            rel = p.relative_to(path)
            h.update(str(rel).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


SANDBOX_AUDIT_SCRIPT = '''
import sys
import os
from pathlib import Path

REFERENCE = Path("{reference}").resolve()
OUTPUT = Path("{output}").resolve()

# Collect allowed read roots BEFORE installing the hook, so that
# sys.path is fully populated (by the interpreter startup) and we
# can enumerate stdlib + site-packages directories.
#
# Empty-string entries and "." mean cwd — the participant's working
# directory, which must NOT be an allowed read source.  We also
# explicitly exclude the resolved cwd itself so that absolute-path
# entries that happen to equal cwd are rejected too.
_CWD = Path(".").resolve()
_ALLOWED_READ_ROOTS = set()
for _sp in sys.path:
    if not _sp or _sp == ".":
        continue
    _candidate = Path(_sp).resolve()
    if _candidate == _CWD or _CWD in _candidate.parents:
        continue
    if _candidate.is_dir():
        _ALLOWED_READ_ROOTS.add(_candidate)

def _is_within(path, parent):
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    return parent in resolved.parents

def _is_allowed_read(path):
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    # Always allow reads from the reference weights.
    if _is_within(path, REFERENCE):
        return True
    # Allow reads from stdlib / site-packages so that `import numpy`,
    # `import mlx`, etc. work inside transform.py.
    for root in _ALLOWED_READ_ROOTS:
        if root in resolved.parents or resolved == root:
            return True
    return False

def _is_write(mode, flags):
    """Return True if the open call is for writing.

    `mode` may be a string (builtins.open) or an integer (os.open flags).
    """
    if isinstance(mode, str):
        return any(c in mode for c in ("w", "a", "x"))
    # os.open passes integer flags: O_RDONLY=0, O_WRONLY=1, O_RDWR=2.
    O_WRONLY = 1
    O_RDWR = 2
    try:
        return bool(int(mode) & (O_WRONLY | O_RDWR))
    except (TypeError, ValueError):
        return True  # Unknown mode — treat conservatively as write.

def _audit_hook(event, args):
    if event == "open":
        # args: (path, mode, flags)
        path = args[0] if args else ""
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        if _is_write(mode, flags):
            if not _is_within(path, OUTPUT):
                sys.stderr.write(f"BLOCKED: write to {{path}} outside OUTPUT\\n")
                sys.exit(3)
        else:
            if not _is_allowed_read(path):
                sys.stderr.write(f"BLOCKED: read from {{path}} outside allowed paths\\n")
                sys.exit(3)
    elif event in ("socket.connect", "socket.bind"):
        sys.stderr.write("BLOCKED: network access\\n")
        sys.exit(3)
    elif event in ("os.system", "subprocess.Popen"):
        sys.stderr.write("BLOCKED: subprocess\\n")
        sys.exit(3)
    elif event == "os.fork":
        sys.stderr.write("BLOCKED: fork\\n")
        sys.exit(3)
    elif event in ("os.putenv", "os.unsetenv"):
        sys.stderr.write("BLOCKED: env mutation\\n")
        sys.exit(3)
    elif event == "ctypes.dlopen":
        # ctypes can call arbitrary C code without going through open().
        # Block loading new native libraries. Extensions already loaded
        # before the hook was installed are in ctypes' internal cache and
        # will not trigger dlopen again.
        # Residual risk (A-043): libraries imported before this hook was
        # installed (e.g. numpy, mlx) retain their function pointers and
        # can still call C code.  This is accepted: those libraries are
        # part of the approved Python environment and are not under
        # participant control.
        sys.stderr.write(
            f"BLOCKED: ctypes.dlopen({{args[0] if args else '?'}})\\n"
        )
        sys.exit(3)
    elif event == "ctypes.call_function":
        sys.stderr.write("BLOCKED: ctypes.call_function\\n")
        sys.exit(3)
    elif event == "os.sendfile":
        sys.stderr.write("BLOCKED: os.sendfile\\n")
        sys.exit(3)

sys.addaudithook(_audit_hook)

# Clear the environment so the participant cannot read secrets or
# host-specific paths from it.
os.environ.clear()

# Block only clock functions that return wall-clock time and could be used
# to seed an RNG or introduce non-determinism.  Performance counters and
# sleep() are harmless: they don't affect output, and some internal numpy/
# mlx routines call perf_counter() for logging.
import time as _time
class _FrozenAttr:
    def __getattr__(self, name):
        sys.stderr.write(f"BLOCKED: time.{{name}} access\\n")
        sys.exit(3)
    def __call__(self, *a, **kw):
        sys.stderr.write("BLOCKED: time() call\\n")
        sys.exit(3)
_frozen = _FrozenAttr()
for _name in (
    "time", "time_ns",
    "gmtime", "localtime", "mktime",
):
    try:
        setattr(_time, _name, _frozen)
    except (AttributeError, TypeError):
        pass

# Freeze datetime.datetime so now() / utcnow() / today() are blocked.
import datetime as _datetime
_orig_datetime_cls = _datetime.datetime
class _FrozenDatetime(_orig_datetime_cls):
    @classmethod
    def now(cls, *a, **kw):
        sys.stderr.write("BLOCKED: datetime.datetime.now()\\n")
        sys.exit(3)
    @classmethod
    def utcnow(cls, *a, **kw):
        sys.stderr.write("BLOCKED: datetime.datetime.utcnow()\\n")
        sys.exit(3)
    @classmethod
    def today(cls, *a, **kw):
        sys.stderr.write("BLOCKED: datetime.datetime.today()\\n")
        sys.exit(3)
_datetime.datetime = _FrozenDatetime

import runpy
sys.argv = ["transform.py"]
runpy.run_path("{transform}", run_name="__main__")
'''


def verify_transform(
    transform_path: Path,
    reference_path: Path,
    submitted_weights: Path,
    python: str = sys.executable,
) -> tuple[bool, str]:
    """Re-run transform.py in a sandbox and verify byte-equal output.

    Returns (ok, message). ok=True means the submitted weights are
    a deterministic function of the reference weights.
    """
    reference_path = reference_path.resolve()
    submitted_weights = submitted_weights.resolve()
    transform_path = transform_path.resolve()

    if not transform_path.exists():
        return False, f"transform.py not found at {transform_path}"
    if not reference_path.exists():
        return False, f"reference_weights not found at {reference_path}"
    if not submitted_weights.exists():
        return False, f"submitted weights not found at {submitted_weights}"

    # Hash the submitted weights before we wipe them.
    submitted_hash = content_hash(submitted_weights)

    # Wipe the output directory and re-run.
    sandbox_output = submitted_weights.parent / f".{submitted_weights.name}.sandbox"
    if sandbox_output.exists():
        shutil.rmtree(sandbox_output)
    sandbox_output.mkdir(parents=True)

    audit_script = SANDBOX_AUDIT_SCRIPT.format(
        reference=str(reference_path),
        output=str(sandbox_output),
        transform=str(transform_path),
    )

    proc = subprocess.run(
        [python, "-c", audit_script],
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour hard cap
    )

    if proc.returncode != 0:
        return False, (
            f"transform.py re-run failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )

    sandbox_hash = content_hash(sandbox_output)
    if sandbox_hash != submitted_hash:
        return False, (
            f"transform.py is non-deterministic.\n"
            f"  submitted hash: {submitted_hash[:16]}...\n"
            f"  re-run hash:    {sandbox_hash[:16]}..."
        )

    # Cleanup the sandbox copy.
    shutil.rmtree(sandbox_output)
    return True, submitted_hash
