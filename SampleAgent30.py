from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request

_T0 = time.time()

def log(msg: str) -> None:
    el = time.time() - _T0
    sys.stdout.write(f"[{int(el // 60)}:{el % 60:04.1f}] {msg}\n")
    sys.stdout.flush()

MODEL = os.getenv("TG_MODEL") or "openai/gpt-6-luna"
FALLBACK_MODELS = [m.strip() for m in (os.getenv("TG_FALLBACK_MODELS") or "openai/gpt-5.6-luna").split(",")
                   if m.strip()]
REASONING_EFFORT = os.getenv("TG_REASONING") or ""
MAX_OUTPUT_TOKENS = int(os.getenv("TG_MAX_OUTPUT_TOKENS") or 32000)
TEMPERATURE = 0.2

PRICES = {
    "openai/gpt-6-luna": (0.10, 0.50, 0.01),
    "openai/gpt-5.6-luna": (0.20, 1.20, 0.02),
    "deepseek/deepseek-v4.1-flash": (0.03, 0.60, 0.027),
    "z-ai/glm-5.3-flash": (0.15, 0.50, 0.03),
    "minimax/minimax-m3": (0.30, 1.20, 0.06),
    "google/gemini-3.8-flash": (0.75, 3.75, 0.075),
    "deepseek/deepseek-v4-pro": (0.955, 1.911, 0.08),
    "anthropic/claude-sonnet-5": (2.0, 10.0, 0.2),
}

CASE_TIMEOUT = 5.0
MAX_RESULT_CHARS = 8000
MAX_CASE_FILES = 400
MUTANTS_PER_ROUND = 25
PER_FUNCTION = 4
FUNCS_PER_ROUND = 8
WEAK_FUNCTION = 0.5
IDLE_STOP = int(os.getenv("TG_IDLE_STOP") or 4)
RETRY_ROUNDS = int(os.getenv("TG_RETRY_ROUNDS") or 2)
COVERAGE_SWEEPS = int(os.getenv("TG_COVERAGE_SWEEPS") or 1)
EXTRA_LOOKS = 3
EXTRA_LOOK_FRAC = 0.7
SET_ASIDE_LOOK_FRAC = 0.5
SECOND_KINDS_FRAC = 0.6
CHECK_CASES_CAP = 30
SHELL_SECONDS_PER_REPLY = 150
CLOCK_SHIFT = 1e9
FAIL_STREAK_SECONDS = 300
CPU_WORKERS_CAP = 4

SAME_SRC = '''
def _same_value(actual, expected):
    """True when `actual` equals the recorded value `expected` (floats compared closely)."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        if not isinstance(actual, float):
            return False
        if math.isnan(expected):
            return math.isnan(actual)
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(expected, complex):
        return (isinstance(actual, complex) and _same_value(actual.real, expected.real)
                and _same_value(actual.imag, expected.imag))
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, bytes):
        return isinstance(actual, (bytes, bytearray)) and bytes(actual) == expected
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_value(a, e) for a, e in zip(actual, expected)))
    if isinstance(expected, tuple):
        return (isinstance(actual, tuple) and len(actual) == len(expected)
                and all(_same_value(a, e) for a, e in zip(actual, expected)))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and len(actual) == len(expected)
                and all(k in actual and _same_value(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, (set, frozenset)):
        return isinstance(actual, (set, frozenset)) and actual == expected
    return actual == expected
'''

CONFTEST_SRC = '''"""Suite-wide settings: a test that runs far longer than expected fails instead of hanging."""
import signal

import pytest

TIME_LIMIT_SECONDS = 10


class TimeLimitExceeded(BaseException):
    """Raised inside a test that ran longer than TIME_LIMIT_SECONDS."""


_state = {"expired": False}


def _expire(signum, frame):
    _state["expired"] = True
    raise TimeLimitExceeded("test ran longer than %s seconds" % TIME_LIMIT_SECONDS)


@pytest.fixture(autouse=True)
def _time_limit():
    if _state["expired"]:
        pytest.fail("not run: an earlier test exceeded the time limit")
    if not hasattr(signal, "setitimer"):
        yield
        return
    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, TIME_LIMIT_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
'''

_EVAL_NS = {"__builtins__": {}, "float": float, "set": set, "frozenset": frozenset, "complex": complex}

class _Unrecordable(Exception):
    pass

class _CaseTimeout(BaseException):
    pass

def _to_src(v, depth=0):
    if depth > 40:
        raise _Unrecordable("the result is nested too deeply")
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return repr(int(v))
    if isinstance(v, float):
        f = float(v)
        if math.isnan(f):
            return "float('nan')"
        if math.isinf(f):
            return "float('inf')" if f > 0 else "-float('inf')"
        return repr(f)
    if isinstance(v, complex):
        return "complex(%s, %s)" % (_to_src(float(v.real)), _to_src(float(v.imag)))
    if isinstance(v, str):
        return repr(v[:])
    if isinstance(v, (bytes, bytearray)):
        return repr(bytes(v))
    if isinstance(v, tuple):
        items = [_to_src(x, depth + 1) for x in v]
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_to_src(x, depth + 1) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join("%s: %s" % (_to_src(k, depth + 1), _to_src(x, depth + 1)) for k, x in v.items()) + "}"
    if isinstance(v, (set, frozenset)):
        items = sorted(_to_src(x, depth + 1) for x in v)
        if isinstance(v, frozenset):
            return "frozenset({%s})" % ", ".join(items) if items else "frozenset()"
        return "{%s}" % ", ".join(items) if items else "set()"
    if hasattr(v, "__next__"):
        raise _Unrecordable("returned an iterator; wrap it in list()")
    raise _Unrecordable("returned a %s.%s, which is not plain data; return plain data derived from it "
                        "through the public API" % (type(v).__module__, type(v).__qualname__))

class _LineTracer:

    def __init__(self, root):
        self.root = root
        self.lines = set()

    def _global(self, frame, event, arg):
        if frame.f_code.co_filename.startswith(self.root):
            self.lines.add((frame.f_code.co_filename, frame.f_lineno))
            return self._local
        return None

    def _local(self, frame, event, arg):
        if event == "line":
            self.lines.add((frame.f_code.co_filename, frame.f_lineno))
        return self._local

    def start(self):
        self.lines = set()
        threading.settrace(self._global)
        sys.settrace(self._global)

    def stop(self):
        sys.settrace(None)
        threading.settrace(None)
        return sorted([os.path.relpath(f, self.root), n] for f, n in self.lines)

def _fmt_exc(e):
    text = "%s: %s" % (type(e).__name__, e)
    return text if len(text) < 300 else text[:300] + "..."

def _describe_api(mod, limit=16000):
    import inspect
    out = []
    declared = getattr(mod, "__all__", None)
    names = list(declared) if isinstance(declared, (list, tuple)) else [n for n in dir(mod) if not n.startswith("_")]
    pkg_root = mod.__name__.split(".")[0]
    for name in names:
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        omod = getattr(obj, "__module__", "") or ""
        if inspect.ismodule(obj):
            if obj.__name__.split(".")[0] == pkg_root:
                out.append("module %s" % obj.__name__)
            continue
        if declared is None and omod and omod.split(".")[0] != pkg_root:
            continue
        doc = (inspect.getdoc(obj) or "").strip().split("\n")[0][:160]
        if inspect.isclass(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(...)"
            out.append("class %s%s  # %s" % (name, sig, doc))
            for mname, member in sorted(vars(obj).items()):
                if mname.startswith("_") and mname not in ("__call__", "__iter__", "__len__", "__getitem__",
                                                           "__contains__", "__eq__", "__enter__", "__exit__"):
                    continue
                fn = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
                if isinstance(member, property):
                    out.append("    .%s (property)" % mname)
                elif callable(fn):
                    try:
                        msig = str(inspect.signature(fn))
                    except Exception:
                        msig = "(...)"
                    mdoc = (inspect.getdoc(fn) or "").strip().split("\n")[0][:120]
                    out.append("    .%s%s  # %s" % (mname, msig, mdoc))
        elif callable(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(...)"
            out.append("%s%s  # %s" % (name, sig, doc))
        else:
            try:
                r = repr(obj)
            except Exception:
                r = "<%s>" % type(obj).__name__
            out.append("%s = %s" % (name, r if len(r) < 80 else r[:80] + "..."))
        if sum(len(x) for x in out) > limit:
            out.append("... (truncated)")
            break
    return "\n".join(out)

def _public_class_path(cls, roots):
    for klass in cls.__mro__:
        if klass in (BaseException, Exception, object):
            break
        if klass.__module__ == "builtins":
            return klass.__name__
        if klass.__name__.startswith("_"):
            continue
        mods = sorted((name for name in list(sys.modules) if name.split(".")[0] in roots
                       and not any(part.startswith("_") for part in name.split("."))),
                      key=lambda name: (name.count("."), name))
        for name in mods:
            if getattr(sys.modules.get(name), klass.__name__, None) is klass:
                return name + "." + klass.__name__
    return "Exception"

def _resolve_class(path):
    import importlib
    if "." not in path:
        return getattr(__import__("builtins"), path, Exception)
    mod, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(mod), attr)

def _runner_main(cfg_path):
    import importlib
    import inspect
    import pathlib

    with open(cfg_path) as fh:
        cfg = json.load(fh)
    for p in reversed(cfg.get("sys_path", [])):
        sys.path.insert(0, p)
    out = open(cfg["out"], "a", buffering=1)

    def emit(obj):
        out.write(json.dumps(obj) + "\n")

    mode = cfg["mode"]
    if mode == "api":
        for name in cfg["candidates"]:
            info = {"kind": "api", "name": name}
            try:
                m = importlib.import_module(name)
                info["file"] = getattr(m, "__file__", None)
                info["path"] = list(getattr(m, "__path__", []) or [])
                try:
                    ver = getattr(m, "__version__", None)
                except Exception:
                    ver = None
                info["version"] = ver if isinstance(ver, str) else None
                try:
                    info["api"] = _describe_api(m)
                except Exception as e:
                    info["api"] = "(could not list the public names: %s)" % _fmt_exc(e)
            except BaseException as e:
                info["error"] = _fmt_exc(e)
            emit(info)
        emit({"kind": "end"})
        return

    ns = {"math": math}
    exec(SAME_SRC, ns)
    same = ns["_same_value"]
    if cfg.get("clock_shift"):
        _shift_clocks(float(cfg["clock_shift"]))
    if mode == "check":
        for pkg in cfg.get("packages") or ():
            try:
                importlib.import_module(pkg)
            except BaseException as e:
                emit({"kind": "library_error", "error": _fmt_exc(e)})
                emit({"kind": "end"})
                return
    modules = {}
    for name in cfg["modules"]:
        try:
            modules[name] = importlib.import_module(name)
        except BaseException as e:
            emit({"kind": "module_error", "module": name, "error": _fmt_exc(e),
                  "trace": traceback.format_exc()[-1500:], "cause": _import_failure_cause(e, name)})
    tracer = _LineTracer(cfg["trace_root"]) if cfg.get("trace_root") else None
    timeout = float(cfg.get("timeout", CASE_TIMEOUT))
    tmp_base = cfg.get("tmp_base") or tempfile.gettempdir()
    expected = cfg.get("expected") or {}

    def on_alarm(signum, frame):
        raise _CaseTimeout()

    signal.signal(signal.SIGALRM, on_alarm)
    for key in cfg["cases"]:
        mod_name, case_name = key.split("::", 1)
        m = modules.get(mod_name)
        fn = getattr(m, case_name, None) if m is not None else None
        if fn is None:
            if mode == "record":
                emit({"kind": "case", "key": key, "status": "error", "error": "case not found"})
            else:
                emit({"kind": "mismatch", "key": key, "status": "missing"})
                if cfg.get("stop_on_first"):
                    break
            continue
        emit({"kind": "start", "key": key})
        kwargs = {}
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = pathlib.Path(tempfile.mkdtemp(prefix="case-", dir=tmp_base))
        except (TypeError, ValueError):
            pass
        status, value, exc, lines = "value", None, None, None
        t0 = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            if tracer:
                tracer.start()
            try:
                value = fn(**kwargs)
            finally:
                if tracer:
                    lines = tracer.stop()
        except _CaseTimeout:
            status = "timeout"
        except Exception as e:
            status, exc = "raises", e
        except BaseException as e:
            status, exc = "fatal", e
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        secs = time.perf_counter() - t0
        if mode == "record":
            rec = {"kind": "case", "key": key, "status": status, "secs": round(secs, 4)}
            if status == "value":
                try:
                    src = _to_src(value)
                    if len(src) > MAX_RESULT_CHARS:
                        raise _Unrecordable("the result is too large (%d characters); return a smaller "
                                            "summary of it" % len(src))
                    rec["src"] = src
                except _Unrecordable as u:
                    rec["status"], rec["error"] = "error", str(u)
                except Exception as e:
                    rec["status"], rec["error"] = "error", "could not record the result: " + _fmt_exc(e)
            elif status == "raises":
                rec["exc"] = type(exc).__name__
                rec["msg"] = str(exc)[:160]
                try:
                    rec["cls"] = _public_class_path(type(exc), set(cfg.get("packages") or ()))
                except Exception:
                    rec["cls"] = "Exception"

            elif status == "timeout":
                rec["status"], rec["error"] = "error", "took longer than %.0f s" % timeout
            else:
                rec["status"], rec["error"] = "error", "raised %s, which a test cannot catch" % type(exc).__name__
            if lines is not None:
                rec["lines"] = lines
            emit(rec)
        else:
            exp = expected.get(key)
            if exp is None:
                continue
            if exp["status"] == "value":
                ok = status == "value"
                if ok:
                    try:
                        ok = bool(same(value, eval(exp["src"], dict(_EVAL_NS))))
                    except Exception:
                        ok = False
            else:
                ok = status == "raises"
                if ok and exp.get("cls"):
                    try:
                        ok = isinstance(exc, _resolve_class(exp["cls"]))
                    except Exception:
                        ok = False
            if not ok:
                emit({"kind": "mismatch", "key": key, "status": status,
                      "exc": _fmt_exc(exc)[:200] if exc is not None else None})
                if cfg.get("stop_on_first"):
                    break
    emit({"kind": "end"})

class _Shifted:
    __slots__ = ("fn", "d")

    def __init__(self, fn, d):
        self.fn, self.d = fn, d

    def __call__(self):
        return self.fn() + self.d

class _AtNow:
    __slots__ = ("fn", "struct")

    def __init__(self, fn, struct=False):
        self.fn, self.struct = fn, struct

    def __call__(self, t=None):
        if t is None:
            t = time.localtime() if self.struct else time.time()
        return self.fn(t)

class _StrftimeAtNow:
    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, fmt, t=None):
        return self.fn(fmt, time.localtime() if t is None else t)


def _shift_clocks(shift: float) -> None:
    for name, amount in (("monotonic", shift), ("perf_counter", shift), ("time", shift / 10)):
        f = getattr(time, name, None)
        if f is not None:
            setattr(time, name, _Shifted(f, amount))
        f_ns = getattr(time, name + "_ns", None)
        if f_ns is not None:
            setattr(time, name + "_ns", _Shifted(f_ns, int(amount * 1e9)))
    for name in ("localtime", "gmtime", "ctime"):
        f = getattr(time, name, None)
        if f is not None:
            setattr(time, name, _AtNow(f))
    if hasattr(time, "asctime"):
        time.asctime = _AtNow(time.asctime, struct=True)
    if hasattr(time, "strftime"):
        time.strftime = _StrftimeAtNow(time.strftime)

def _import_failure_cause(exc, module_name):
    if isinstance(exc, SyntaxError) and exc.filename and \
            os.path.splitext(os.path.basename(exc.filename))[0] != module_name:
        return "library"
    tb = exc.__traceback__
    own = own_line = None
    while tb is not None:
        code = tb.tb_frame.f_code
        if code.co_name == "<module>" and not code.co_filename.startswith("<"):
            if own is None and os.path.splitext(os.path.basename(code.co_filename))[0] == module_name:
                own, own_line = code.co_filename, tb.tb_lineno
            elif code.co_filename != own:
                return "library"
        tb = tb.tb_next
    if own is None:
        return "library"
    try:
        with open(own, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and \
                    node.lineno <= own_line <= (node.end_lineno or node.lineno):
                return "library"
    except Exception:
        pass
    return "file"

def _swap_code(swap, packages):
    import types
    mod = sys.modules.get(swap["modname"])
    if mod is None or not getattr(mod, "__file__", None):
        return False
    with open(mod.__file__, "rb") as fh:
        code = compile(fh.read(), mod.__file__, "exec", dont_inherit=True)
    target, stack = None, [code]
    while stack and target is None:
        c = stack.pop()
        for k in c.co_consts:
            if isinstance(k, types.CodeType):
                if k.co_name == swap["name"] and k.co_firstlineno == swap["line"]:
                    target = k
                    break
                stack.append(k)
    if target is None:
        return False
    obj = mod
    for part in swap["qual"].split("."):
        d = getattr(obj, "__dict__", None)
        obj = d.get(part) if isinstance(d, dict) or hasattr(d, "get") else None
        if obj is None:
            return False
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if type(obj).__name__ == "cached_property" and callable(getattr(obj, "func", None)):
        obj = obj.func
    funcs = [f for f in (obj.fget, obj.fset, obj.fdel) if f] if isinstance(obj, property) else [obj]
    for f in funcs:
        c0 = getattr(f, "__code__", None)
        if c0 is not None and c0.co_name == swap["name"] and c0.co_firstlineno == swap["line"]:
            if c0.co_freevars != target.co_freevars:
                return False
            f.__code__ = target
            return True
    return False

def _zygote_main(cfg_path):
    import importlib
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    for p in reversed(cfg.get("sys_path", [])):
        sys.path.insert(0, p)
    packages = set(cfg.get("packages") or ())
    called = set()

    def _profile(frame, event, arg):
        if event == "call":
            called.add((frame.f_code.co_filename, frame.f_code.co_firstlineno))
    sys.setprofile(_profile)
    try:
        for name in sorted(packages):
            try:
                importlib.import_module(name)
            except BaseException:
                pass
    finally:
        sys.setprofile(None)
    try:
        import pytest
    except BaseException:
        pass
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            job = json.loads(line)
        except ValueError:
            continue
        pid = os.fork()
        if pid == 0:
            try:
                devnull = os.open(os.devnull, os.O_RDWR)
                os.dup2(devnull, 0)
                os.dup2(devnull, 1)
                sw = job.get("swap")
                swapped = sw == "pristine"
                sw = None if swapped else sw
                mod = sys.modules.get(sw["modname"]) if sw else None
                if sw and mod is not None and (getattr(mod, "__file__", None), sw["line"]) not in called:
                    try:
                        swapped = _swap_code(sw, packages)
                    except BaseException:
                        swapped = False
                if not swapped:
                    for k in [k for k in list(sys.modules) if k.split(".")[0] in packages]:
                        del sys.modules[k]
                _runner_main(job["cfg"])
            except BaseException:
                pass
            os._exit(0)
        deadline = time.time() + float(job.get("wall", 60))
        status = "done"
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.time() > deadline:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                os.waitpid(pid, 0)
                status = "timeout"
                break
            time.sleep(0.003)
        sys.stdout.write(status + "\n")
        sys.stdout.flush()

def _readline(stream, timeout: float):
    import select
    buf = b""
    end = time.time() + timeout
    fd = stream.fileno()
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], max(0.0, min(1.0, end - time.time())))
        if not r:
            continue
        chunk = os.read(fd, 1)
        if not chunk:
            return None
        buf += chunk
        if chunk == b"\n":
            return buf.decode(errors="replace")
    return None

def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + "\n... [%d characters omitted] ...\n" % (len(text) - limit) + text[-(limit - head):]

def _available_cpus() -> int:
    n = os.cpu_count() or 2
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(math.ceil(int(quota) / int(period)))))
    except Exception:
        pass
    return max(1, n)

def _stdlib_names() -> set:
    names = set(getattr(sys, "stdlib_module_names", ()))
    if not names:
        names = {"abc", "argparse", "array", "ast", "asyncio", "base64", "binascii", "bisect", "builtins",
                 "calendar", "cmath", "codecs", "collections", "contextlib", "copy", "csv", "dataclasses",
                 "datetime", "decimal", "difflib", "enum", "errno", "fnmatch", "fractions", "functools",
                 "gc", "glob", "gzip", "hashlib", "heapq", "hmac", "html", "http", "inspect", "io",
                 "ipaddress", "itertools", "json", "keyword", "locale", "logging", "math", "numbers",
                 "operator", "os", "pathlib", "pickle", "pprint", "queue", "random", "re", "shutil",
                 "signal", "statistics", "string", "struct", "sys", "tempfile", "textwrap", "threading",
                 "time", "timeit", "traceback", "types", "typing", "unicodedata", "unittest", "urllib",
                 "uuid", "warnings", "weakref", "xml", "zipfile", "zlib"}
    return names

def _nobody_ids():
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return None
    try:
        import pwd
        pw = pwd.getpwnam("nobody")
        return pw.pw_uid, pw.pw_gid
    except Exception:
        return 65534, 65534

def _run_proc(cmd, *, cwd, env, timeout, as_nobody=False):
    kw = {}
    ids = _nobody_ids() if as_nobody else None
    if ids:
        if sys.version_info >= (3, 9):
            kw.update(user=ids[0], group=ids[1], extra_groups=[])
        else:
            def _drop(uid=ids[0], gid=ids[1]):
                os.setgroups([])
                os.setgid(gid)
                os.setuid(uid)
            kw["preexec_fn"] = _drop
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, **kw)
    except Exception as e:
        return -1, "could not start %s: %s" % (cmd[0], e)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out.decode(errors="replace")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        try:
            out, _ = proc.communicate(timeout=5)
        except Exception:
            out = b""
        return None, (out or b"").decode(errors="replace")

def _transcript(role: str, text: str) -> None:
    path = os.getenv("TG_TRANSCRIPT")
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write("\n\n======== %s (%.0fs) ========\n\n%s\n" % (role, time.time() - _T0, text))
    except OSError:
        pass

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL_REFUSED = re.compile(r"not a valid model|invalid model|unknown model|no endpoints found|does not exist|"
                            r"not (?:in the )?allow|allowed (?:model )?list|allowlist|not permitted|"
                            r"not available|unsupported model|not supported|not found", re.I)

def _inference_urls() -> list:
    urls = []
    proxy = (os.getenv("SANDBOX_PROXY_URL") or "").strip().rstrip("/")
    if proxy:
        urls.append(proxy + "/api/v1/chat/completions")
    base = (os.getenv("GMMT_INFERENCE_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or "").strip().rstrip("/")
    if base:
        urls.append(base + "/chat/completions")
    urls.append(OPENROUTER_CHAT_URL)
    return list(dict.fromkeys(urls))

def _remaining_budget():
    proxy = (os.getenv("SANDBOX_PROXY_URL") or "").strip().rstrip("/")
    if not proxy:
        return None
    try:
        with urllib.request.urlopen(proxy + "/api/v1/usage", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        value = data.get("budget_remaining_usd")
        return float(value) if isinstance(value, (int, float)) else None
    except Exception:
        return None

def _api_key() -> str:
    return (os.getenv("OPENROUTER_API_KEY") or os.getenv("GMMT_INFERENCE_API_KEY") or "").strip()

class _HardTimeout(BaseException):
    pass

class _hard_deadline:

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.active = False
        self.previous = None

    def __enter__(self):
        if threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer"):
            def _fire(signum, frame):
                raise _HardTimeout()
            self.previous = signal.signal(signal.SIGALRM, _fire)
            signal.setitimer(signal.ITIMER_REAL, max(1.0, self.seconds))
            self.active = True
        return self

    def __exit__(self, *exc):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous or signal.SIG_DFL)
        return False

class LLM:

    def __init__(self, model: str, budget: float, deadline: float, summarizer=None):
        self.summarizer = summarizer
        self.models = [model] + [m for m in FALLBACK_MODELS if m != model]
        self.model_i = 0
        self.urls = _inference_urls()
        self.url_i = 0
        self.bad_urls: set = set()
        self.budget = budget
        self.deadline = deadline
        self.spent = 0.0
        self.calls = 0
        self.exhausted = False
        self.refused = False
        self.fail_since = None
        self.messages: list = []
        self.last_call_cost = 0.0

    @property
    def model(self) -> str:
        return self.models[self.model_i]

    def _next_url(self, why: str, retire: bool = False) -> bool:
        if retire:
            self.bad_urls.add(self.urls[self.url_i])
        n = len(self.urls)
        for step in range(1, n + 1):
            i = (self.url_i + step) % n
            if i != self.url_i and self.urls[i] not in self.bad_urls:
                self.url_i = i
                log("[LLM] %s; switching to %s" % (why, self.urls[i]))
                return True
        return False

    def _next_model(self, why: str) -> bool:
        if self.model_i + 1 >= len(self.models):
            return False
        self.model_i += 1
        log("[LLM] %s; switching to model %s" % (why, self.model))
        return True

    def _out_of_budget(self, detail: str) -> bool:
        remaining = _remaining_budget()
        if remaining is not None:
            return remaining < max(0.005, self.last_call_cost)
        low = detail.lower()
        return "budget" in low or "cost" in low or "credit" in low

    def affordable(self) -> bool:
        reserve = max(self.last_call_cost * 1.5, 0.01)
        return (not self.exhausted and self.spent + reserve < self.budget * 0.95
                and time.time() < self.deadline - 30)

    def ask(self, text: str, max_tokens: int = MAX_OUTPUT_TOKENS):
        if not self.affordable():
            return None, None
        self.messages.append({"role": "user", "content": text})
        self._trim()
        _transcript("user", text)
        reply, finish = self._call(max_tokens)
        if reply is None:
            self.messages.pop()
            return None, None
        _transcript("assistant", reply)
        self.messages.append({"role": "assistant", "content": reply})
        return reply, finish

    def _trim(self, limit_chars: int = 600000) -> None:
        total = sum(len(m["content"]) for m in self.messages)
        if total <= limit_chars or len(self.messages) < 9:
            return
        head = self.messages[:2]
        tail = self.messages[-5:]
        while tail and tail[0]["role"] != "user":
            tail = tail[1:]
        summary = "(Earlier work in this conversation is summarised here.)"
        if self.summarizer:
            try:
                summary = self.summarizer()
            except Exception:
                pass
        self.messages = head + [{"role": "assistant", "content": summary}] + tail
        log("[LLM] conversation compacted from %d to %d characters" % (
            total, sum(len(m["content"]) for m in self.messages)))

    def _payload_messages(self):
        out = []
        n = len(self.messages)
        for i, m in enumerate(self.messages):
            if m["role"] in ("system", "user") and (i <= 1 or i == n - 1):
                out.append({"role": m["role"], "content": [
                    {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def _call(self, max_tokens: int):
        self.refused = False
        reply, finish = self._attempts(max_tokens)
        if reply is not None:
            self.fail_since = None
        elif not self.exhausted:
            now = time.time()
            if self.refused:
                log("[LLM] every endpoint and model refused the request; no more model calls")
                self.exhausted = True
            elif self.fail_since is None:
                self.fail_since = now
            elif now - self.fail_since > FAIL_STREAK_SECONDS:
                log("[LLM] calls have failed for %.0f s; no more model calls" % (now - self.fail_since))
                self.exhausted = True
        return reply, finish

    def _attempts(self, max_tokens: int):
        key = _api_key()
        if not key:
            log("[LLM] no API key")
            self.exhausted = True
            return None, None
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        failures = empty = hops = 0
        refused = set()
        while failures < 4:
            remaining = self.deadline - time.time()
            if remaining < 30:
                return None, None
            payload = {"model": self.model, "messages": self._payload_messages(), "max_tokens": max_tokens,
                       "temperature": TEMPERATURE, "usage": {"include": True}}
            if REASONING_EFFORT:
                payload["reasoning"] = {"effort": REASONING_EFFORT}
            body = json.dumps(payload).encode()
            timeout = max(20.0, min(300.0, remaining - 15))
            t0 = time.time()
            try:
                req = urllib.request.Request(self.urls[self.url_i], data=body, headers=headers, method="POST")
                with _hard_deadline(timeout + 5):
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode(errors="replace")[:400]
                except Exception:
                    detail = ""
                low = detail.lower()
                about_model = ("model" in low or self.model.lower() in low) and bool(_MODEL_REFUSED.search(low))
                if e.code == 402 or (e.code == 429 and self._out_of_budget(detail)):
                    log("[LLM] budget exhausted (HTTP %d): %s" % (e.code, detail[:200]))
                    self.exhausted = True
                    return None, None
                log("[LLM] HTTP %d from %s: %s" % (e.code, self.urls[self.url_i], detail[:200]))
                if e.code in (400, 403, 404) and about_model:
                    if self._next_model("model %s refused" % self.model):
                        refused.clear()
                        continue
                    self.refused = True
                    return None, None
                if e.code in (404, 405):
                    if self._next_url("endpoint answered HTTP %d" % e.code, retire=True):
                        continue
                    self.refused = True
                    return None, None
                if e.code in (401, 403):
                    refused.add(self.url_i)
                    hops += 1
                    usable = [i for i, u in enumerate(self.urls) if u not in self.bad_urls]
                    if hops <= 2 * len(self.urls) * len(self.models):
                        if any(i not in refused for i in usable) and \
                                self._next_url("endpoint answered HTTP %d" % e.code):
                            continue
                        if self._next_model("request refused by every endpoint"):
                            refused.clear()
                            continue
                    self.refused = True
                    return None, None
                if e.code in (408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529):
                    failures += 1
                    time.sleep(min(20, 2 * failures ** 2))
                    continue
                return None, None
            except _HardTimeout:
                log("[LLM] call exceeded %.0f s" % timeout)
                failures += 1
                continue
            except Exception as e:
                log("[LLM] request to %s failed: %s" % (self.urls[self.url_i], _fmt_exc(e)))
                failures += 1
                slow = isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)
                if not slow and isinstance(e, OSError) and self._next_url("endpoint unreachable"):
                    continue
                time.sleep(min(20, 2 * failures ** 2))
                continue
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            if not isinstance(cost, (int, float)):
                pin, pout, pcache = PRICES.get(self.model, (5.0, 25.0, 0.5))
                prompt = usage.get("prompt_tokens") or 0
                cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
                comp = usage.get("completion_tokens") or 0
                cost = ((prompt - cached) * pin + cached * pcache + comp * pout) / 1e6
            self.spent += float(cost)
            self.last_call_cost = float(cost)
            self.calls += 1
            log("[LLM] call %d: %.1fs, in=%s out=%s cost=$%.4f total=$%.4f finish=%s provider=%s" % (
                self.calls, time.time() - t0, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                cost, self.spent, choice.get("finish_reason"), data.get("provider")))
            if not text.strip():
                empty += 1
                if data.get("error"):
                    log("[LLM] error in the response: %s" % str(data.get("error"))[:200])
                if empty < 3:
                    continue
                return None, None
            return text, choice.get("finish_reason")
        return None, None

SYSTEM_PROMPT = """\
You are an expert Python test engineer. You write regression test suites that pin down the observable
behaviour of an existing library precisely: any change to the covered behaviour, however small, must make a
test fail, while a refactoring that keeps the behaviour unchanged must keep every test passing.

You do not write pytest files yourself. You write *case functions*. The tooling runs every case against the
library as it is today, records its exact result, and turns each case into a pytest test that asserts that
result. You never type an expected value; you choose what to run and what to observe.

## Case files

Write case files as fenced blocks whose info string is `python` followed by the file name:

```python cases_parsing.py
from mylib import parse, Parser        # the package under test, pytest and the standard library only

WORDS = ["alpha", "beta"]              # module level: imports, literal constants, helpers only

def case_parse_plain_number():
    return parse("12")

def case_parser_keeps_state_between_calls():
    p = Parser(strict=True)
    first = p.feed("a=1")
    second = p.feed("b=2")
    return [first, second, p.count]

def case_parse_rejects_empty_input():
    return parse("")                   # when a case raises, its test asserts that the call raises
```

Rules:
- A function named `case_<name>` is one test. It takes no arguments (or only `tmp_path`, a fresh
  pathlib.Path directory) and its last statement is its only `return <expression>`.
- Return plain data only: None, bool, int, float, str, bytes, and lists, tuples, dicts and sets of those.
  Turn library objects into plain data through their public API (attributes, methods, list(), sorted()).
  Return several observations at once as a tuple, list or dict when they belong together.
- A case whose body raises is recorded as "raises an exception", whatever the class. Where the task says
  exception classes are part of the contract, put those cases in their own case file whose first code line
  is `EXCEPTION_CLASSES = True`; every raising case in such a file asserts the exact exception class it
  raises today:

  ```python cases_errors.py
  EXCEPTION_CLASSES = True
  from mylib import parse

  def case_parse_rejects_bytes():
      return parse(b"12")
  ```

  Keep raising cases whose class is not part of the contract in files without that line.
- Import only the package under test (public names only: nothing starting with an underscore, and only from
  the modules the task allows), pytest and the standard library. Never run library code when the file is
  imported: no module-level calls, and no module-level class that subclasses or is decorated with library
  objects. Define such classes inside a helper function that the cases call.
- Keep every case deterministic, independent and fast (well under 0.2 s): no real clock, sleeping,
  randomness, network, environment variables, file paths of the library, object ids or process state;
  restore any global state a case changes before it returns. Never compute a value from a clock reading
  the library took (such as `obj.started_at + 1.5`): float rounding then depends on the machine; set
  such attributes to fixed numbers first. Never return anything that contains the current date or time
  (a log line with a time field, say); pass explicit times where the API accepts them.
- Never return error messages, str()/repr() of exceptions or library objects, or anything the task lists
  as not part of the contract. Where ordering is not part of the contract, return sorted data.
- Randomness: exact random draws are never part of a contract, but ranges, bounds and defaults are. Seed
  the random module inside the case (restore its state before returning) and return properties that hold
  for every draw, such as the minimum and maximum of a few hundred draws.

Group cases by topic, about 20-60 cases per file. To add cases later, write a NEW file (e.g.
`cases_parsing_2.py`). To fix dropped cases, write corrected versions of just those cases into a new file
too; the good cases of the old file stay. Only when a whole file is reported NOT USABLE, send the complete
corrected file again under the same name.

## Shell

Use ```bash blocks to read code or try things out. Each block runs in a fresh shell in the repository
root with a 60 s limit; long output is truncated. Do not modify the repository.

After each of your replies the tooling reports shell output, what it recorded, and problems to fix.
"""

AUTHOR_TASK = """\
Write the regression cases for the scope described in the task above.

1. Read what you still need (the main in-scope modules are included above; use the shell for the rest,
   including the repository's own tests and docs, which are good sources of behaviours and edge cases).
2. Write case files covering the whole scope: every public function, class, method, parameter and option
   the task lists; each parameter's default and non-default values; boundaries and edge cases (empty,
   single element, zero, negative, very large, unicode, None where accepted); documented invalid inputs;
   combinations of options that interact; sequences of calls where state matters; whether inputs are left
   unmodified.
3. Choose inputs that make different code paths give different results: asymmetric values, values just
   below, at and above every threshold, inputs that exercise every branch and flag.

{size}
Coverage and mutation feedback follows afterwards. Once the files are written and the reported
problems are fixed, reply with DONE on its own line.
"""

SMALL_SCOPE = """Aim for broad coverage now: typically 120-300 cases in total over several files, written in one or two
replies."""

LARGE_SCOPE = """This scope is large ({n} named items): aim for about {lo}-{hi} cases in total over several files, written
over several replies, one part of the scope per reply (at most about 150 cases per reply), until every item has
cases."""

ROUND_MODE = """\
From here on you work in short rounds. Each round lists small changes to the library that the current
cases do not detect, or code they never run. Write NEW case files for them (never reuse an existing file
name), or SKIP changes that cannot alter in-scope behaviour. The cases written so far are saved and recorded.
"""

THIRD_LOOK = """\
Third look: the cases written for these changes so far still return their recorded results. For each change, work
out what the changed line computes or stores and where that is used later: a later call on the same object, a later
step of a loop, a second visit of the same object or key, a cache, a value shared between two results. Build a case
in which that later use decides the result, and return what differs (values, `is` identity between results, how
often a callback ran, the order of events). SKIP a change only when no input through the public API can make it
matter."""

MUTATION_INTRO = """\
Mutation analysis of the current suite ({tried} changes tried so far: {killed} detected, {alive} undetected).
Each item below is a small deliberate change to the library source that none of your cases notices: every case
still returns its recorded result. Items are grouped by function. For each item, find inputs that reach the changed line through the in-scope
public API and whose result differs because of the change, and write new cases for them in a new case file.
The best case makes the changed line decide the result: a value exactly at a changed boundary, an input for
which the removed statement matters, an option combination that takes the changed branch.
If a change cannot alter in-scope behaviour that the contract covers (it only affects excluded details, or it
is equivalent), put its id on a line `SKIP: m12, m15` and write nothing for it."""

class Run:
    def __init__(self, statement: str):
        self.statement = statement
        self.t_start = _T0
        raw = (os.getenv("AGENT_TIMEOUT") or "").strip()
        try:
            wall = float(raw) if raw else 0.0
        except ValueError:
            wall = 0.0
        self.wall = wall if wall > 0 else 1800.0
        self.reserve = max(75.0, min(150.0, 0.07 * self.wall))
        self.deadline = self.t_start + self.wall - self.reserve
        budget = 0.29
        try:
            budget = float(os.getenv("GMMT_MAX_COST_USD") or budget)
        except ValueError:
            pass
        self.budget = budget
        self.py = sys.executable or shutil.which("python3") or "python3"
        self.stdlib = _stdlib_names()
        self.work = tempfile.mkdtemp(prefix="tg-")
        os.chmod(self.work, 0o755)
        self.run_counter = 0
        self.counter_lock = threading.Lock()
        self.repo = ""
        self.test_rel = "tests"
        self.suite_limit = 30.0
        self.packages: list = []
        self.src_root = os.path.join(self.work, "src")
        self.cases_dir = os.path.join(self.work, "cases")
        self.runner_file = os.path.join(self.work, "tg_runner.py")
        self.record_dir = os.path.join(self.work, "record")
        self.test_names: dict = {}
        self.extra_path: list = []
        self.scope_files: list = []
        self.initial_test_entries = None
        self.case_sources: dict = {}
        self.module_problems: dict = {}
        self.case_problems: dict = {}
        self.records: dict = {}
        self.case_order: dict = {}
        self.excluded: dict = {}
        self.best_files: dict = {}
        self.notes: list = []
        self.llm = LLM(MODEL, self.budget, self.deadline, summarizer=self.conversation_summary)
        self.cov_lines: dict = {}
        self.line_cases: dict = {}
        self.mutants: dict = {}
        self.mutant_keys: set = set()
        self.mutant_status: dict = {}
        self.shown: set = set()
        self.skipped: set = set()
        self.retried: set = set()
        self.context_files: set = set()
        self.round_files: set = set()
        self.zygotes: dict = {}
        self.round_context = ""
        self.func_feedback: dict = {}
        self.mutant_full: dict = {}
        self.pool_cv = threading.Condition()
        self.pool_queue: list = []
        self.pool_seq = 0
        self.pool_threads: list = []
        self.pool_stop = False
        self.pool_paused = False
        self.pool_busy = 0
        self.worker_roots: list = []
        self.import_fragile: dict = {}
        self.unreliable: set = set()
        self.error_retried: set = set()
        self.doubted: set = set()
        self.doubt_shown: set = set()
        self.func_rounds: dict = {}
        self.kinds = 1
        self.alone_ok: dict = {}
        self.import_fragile_done: set = set()
        self.repo_baseline: dict = {}

    def left(self) -> float:
        return self.deadline - time.time()

    def end_left(self) -> float:
        return self.t_start + self.wall - 25 - time.time()

    def frac(self) -> float:
        return (time.time() - self.t_start) / self.wall

    def setup(self) -> None:
        st = self.statement
        m_repo = re.search(r"repository at `(/[^`]+)`", st)
        m_tests = re.search(r"[Aa]dd files only under `([^`]+)`", st) or re.search(r"only under `([^`]+/)`", st)
        asked_repo = asked_tests = None
        if not m_repo or not m_tests:
            asked_repo, asked_tests = self.ask_layout()
        candidates = []
        if m_repo:
            candidates.append(m_repo.group(1).rstrip("/"))
        elif asked_repo and os.path.isabs(asked_repo):
            candidates.append(asked_repo)
        candidates += [os.getcwd(), "/app", "/repo", "/testbed", "/workspace"]
        for c in candidates:
            if c and os.path.isdir(c) and (os.path.isdir(os.path.join(c, ".git")) or c == candidates[0]):
                self.repo = os.path.realpath(c)
                break
        if not self.repo:
            self.repo = os.path.realpath(os.getcwd())
        p = (m_tests.group(1) if m_tests else asked_tests or "").strip().rstrip("/")
        if os.path.isabs(p):
            p = os.path.relpath(p, self.repo) if p.startswith(self.repo + "/") else os.path.basename(p)
        if p and p != "." and not p.startswith(".."):
            self.test_rel = p
        self.snapshot_repo()
        m = re.search(r"finish in under (\d+(?:\.\d+)?) seconds", st)
        if m:
            self.suite_limit = float(m.group(1))
        log("[SETUP] repo=%s tests=%s suite_limit=%.0fs wall=%.0fs budget=$%.2f model=%s" % (
            self.repo, self.test_rel, self.suite_limit, self.wall, self.budget, MODEL))
        for d in (self.src_root, self.cases_dir, self.record_dir):
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o755)
        shutil.copyfile(os.path.abspath(__file__), self.runner_file)
        os.chmod(self.runner_file, 0o644)
        tdir = os.path.join(self.repo, self.test_rel)
        self.initial_test_entries = set(os.listdir(tdir)) if os.path.isdir(tdir) else None
        self.discover_packages()
        if self.packages:
            self.copy_library()

    def discover_packages(self) -> None:
        st = self.statement
        names = []

        def add(name):
            root = name.split(".")[0]
            if root.isidentifier() and root not in names:
                names.append(root)
        for base in (self.repo, os.path.join(self.repo, "src"), os.path.join(self.repo, "lib")):
            if not os.path.isdir(base):
                continue
            for entry in sorted(os.listdir(base)):
                full = os.path.join(base, entry)
                if os.path.isdir(full) and os.path.isfile(os.path.join(full, "__init__.py")) and \
                        entry.lower() not in _NOT_PACKAGES:
                    add(entry)
        for tok in re.findall(r"[Ii]mport(?:ed)? (?:only |from )?`([A-Za-z_][\w.]*)`", st):
            add(tok)
        for tok in re.findall(r"`([A-Za-z_][\w.]*)`", st):
            add(tok)
        names = [n for n in names if n not in self.stdlib and n not in ("pytest", "tests", "test", "docs")
                 and not n.startswith("_")][:60]
        recs = self.runner({"mode": "api", "candidates": names}, timeout=120, as_nobody=False,
                           pythonpath=os.environ.get("PYTHONPATH", ""), sys_path=[])
        found = []
        for r in recs:
            if r.get("kind") != "api" or r.get("error") or not r.get("file"):
                continue
            f = os.path.realpath(r["file"])
            r["is_pkg"] = bool(r.get("path"))
            r["dir"] = os.path.dirname(f) if r["is_pkg"] else f
            r["in_repo"] = f.startswith(self.repo + os.sep)
            found.append(r)
        chosen = [r for r in found if r["in_repo"]]
        if not chosen:
            backticked = re.findall(r"`([A-Za-z_]\w*)`", st)
            chosen = [r for r in found if r["name"] in backticked[:12]] or found[:1]
        explicit = [r for r in chosen if re.search(r"[Ii]mport(?:ed)? from `%s[`.]" % re.escape(r["name"]), st)]
        if explicit:
            chosen = explicit
        self.packages = chosen[:3]
        log("[SETUP] packages: %s" % ", ".join("%s (%s)" % (p["name"], p["dir"]) for p in self.packages))

    def copy_library(self) -> None:
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
        for p in self.packages:
            dst = os.path.join(self.src_root, os.path.basename(p["dir"]))
            if p["is_pkg"]:
                shutil.copytree(p["dir"], dst, ignore=ignore, dirs_exist_ok=True)
            else:
                shutil.copy2(p["dir"], dst)
            p["rel"] = os.path.basename(p["dir"])
        _make_readable(self.src_root)
        probe = self.runner({"mode": "api", "candidates": [p["name"] for p in self.packages]}, timeout=120,
                            pythonpath=self.src_root, sys_path=[self.src_root])
        bad = [r for r in probe if r.get("kind") == "api" and
               (r.get("error") or not str(r.get("file") or "").startswith(self.src_root))]
        if bad:
            log("[SETUP] the package does not import from the copy alone (%s); adding the repository" %
                (bad[0].get("error") or bad[0].get("file")))
            self.extra_path = [self.repo]
        for p in self.packages:
            for r in probe:
                if r.get("name") == p["name"]:
                    p["api"] = r.get("api") or p.get("api") or ""

    def new_run_dir(self) -> str:
        with self.counter_lock:
            self.run_counter += 1
            n = self.run_counter
        d = os.path.join(self.work, "run", "%05d" % n)
        for sub in ("tmp", "out"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
            os.chmod(os.path.join(d, sub), 0o777)
        os.chmod(os.path.dirname(d), 0o755)
        os.chmod(d, 0o755)
        return d

    def base_env(self, tmp: str, pythonpath: str, hash_seed: str = "0") -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "PYTHON"))}
        env.update({"HOME": tmp, "TMPDIR": tmp, "PYTHONHASHSEED": hash_seed, "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": pythonpath, "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"})
        for k in ("OPENROUTER_API_KEY", "GMMT_INFERENCE_API_KEY"):
            env.pop(k, None)
        return env

    def runner(self, cfg: dict, *, timeout: float, as_nobody: bool = True, pythonpath: str | None = None,
               hash_seed: str = "0", sys_path: list | None = None) -> list:
        d = self.new_run_dir()
        cfg = dict(cfg)
        cfg["out"] = os.path.join(d, "out", "out.jsonl")
        cfg["tmp_base"] = os.path.join(d, "tmp")
        cfg.setdefault("packages", [p["name"] for p in self.packages])
        cfg["sys_path"] = sys_path if sys_path is not None else [self.src_root] + self.extra_path + [self.record_dir]
        cfg_path = os.path.join(d, "cfg.json")
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh)
        os.chmod(cfg_path, 0o644)
        pp = pythonpath if pythonpath is not None else os.pathsep.join([self.src_root] + self.extra_path)
        env = self.base_env(os.path.join(d, "tmp"), pp, hash_seed)
        _seal(d)
        rc, output = _run_proc([self.py, self.runner_file, "--tg-runner", cfg_path], cwd=d,
                               env=env, timeout=timeout, as_nobody=as_nobody)
        recs = []
        try:
            with open(cfg["out"], errors="replace") as fh:
                for line in fh:
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        if rc is None:
            recs.append({"kind": "runner_timeout"})
        elif not recs or recs[-1].get("kind") != "end":
            recs.append({"kind": "runner_crash", "output": output[-2000:]})
        _remove_dir(d)
        return recs

    def package_files(self) -> list:
        files = []
        for p in self.packages:
            base = os.path.join(self.src_root, p["rel"])
            if os.path.isfile(base):
                files.append(p["rel"])
                continue
            for root, dirs, fs in os.walk(base):
                dirs[:] = sorted(d for d in dirs if d != "__pycache__")
                for f in sorted(fs):
                    if f.endswith(".py"):
                        files.append(os.path.relpath(os.path.join(root, f), self.src_root))
        return files

    def scope_text(self) -> str:
        keep, skipping = [], False
        for line in self.statement.split("\n"):
            if re.match(r"\s*#+\s", line):
                low = line.lower()
                skipping = any(w in low for w in ("not part of", "out of scope", "not in scope", "excluded",
                                                  "unstable", "non-goals"))
            if not skipping:
                keep.append(line)
        return "\n".join(keep)

    def mentioned_files(self, files: list) -> list:
        st = self.scope_text()
        out = []
        by_stem = {}
        for rel in files:
            by_stem.setdefault(os.path.basename(rel)[:-3], []).append(rel)
        for rel in files:
            dotted = rel[:-3].replace(os.sep, ".")
            stem = os.path.basename(rel)[:-3]
            if re.search(re.escape(rel), st) or re.search(r"`%s`" % re.escape(dotted), st):
                out.append(rel)
            elif stem not in ("__init__", "utils", "util", "core", "compat", "base", "main", "_compat") \
                    and re.search(r"`%s`" % re.escape(stem), st) \
                    and rel == min(by_stem[stem], key=lambda r: (r.count(os.sep), r)):
                out.append(rel)
        names = set(re.findall(r"`([A-Za-z_]\w*)(?:\(\))?`", st)) | set(re.findall(r"`[\w.]*\.([A-Za-z_]\w*)`", st))
        for rel in files:
            if rel in out:
                continue
            try:
                with open(os.path.join(self.src_root, rel), errors="replace") as fh:
                    tree = ast.parse(fh.read())
            except Exception:
                continue
            defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
            if len(defined & names) >= 1 and not os.path.basename(rel).startswith("test"):
                out.append(rel)
        return out

    def build_context(self) -> str:
        files = self.package_files()
        mentioned = self.mentioned_files(files)
        inits = [f for f in files if f.endswith("__init__.py") and f.count(os.sep) <= 1]
        self.scope_files = list(dict.fromkeys(mentioned + inits)) or files[:6]
        parts = ["# Task\n\n" + self.statement.strip()]
        facts = ["# Facts gathered by the tooling", "",
                 "- Repository root: %s (shell commands run there)" % self.repo,
                 "- Python: %s" % sys.version.split()[0],
                 "- The generated suite goes to `%s/`; you only write case files." % self.test_rel]
        for p in self.packages:
            facts.append("- Package under test: `%s`%s, imported from %s" % (
                p["name"], " %s" % p["version"] if p.get("version") else "", p["dir"]))
        parts.append("\n".join(facts))
        listing = []
        for rel in files:
            try:
                with open(os.path.join(self.src_root, rel), errors="replace") as fh:
                    n = sum(1 for _ in fh)
            except OSError:
                n = 0
            listing.append("%s (%d lines)" % (rel, n))
        parts.append("# Package files\n\n" + "\n".join(listing[:200]))
        for p in self.packages:
            if p.get("api"):
                parts.append("# Public names of `%s`\n\n%s" % (p["name"], _clip(p["api"], 14000)))
        budget = 90000
        included = []
        for rel in self.scope_files:
            try:
                with open(os.path.join(self.src_root, rel), errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if len(text) <= budget:
                parts.append("# Source: %s\n\n```python\n%s\n```" % (rel, _number(text)))
                budget -= len(text)
                included.append(rel)
                self.context_files.add(rel)
            else:
                parts.append("# Outline: %s (too long to include; read parts with the shell)\n\n%s" % (
                    rel, _outline(text)))
        for rel in files:
            if rel in included or rel in self.scope_files or budget < 2000:
                continue
            try:
                with open(os.path.join(self.src_root, rel), errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            outline = _outline(text)
            parts.append("# Outline: %s\n\n%s" % (rel, outline))
            budget -= len(outline)
        docs = []
        for tok in re.findall(r"`([\w./-]+\.(?:rst|md|txt))`", self.statement):
            if os.path.isfile(os.path.join(self.repo, tok)) and tok not in docs:
                docs.append(tok)
        if not docs:
            for tok in ("README.md", "README.rst", "README"):
                if os.path.isfile(os.path.join(self.repo, tok)):
                    docs.append(tok)
                    break
        dbudget = 30000
        for rel in docs[:4]:
            try:
                with open(os.path.join(self.repo, rel), errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            chunk = _clip(text, max(3000, dbudget // max(1, len(docs))))
            parts.append("# Documentation: %s\n\n%s" % (rel, chunk))
        return "\n\n".join(parts)

    def allowed_roots(self) -> set:
        return set(self.stdlib) | {"pytest"} | {p["name"] for p in self.packages}

    def import_rule(self):
        if not hasattr(self, "_import_rule"):
            pkg_names = {p["name"] for p in self.packages}
            rule = set()
            for m in re.finditer(r"\b[Ii]mport (?:only |from )?((?:[^.;\n]|\.(?=\S))+)", self.statement):
                before = self.statement[max(0, m.start() - 16):m.start()].lower()
                if re.search(r"\b(?:not|never|no|avoid|don't|cannot|can't|must not)\s+$", before):
                    continue
                rule |= {t for t in re.findall(r"`([A-Za-z_][\w.]*)`", m.group(1)) if t.split(".")[0] in pkg_names}
            self._import_rule = rule or None
        return self._import_rule

    def import_rule_problem(self, modname: str, names) -> str:
        rule = self.import_rule()
        if not rule or modname.split(".")[0] not in {p["name"] for p in self.packages}:
            return ""
        if modname in rule:
            return ""
        if names is None and any(r.startswith(modname + ".") for r in rule):
            return ""
        if names is not None and all("%s.%s" % (modname, n) in rule for n in names):
            return ""
        return "the task allows imports only from %s; import the public names from there" % (
            ", ".join("`%s`" % r for r in sorted(rule)))

    def lint(self, source: str):
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return ["syntax error: %s (line %s)" % (e.msg, e.lineno)], {}, []
        mod_problems, case_problems, cases = [], {}, []
        allowed = self.allowed_roots()
        pkg_names = {p["name"] for p in self.packages}
        lib_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root = a.name.split(".")[0]
                    if root not in allowed:
                        mod_problems.append("line %d: `import %s` is not allowed (package under test, pytest and "
                                            "the standard library only)" % (node.lineno, a.name))
                    if any(part.startswith("_") for part in a.name.split(".")):
                        mod_problems.append("line %d: `import %s` uses a private module" % (node.lineno, a.name))
                    if root in pkg_names:
                        lib_aliases.add((a.asname or a.name).split(".")[0])
                    problem = self.import_rule_problem(a.name, None)
                    if problem:
                        mod_problems.append("line %d: %s" % (node.lineno, problem))
            elif isinstance(node, ast.ImportFrom):
                modname = node.module or ""
                root = modname.split(".")[0]
                if node.level:
                    mod_problems.append("line %d: relative imports are not allowed" % node.lineno)
                elif modname != "__future__":
                    if root not in allowed:
                        mod_problems.append("line %d: `from %s import ...` is not allowed (package under test, "
                                            "pytest and the standard library only)" % (node.lineno, modname))
                    if any(part.startswith("_") for part in modname.split(".")):
                        mod_problems.append("line %d: `from %s import ...` uses a private module" % (node.lineno, modname))
                    for a in node.names:
                        if a.name.startswith("_"):
                            mod_problems.append("line %d: imports the private name `%s`" % (node.lineno, a.name))
                        if root in pkg_names:
                            lib_aliases.add(a.asname or a.name)
                    problem = self.import_rule_problem(modname, [a.name for a in node.names])
                    if problem:
                        mod_problems.append("line %d: %s" % (node.lineno, problem))
        lib_helpers = _library_helpers(tree, lib_aliases)
        for node in tree.body:
            what = _import_time_library_use(node, lib_aliases, lib_helpers)
            if what:
                mod_problems.append(
                    "line %d: %s runs library code when the file is imported; if a change to the library breaks that "
                    "code, the file cannot be imported and none of its tests can report the change. Create such "
                    "classes, decorated functions and values inside a helper function that the cases call (for "
                    "example `def make_plugin():` defining the class and returning an instance)" % (node.lineno, what))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("case_"):
                    problem = _lint_case(node)
                    if problem:
                        case_problems[node.name] = problem
                    cases.append(node.name)
                elif node.name.startswith("test"):
                    mod_problems.append("line %d: rename helper `%s` (names starting with `test` are reserved)" % (
                        node.lineno, node.name))
                continue
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                mod_problems.append("line %d: rename class `%s` (names starting with `Test` are reserved)" % (
                    node.lineno, node.name))
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "EXCEPTION_CLASSES" for t in targets) and \
                        not (isinstance(value, ast.Constant) and isinstance(value.value, bool)):
                    mod_problems.append("line %d: EXCEPTION_CLASSES must be True or False" % node.lineno)
                continue
            if isinstance(node, ast.ClassDef):
                continue
            mod_problems.append("line %d: only imports, constants, helper functions and classes are allowed at "
                                "module level" % node.lineno)
        seen = set()
        for c in cases:
            if c in seen:
                case_problems[c] = "defined twice in the file"
            seen.add(c)
        users = _helper_users(tree, cases)

        def report(node, text):
            owner = _owner_top(tree, node)
            if owner is None:
                mod_problems.append(text)
            elif owner.startswith("case_"):
                case_problems.setdefault(owner, text)
            else:
                for c in users.get(owner, ()):
                    case_problems.setdefault(c, "%s (in `%s`, which this case uses)" % (text, owner))
        for node in ast.walk(tree):
            msg = _forbidden(node)
            if msg:
                report(node, "line %d: %s" % (node.lineno, msg))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for item in cls.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in _PROTOCOL_METHODS and \
                        _counts_calls(item):
                    report(item, "line %d: `%s.%s` counts or logs its calls, but how often the library calls `%s` "
                                 "is not part of the contract (a refactoring may call it more or less often)" % (
                                     item.lineno, cls.name, item.name, item.name))
        for fn in [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]:
            for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler) and n.name]:
                for sub in ast.walk(h):
                    if _uses_exception_text(sub, h.name):
                        report(sub, "line %d: uses the text or arguments of a caught exception" % sub.lineno)
                        break
        return mod_problems[:10], case_problems, cases

    def add_case_file(self, fname: str, source: str) -> str:
        name = fname[:-3]
        if name not in self.case_sources and len(self.case_sources) >= MAX_CASE_FILES:
            return "%s: not saved (at most %d case files)" % (fname, MAX_CASE_FILES)
        self.case_sources[name] = source
        path = os.path.join(self.cases_dir, fname)
        with open(path, "w") as fh:
            fh.write(source)
        os.chmod(path, 0o644)
        for store in (self.records, self.excluded, self.case_problems):
            for key in [k for k in store if k.startswith(name + "::")]:
                del store[key]
        self.module_problems.pop(name, None)
        return ""

    def keep_previous_cases(self, module: str, new_source: str):
        if module not in self.case_sources:
            return None
        good = [c for c in self.case_order.get(module, []) if "%s::%s" % (module, c) in self.records]
        try:
            new_cases = {n.name for n in ast.parse(new_source).body if isinstance(n, ast.FunctionDef)}
        except SyntaxError:
            new_cases = set()
        keep = [c for c in good if c not in new_cases]
        if not keep:
            return None
        try:
            tree = ast.parse(self.case_sources[module])
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name.startswith("case_")
                                                     and n.name not in keep)]
            prev_source = ast.unparse(tree) + "\n"
        except Exception:
            return None
        k = 1
        while "%s_prev%d" % (module, k) in self.case_sources:
            k += 1
        prev = "%s_prev%d" % (module, k)
        if self.add_case_file(prev + ".py", prev_source):
            return None
        return prev, ("%s.py was replaced; %d recorded cases missing from the new version were kept in %s.py" % (
            module, len(keep), prev))

    def record_modules(self, names: list) -> dict:
        report = {}
        todo = []
        for name in names:
            mprobs, cprobs, cases = self.lint(self.case_sources[name])
            self.case_order[name] = cases
            rep = {"cases": len(cases), "values": 0, "raises": [], "problems": [], "module": list(mprobs)}
            report[name] = rep
            if mprobs:
                self.module_problems[name] = list(mprobs)
                continue
            for c, prob in cprobs.items():
                self.case_problems["%s::%s" % (name, c)] = prob
                rep["problems"].append("%s: %s" % (c, prob))
            keep = [c for c in cases if c not in cprobs]
            try:
                self.test_names[name] = _test_names(self.case_sources[name])
                self.write_record_module(name, set(keep))
            except Exception as e:
                err = "could not turn the file into tests: " + _fmt_exc(e)
                rep["module"].append(err)
                self.module_problems[name] = [err]
                continue
            todo += ["%s::%s" % (name, c) for c in keep]
        if not todo:
            return report
        plans = [dict(trace=True, reverse=False, seed="0"),
                 dict(trace=False, reverse=True, seed="4217", clock_shift=CLOCK_SHIFT),
                 dict(trace=False, reverse=False, seed="91", clock_shift=CLOCK_SHIFT / 37.0)]
        results = [None] * len(plans)

        def record(i):
            try:
                results[i] = self._record(todo, **plans[i])
            except Exception:
                results[i] = None
        helpers = [threading.Thread(target=record, args=(i,), daemon=True) for i in range(1, len(plans))]
        for t in helpers:
            t.start()
        record(0)
        for t in helpers:
            t.join()
        for i, r in enumerate(results):
            if r is None:
                results[i] = self._record(todo, **plans[i])
        first, second, third = results
        for name, errs in first.get("__module_errors__", {}).items():
            if name in report:
                report[name]["module"] += errs
                self.module_problems.setdefault(name, []).extend(errs)
        for key in todo:
            name, case = key.split("::")
            rep = report[name]
            if name in self.module_problems:
                continue
            a, b, c = first.get(key), second.get(key), third.get(key)
            if a is None:
                problem = "did not finish (it may hang)"
            elif a["status"] == "error":
                problem = a.get("error", "error")
            elif a["status"] == "raises" and a.get("exc") in ("NameError", "UnboundLocalError", "SyntaxError",
                                                                "ImportError", "ModuleNotFoundError"):
                problem = "raised %s: %s (a mistake in the case, not library behaviour)" % (
                    a.get("exc"), (a.get("msg") or "")[:120])
            elif any(x is None or x["status"] != a["status"] or (a["status"] == "value" and x.get("src") != a.get("src"))
                     for x in (b, c)):
                other = next(x for x in (b, c) if x is None or x["status"] != a["status"] or
                             (a["status"] == "value" and x.get("src") != a.get("src")))
                problem = ("result differs between two runs (depends on order, hashing, the current clock reading or "
                           "state)" + _difference(a, other))
            else:
                problem = ""
            if problem:
                self.case_problems[key] = problem
                rep["problems"].append("%s: %s" % (case, problem))
                continue
            self.records[key] = a
            if a["status"] == "value":
                rep["values"] += 1
            else:
                rep["raises"].append("%s -> %s: %s" % (case, a.get("exc"), (a.get("msg") or "")[:100]))
        self.rebuild_coverage()
        if self.pool_threads:
            self.refresh_mutants()
        return report

    def _record(self, keys: list, *, trace: bool, reverse: bool, seed: str, clock_shift: float = 0.0) -> dict:
        back = {self.test_key(k): k for k in keys}
        order = [self.test_key(k) for k in (reversed(keys) if reverse else keys)]
        modules = sorted({k.split("::")[0] for k in order})
        mod_back = {self.test_module(k.split("::")[0]): k.split("::")[0] for k in keys}
        results, module_errors = {}, {}
        attempts = 0
        while order and attempts < 4 and self.left() > 20:
            attempts += 1
            timeout = min(self.left() - 10, 60 + len(order) * (1.5 if trace else 0.5))
            cfg = {"mode": "record", "modules": modules, "cases": order,
                   "timeout": CASE_TIMEOUT * (3 if trace else 1), "clock_shift": clock_shift}
            if trace:
                cfg["trace_root"] = self.src_root + os.sep
            recs = self.runner(cfg, timeout=max(10, timeout), hash_seed=seed)
            started = None
            for r in recs:
                if r.get("kind") == "module_error":
                    module_errors.setdefault(mod_back.get(r["module"], r["module"]), []).append(
                        "importing the file failed: " + r["error"])
                elif r.get("kind") == "start":
                    started = r["key"]
                elif r.get("kind") == "case":
                    results[r["key"]] = r
                    started = None
            progressed = any(k in results for k in order)
            order = [k for k in order if k not in results]
            if not order:
                break
            if started and started in order:
                results[started] = {"key": started, "status": "error",
                                    "error": "did not finish in time (hangs or is far too slow)"}
                order.remove(started)
            elif not progressed:
                break
        out = {back[k]: v for k, v in results.items() if k in back}
        out["__module_errors__"] = module_errors
        return out

    def pins_classes(self, module: str) -> bool:
        return bool(re.search(r"(?m)^EXCEPTION_CLASSES\s*(?::[^=]*)?=\s*True\b", self.case_sources.get(module, "")))

    def test_module(self, module: str) -> str:
        return "test_" + (module[len("cases_"):] if module.startswith("cases_") else module)

    def test_key(self, key: str) -> str:
        module, case = key.split("::")
        return "%s::%s" % (self.test_module(module), self.test_names[module][case])

    def write_record_module(self, module: str, keep: set) -> None:
        topic = module[len("cases_"):] if module.startswith("cases_") else module
        text = _test_module_source(self.case_sources[module], topic, self.test_names[module], keep)
        path = os.path.join(self.record_dir, self.test_module(module) + ".py")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)

    def rebuild_coverage(self) -> None:
        cov, lc = {}, {}
        for key, rec in self.records.items():
            if key in self.excluded:
                continue
            for rel, line in rec.get("lines") or []:
                cov.setdefault(rel, set()).add(line)
                lc.setdefault((rel, line), set()).add(key)
        self.cov_lines, self.line_cases = cov, lc

    def usable_keys(self, module: str) -> list:
        out = []
        for c in self.case_order.get(module, []):
            key = "%s::%s" % (module, c)
            if key in self.records and key not in self.excluded:
                out.append(key)
        return out

    def format_report(self, report: dict) -> str:
        lines = []
        for name in sorted(report):
            rep = report[name]
            if rep["module"]:
                lines.append("%s.py: NOT USABLE until fixed:" % name)
                lines += ["  - " + p for p in rep["module"][:8]]
                continue
            lines.append("%s.py: %d cases, %d recorded values, %d raise" % (
                name, rep["cases"], rep["values"], len(rep["raises"])))
            if rep["raises"]:
                lines.append("  raising cases (make sure each is meant to raise; a typo also raises):")
                lines += ["    " + x for x in rep["raises"][:15]]
                if len(rep["raises"]) > 15:
                    lines.append("    ... %d more" % (len(rep["raises"]) - 15))
            if rep["problems"]:
                lines.append("  dropped until fixed:")
                lines += ["    " + x for x in rep["problems"][:20]]
        total = sum(1 for k in self.records if k not in self.excluded)
        lines.append("Recorded cases in the suite so far: %d." % total)
        return "\n".join(lines)

    def conversation_summary(self) -> str:
        lines = ["Summary of my earlier work (the conversation was shortened). These case files are saved and "
                 "recorded; I keep adding new files rather than rewriting them:"]
        for name in sorted(self.case_sources):
            keys = self.usable_keys(name)
            lines.append("- %s.py: %d recorded cases%s" % (name, len(keys),
                                                           " (not usable)" if name in self.module_problems else ""))
        stats = self.mutation_stats()
        if stats:
            lines.append("Mutation analysis so far: %d changes detected, %d undetected, %d set aside." % (
                stats.get("killed", 0), stats.get("survived", 0), len(self.skipped)))
        return "\n".join(lines)

    def build_suite(self) -> tuple:
        files, idmap = {}, {}
        for name in sorted(self.case_sources):
            if name in self.module_problems:
                continue
            keys = self.usable_keys(name)
            if not keys:
                continue
            topic = name[len("cases_"):] if name.startswith("cases_") else name
            names = self.test_names.get(name) or _test_names(self.case_sources[name])
            recs = {k.split("::")[1]: self.records[k] for k in keys}
            if not self.pins_classes(name):
                recs = {c: dict(r, cls=None) for c, r in recs.items()}
            try:
                src = _test_module_source(self.case_sources[name], topic, names, set(recs), recs)
            except Exception as e:
                log("[BUILD] %s: %s" % (name, _fmt_exc(e)))
                continue
            tname = "test_" + topic
            files[tname + ".py"] = src
            for case in recs:
                idmap["%s::%s" % (tname, names[case])] = "%s::%s" % (name, case)
        if files:
            files["conftest.py"] = CONFTEST_SRC
        return files, idmap

    def ci_run(self, files: dict, hash_seed: str = "0", src_root: str | None = None) -> dict:
        d = self.new_run_dir()
        tdir = os.path.join(d, os.path.basename(self.test_rel) or "tests")
        os.makedirs(tdir)
        for rel, text in files.items():
            with open(os.path.join(tdir, rel), "w") as fh:
                fh.write(text)
            os.chmod(os.path.join(tdir, rel), 0o444)
        os.chmod(tdir, 0o555)
        _seal(d)
        out_dir = os.path.join(d, "out")
        junit = os.path.join(out_dir, "junit.xml")
        env = self.base_env(os.path.join(d, "tmp"), os.pathsep.join([src_root or self.src_root] + self.extra_path),
                            hash_seed)
        cmd = [self.py, "-m", "pytest", tdir, "-q", "-p", "no:cacheprovider", "-c", os.devnull,
               "--rootdir=" + d, "--junitxml=" + junit]
        t0 = time.time()
        import resource
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        rc, output = _run_proc(cmd, cwd=d, env=env,
                               timeout=max(20, min(self.end_left() - 5, self.suite_limit * 4 + 30)), as_nobody=True)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        secs = time.time() - t0
        cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
        tests, collect_errors = {}, []
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(junit).getroot()
            for tc in root.iter("testcase"):
                cls = tc.get("classname") or ""
                if not cls and any(child.tag == "error" for child in tc):
                    collect_errors.append((tc.get("name") or "").split(".")[-1] + ".py")
                    continue
                tid = "%s::%s" % (cls.split(".")[-1], tc.get("name"))
                outcome = "passed"
                for child in tc:
                    if child.tag in ("failure", "error"):
                        outcome = "failed"
                        break
                    if child.tag == "skipped":
                        outcome = "skipped"
                tests[tid] = {"outcome": outcome, "time": float(tc.get("time") or 0)}
        except Exception:
            pass
        for line in output.split("\n"):
            mm = re.match(r"ERROR (?:collecting )?\S*?(test_\w+\.py)(?:\s|$)", line.strip())
            if mm and mm.group(1) not in collect_errors:
                collect_errors.append(mm.group(1))
        os.chmod(tdir, 0o755)
        _remove_dir(d)
        return {"rc": rc, "tests": tests, "secs": secs, "cpu": cpu, "output": output[-3000:],
                "collect_errors": collect_errors}

    def verify_suite(self, confirm_runs: int = 2) -> dict:
        self.pause_pool(True)
        try:
            return self._verify_suite(confirm_runs)
        finally:
            self.pause_pool(False)

    def _verify_suite(self, confirm_runs: int) -> dict:
        clean = 0
        for attempt in range(8):
            files, idmap = self.build_suite()
            if not files or self.end_left() < 20:
                break
            seed = "0" if clean == 0 else str(1000 + attempt)
            res = self.ci_run(files, hash_seed=seed)
            failed = [tid for tid, t in res["tests"].items() if t["outcome"] != "passed"]
            broken = []
            if res["rc"] not in (0, 1):
                broken = self.broken_files(res, files, seed)
                if not broken:
                    log("[CI] run %d seed=%s: pytest exit %s and no file to blame; keeping the last passing suite" % (
                        attempt + 1, seed, res["rc"]))
                    break
            log("[CI] run %d seed=%s rc=%s tests=%d failed=%d secs=%.1f cpu=%.1f broken=%s" % (
                attempt + 1, seed, res["rc"], len(res["tests"]), len(failed), res["secs"], res.get("cpu") or 0,
                broken[:3]))
            for f in broken:
                mod = "cases_" + f[len("test_"):-3]
                self.module_problems.setdefault(mod, []).append(
                    "the generated tests for this file do not run under pytest: " + _clip(res["output"], 500))
                self.notes.append("%s.py was dropped: its generated tests do not run under pytest (%s)" % (
                    mod, _clip(res["output"], 300)))
            dropped = 0
            for tid in failed:
                key = idmap.get(tid)
                if key and key not in self.excluded:
                    self.excluded[key] = "failed when the suite ran under pytest"
                    dropped += 1
            if dropped:
                self.notes.append("%d cases were dropped because their tests failed under pytest: %s" % (
                    dropped, ", ".join(idmap.get(t, t).split("::")[-1] for t in failed[:10])))
            if broken or failed:
                clean = 0
                self.rebuild_coverage()
                continue
            if not res["tests"]:
                break
            if min(res["secs"], res.get("cpu") or res["secs"]) > self.suite_limit * 0.45:
                self._trim_slow(res, idmap)
                clean = 0
                continue
            clean += 1
            self.best_files = files
            if clean >= confirm_runs:
                return files
        return self.best_files

    def broken_files(self, res: dict, files: dict, seed: str) -> list:
        tfiles = [f for f in files if f.startswith("test_")]
        broken = [f for f in res.get("collect_errors") or [] if f in tfiles]
        if broken:
            return list(dict.fromkeys(broken))
        for f in tfiles:
            if self.end_left() < 30:
                break
            r1 = self.ci_run({f: files[f], "conftest.py": files["conftest.py"]}, hash_seed=seed)
            if r1["rc"] not in (0, 1):
                broken.append(f)
        return broken

    def contract_check(self) -> None:
        low = self.statement.lower()
        if not re.search(r"message text|error messages?|str\(\)` and `repr\(\)|repr\(\)", low):
            return
        root = os.path.join(self.work, "variant")
        if not os.path.isdir(root):
            shutil.copytree(self.src_root, root)
            changed = 0
            for dirpath, _, fs in os.walk(root):
                for f in fs:
                    if not f.endswith(".py"):
                        continue
                    path = os.path.join(dirpath, f)
                    try:
                        with open(path, "rb") as fh:
                            new = _contract_variant(fh.read())
                    except Exception:
                        new = None
                    if new is not None:
                        with open(path, "wb") as fh:
                            fh.write(new)
                        changed += 1
            _make_readable(root)
            log("[CONTRACT] variant with reworded messages/repr: %d files changed" % changed)
        files, idmap = self.build_suite()
        if not files or self.end_left() < 60:
            return
        res = self.ci_run(files, src_root=root)
        failed = [tid for tid, t in res["tests"].items() if t["outcome"] != "passed"]
        if res["rc"] not in (0, 1) or not res["tests"]:
            log("[CONTRACT] variant run unusable (rc=%s); skipped" % res["rc"])
            return
        for tid in failed:
            key = idmap.get(tid)
            if key:
                self.excluded[key] = "depends on error message text or repr()"
        if failed:
            self.rebuild_coverage()
        log("[CONTRACT] %d tests depend on message text or repr(); dropped" % len(failed))

    def rewrite_check(self) -> None:
        root = os.path.join(self.work, "variant_rw")
        if not os.path.isdir(root):
            shutil.copytree(self.src_root, root)
            changed = 0
            for dirpath, _, fs in os.walk(root):
                for f in fs:
                    if not f.endswith(".py"):
                        continue
                    path = os.path.join(dirpath, f)
                    try:
                        with open(path, "rb") as fh:
                            new = _rewrite_variant(fh.read())
                    except Exception:
                        new = None
                    if new is not None:
                        with open(path, "wb") as fh:
                            fh.write(new)
                        changed += 1
            _make_readable(root)
            log("[REWRITE] refactored copy: %d files changed" % changed)
        files, idmap = self.build_suite()
        if not files or self.end_left() < 60:
            return
        res = self.ci_run(files, src_root=root)
        if res["rc"] not in (0, 1) or not res["tests"]:
            log("[REWRITE] run on the refactored copy unusable (rc=%s); skipped" % res["rc"])
            return
        failed = [tid for tid, t in res["tests"].items() if t["outcome"] != "passed"]
        if len(failed) > max(3, int(0.02 * len(res["tests"]))):
            log("[REWRITE] %d of %d tests fail on the refactored copy; the copy is suspect, nothing dropped" % (
                len(failed), len(res["tests"])))
            return
        for tid in failed:
            key = idmap.get(tid)
            if key:
                self.excluded[key] = "fails on a refactored copy of the library"
        if failed:
            self.rebuild_coverage()
        log("[REWRITE] %d tests fail on the refactored copy; dropped %s" % (len(failed), failed[:5]))

    def _trim_slow(self, res: dict, idmap: dict) -> None:
        budget = self.suite_limit * 0.3
        times = sorted(((t["time"], tid) for tid, t in res["tests"].items()), reverse=True)
        total = sum(t for t, _ in times)
        for t, tid in times:
            if total <= budget:
                break
            key = idmap.get(tid)
            if key:
                self.excluded[key] = "too slow"
                total -= t

    def converse(self, first_message: str, *, max_turns: int, until_frac: float, batch_mode: bool = False) -> list:
        msg = first_message
        replies = []
        nudged = False
        for _ in range(max_turns):
            if self.frac() > until_frac or not self.llm.affordable():
                break
            reply, finish = self.llm.ask(msg)
            if reply is None:
                break
            replies.append(reply)
            blocks, done, truncated = _parse_reply(reply, finish)
            feedback, written, ran_shell = [], [], False
            shell_end = time.time() + max(0.0, min(SHELL_SECONDS_PER_REPLY, self.left() - 90))
            for kind, name, body in blocks:
                if kind == "case-unnamed":
                    k = 1
                    while "cases_unnamed_%d" % k in self.case_sources:
                        k += 1
                    name = "cases_unnamed_%d.py" % k
                    kind = "case"
                    feedback.append("A python block with cases had no file name; it was saved as %s. Name case files "
                                    "on the fence line: ```python cases_<topic>.py" % name)
                if kind == "bash":
                    ran_shell = True
                    allowed = min(60.0, shell_end - time.time())
                    if allowed < 5:
                        feedback.append("$ %s\n[not run: no time left for shell commands in this reply]" % (
                            _clip(body, 300)))
                        continue
                    out, rc = self.shell(body, timeout=allowed)
                    feedback.append("$ %s\n%s[exit %s]" % (_clip(body, 300), out, rc))
                elif kind == "case":
                    if batch_mode and name[:-3] in self.case_sources and name[:-3] not in self.round_files:
                        base, k = name[:-3], 2
                        while "%s_%d" % (base, k) in self.case_sources:
                            k += 1
                        feedback.append("%s already exists; saved as %s_%d.py" % (name, base, k))
                        name = "%s_%d.py" % (base, k)
                    self.round_files.add(name[:-3])
                    kept = self.keep_previous_cases(name[:-3], body)
                    if kept:
                        written.append(kept[0])
                        feedback.append(kept[1])
                    err = self.add_case_file(name, body)
                    if err:
                        feedback.append(err)
                    else:
                        written.append(name[:-3])
            problems = False
            if written:
                report = self.record_modules(sorted(set(written)))
                problems = any(r["module"] or r["problems"] for r in report.values())
                feedback.append(self.format_report(report))
                for name, r in sorted(report.items()):
                    log("[REC] %s: %d cases, %d values, %d raise, %d problems%s" % (
                        name, r["cases"], r["values"], len(r["raises"]), len(r["problems"]),
                        ", MODULE UNUSABLE" if r["module"] else ""))
            if truncated:
                feedback.append("Your reply was cut off; the unfinished block was ignored. Write smaller files.")
            if self.notes:
                feedback += self.notes
                self.notes = []
            if not blocks:
                if done and not self.case_sources and not nudged:
                    nudged = True
                    feedback.append("Your reply contained no case files, only text, so nothing has been saved yet. "
                                    "Write the case files now as ```python cases_<topic>.py blocks in your reply.")
                else:
                    if done or batch_mode:
                        break
                    feedback.append("No action found. Write case files (```python cases_<topic>.py) or shell "
                                    "commands (```bash), or reply DONE.")
            elif written and not problems and not ran_shell and (done or batch_mode):
                break
            elif done and not written and not ran_shell:
                break
            if batch_mode:
                tail = ("Fix dropped cases by writing corrected versions into a new case file (resend a whole file "
                        "only if it is NOT USABLE), or continue with the listed changes.")
            else:
                tail = ("Fix dropped cases by writing corrected versions into a new case file (resend a whole file "
                        "only if it is NOT USABLE), then continue; reply DONE when finished.")
            msg = "\n\n".join(feedback + [tail])
        return replies

    def shell(self, cmd: str, timeout: float = 60.0) -> tuple:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for k in ("OPENROUTER_API_KEY", "GMMT_INFERENCE_API_KEY"):
            env.pop(k, None)
        rc, out = _run_proc(["bash", "-c", cmd], cwd=self.repo, env=env, timeout=timeout)
        self.protect_repo()
        return _clip(out, 6000), ("timeout" if rc is None else rc)

    def changed_files(self):
        r = subprocess.run(["git", "-C", self.repo, "diff", "--name-only", "-z", "HEAD"], timeout=30,
                           capture_output=True)
        if r.returncode != 0:
            return None
        return [x for x in r.stdout.decode(errors="replace").split("\0") if x]

    def snapshot_repo(self) -> None:
        try:
            for rel in self.changed_files() or []:
                try:
                    with open(os.path.join(self.repo, rel), "rb") as fh:
                        self.repo_baseline[rel] = fh.read()
                except OSError:
                    self.repo_baseline[rel] = None
        except Exception:
            pass

    def protect_repo(self) -> None:
        try:
            reverted = 0
            for rel in self.changed_files() or []:
                path = os.path.join(self.repo, rel)
                if rel not in self.repo_baseline:
                    subprocess.run(["git", "-C", self.repo, "checkout", "HEAD", "--", rel], timeout=30,
                                   capture_output=True)
                    reverted += 1
                    continue
                want = self.repo_baseline[rel]
                try:
                    with open(path, "rb") as fh:
                        have = fh.read()
                except OSError:
                    have = None
                if have == want:
                    continue
                if want is None:
                    os.remove(path)
                else:
                    with open(path, "wb") as fh:
                        fh.write(want)
                reverted += 1
            if reverted:
                log("[SHELL] reverted changes to %d tracked files" % reverted)
        except Exception:
            pass

    def ask_layout(self):
        prompt = ("Read the task statement below and answer with exactly two lines:\n"
                  "REPO: <absolute path of the repository the task is about, or unknown>\n"
                  "TESTS: <directory in which the new test files must be added, or unknown>\n\n"
                  "Task statement:\n\n" + _clip(self.statement, 30000))
        self.llm.messages = [{"role": "system", "content": "You answer questions about task statements briefly."}]
        reply, _ = self.llm.ask(prompt, max_tokens=300)
        self.llm.messages = []
        repo = tests = None
        for line in (reply or "").split("\n"):
            mm = re.match(r"\W*(REPO|TESTS)\W*:\s*`?([^`\s]+)`?", line.strip())
            if mm and mm.group(2).lower().strip(".") != "unknown":
                if mm.group(1) == "REPO":
                    repo = mm.group(2).rstrip("/")
                else:
                    tests = mm.group(2)
        log("[SETUP] layout from the model: repo=%s tests=%s" % (repo, tests))
        return repo, tests

    def coverage_report(self, detail: bool = False) -> str:
        items = []
        st = self.scope_text()
        files = list(dict.fromkeys(self.scope_files + sorted(self.cov_lines)))
        for rel in files:
            path = os.path.join(self.src_root, rel)
            try:
                with open(path, errors="replace") as fh:
                    tree = ast.parse(fh.read())
            except Exception:
                continue
            covered = self.cov_lines.get(rel, set())
            in_scope = rel in self.scope_files
            if not covered and not in_scope:
                continue
            try:
                with open(path, errors="replace") as fh:
                    src_lines = fh.read().split("\n")
            except OSError:
                src_lines = []
            for qual, fn in _functions(tree):
                if any((_dotted(d) or "").endswith("overload") for d in fn.decorator_list):
                    continue
                stmts = sorted(set(_stmt_lines(fn.body)))
                if not stmts:
                    continue
                missed = [n for n in stmts if n not in covered]
                if not missed:
                    continue
                short = qual.split(".")[-1]
                if len(missed) == len(stmts):
                    named = not short.startswith("_") and re.search(r"\b%s\b" % re.escape(short), st)
                    owner = qual.split(".")[-2] if "." in qual else ""
                    operator = (short.startswith("__") and short.endswith("__") and short not in _SKIP_FUNCS
                                and owner and re.search(r"\b%s\b" % re.escape(owner), st))
                    if not (named or operator):
                        continue
                    items.append((0, "%s: %s (line %d) is never called" % (rel, qual, fn.lineno)))
                else:
                    text = "%s: %s: lines %s never run" % (rel, qual, _ranges(missed))
                    if detail and src_lines:
                        text += "\n" + "\n".join("    %5d  %s" % (n, src_lines[n - 1].rstrip())
                                                   for n in missed[:4] if n <= len(src_lines))
                    items.append((1, text))
        items.sort(key=lambda x: x[0])
        return "\n".join(t for _, t in items[:60 if not detail else 30])

    def refresh_mutants(self) -> int:
        added = []
        for rel in sorted(self.cov_lines):
            path = os.path.join(self.src_root, rel)
            if not rel.endswith(".py") or not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as fh:
                    source = fh.read()
                muts = _generate_mutants(rel, source, self.cov_lines[rel],
                                         more=self.kinds > 1 and rel in self.scope_files)
            except Exception as e:
                log("[MUT] %s: %s" % (rel, _fmt_exc(e)))
                continue
            for m in muts:
                k = (m["file"], m["start"], m["end"], m["repl"])
                if k not in self.mutant_keys:
                    self.mutant_keys.add(k)
                    added.append(m)
        groups = {}
        for m in added:
            groups.setdefault((m["file"], m["func"]), []).append(m)
        n = len(self.mutants)
        fresh = []
        while groups:
            for g in list(groups):
                m = groups[g].pop(0)
                n += 1
                m["id"] = "m%d" % n
                self.mutants[m["id"]] = m
                fresh.append(m)
                if not groups[g]:
                    del groups[g]
        if fresh and self.pool_threads:
            self.pool_submit(fresh, foreground=False)
        return len(added)

    def relevant_keys(self, m: dict) -> list:
        keys = set()
        line_cases = self.line_cases
        for line in range(m["stmt_start"], m["stmt_end"] + 1):
            keys |= line_cases.get((m["file"], line), set())
        if not keys and m.get("scope_lines"):
            for line in m["scope_lines"]:
                keys |= line_cases.get((m["file"], line), set())
        if not keys and m.get("module_level"):
            keys = {k for (f, _), ks in list(line_cases.items()) if f == m["file"] for k in ks}
        return sorted(k for k in keys if k in self.records and k not in self.excluded)

    def ensure_workers(self) -> None:
        n_workers = max(1, min(CPU_WORKERS_CAP, _available_cpus() + 1))
        while len(self.worker_roots) < n_workers:
            root = os.path.join(self.work, "mut", "w%d" % len(self.worker_roots))
            shutil.copytree(self.src_root, root)
            _make_readable(root)
            self.worker_roots.append(root)

    def check_mutant(self, m: dict, root: str, full: bool) -> str:
        skip = set()
        for attempt in range(4):
            try:
                keys = [k for k in self.relevant_keys(m) if k not in self.unreliable and k not in skip]
                if not keys:
                    return "no-case" if attempt == 0 else "survived"
                if not full and len(keys) > CHECK_CASES_CAP:
                    keys = sorted(keys, key=lambda k: (len(self.records[k].get("lines") or ()),
                                                       self.records[k].get("secs", 0.01)))[:CHECK_CASES_CAP]
                keys.sort(key=self.suite_position)
            except Exception:
                return "error"
            verdict, first = self.run_check(m, root, keys)
            if verdict != "killed" or first is None:
                return verdict
            if not self.alone_reproduces(first, root):
                self.unreliable.add(first)
                continue
            if first == keys[0]:
                return "killed"
            why = m.get("why", "")
            if self.run_check(m, root, [first])[0] == "killed":
                m["why"] = why
                return "killed"
            skip.add(first)
            full = True
        m["why"] = "detected only together with other cases"
        return "survived"

    def suite_position(self, key: str) -> tuple:
        module, case = key.split("::")
        order = self.case_order.get(module) or []
        return (self.test_module(module), order.index(case) if case in order else len(order))

    def alone_reproduces(self, key: str, root: str) -> bool:
        if key not in self.alone_ok:
            verdict, _ = self.run_check(None, root, [key])
            self.alone_ok[key] = verdict != "killed"
        return self.alone_ok[key]

    def run_check(self, m, root: str, keys: list) -> tuple:
        try:
            expected, back = {}, {}
            for k in keys:
                r = self.records[k]
                cls = r.get("cls") if self.pins_classes(k.split("::")[0]) else None
                expected[self.test_key(k)] = {"status": r["status"], "src": r.get("src"), "cls": cls}
                back[self.test_key(k)] = k
            secs = sum(self.records[k].get("secs", 0.01) for k in keys)
            longest = max(self.records[k].get("secs", 0.01) for k in keys)
        except Exception:
            return "error", None
        path = os.path.join(root, m["file"]) if m else None
        src = None
        if m:
            with open(os.path.join(self.src_root, m["file"]), "rb") as fh:
                src = fh.read()
        z = self.zygotes.get(root)
        if z is None or (z != "broken" and z.poll() is not None):
            self.zygotes.pop(root, None)
            if self.start_zygote(root) is None:
                self.zygotes[root] = "broken"
        try:
            if m:
                with open(path, "wb") as fh:
                    fh.write(src[:m["start"]] + m["repl"] + src[m["end"]:])
            tkeys = list(expected)
            recs = None
            for factor in (1, 3):
                cfg = {"mode": "check", "modules": sorted({k.split("::")[0] for k in tkeys}), "cases": tkeys,
                       "expected": expected, "stop_on_first": True, "timeout": factor * max(5.0, 10 * longest)}
                wall = factor * (30 + 5 * secs)
                recs = self.zygote_check(root, cfg, m.get("swap") if m else "pristine", wall=wall)
                if recs is None:
                    recs = self.runner(cfg, timeout=wall, pythonpath=os.pathsep.join([root] + self.extra_path),
                                       sys_path=[root] + self.extra_path + [self.record_dir])
                timed_out = any(r.get("kind") == "runner_timeout" or
                                (r.get("kind") == "mismatch" and r.get("status") == "timeout") for r in recs)
                real = any(r.get("kind") == "mismatch" and r.get("status") not in ("timeout", "missing") or
                           r.get("kind") == "module_error" for r in recs)
                if not timed_out or real:
                    break
        finally:
            if m:
                with open(path, "wb") as fh:
                    fh.write(src)
        first = next((back.get(r.get("key")) for r in recs
                      if r.get("kind") == "mismatch" and r.get("status") != "missing"), None)
        why = next((("%s:%s:%s" % (r.get("kind"), r.get("status"), r.get("exc") or "")) for r in recs
                    if r.get("kind") in ("mismatch", "module_error", "runner_timeout", "runner_crash")), "")
        if m:
            m["why"] = why
        if any(r.get("kind") == "library_error" or (r.get("kind") == "module_error" and r.get("cause") == "library")
               for r in recs):
            return "broken", None
        bad = next((r for r in recs if r.get("kind") == "module_error"), None)
        if bad is not None:
            if m:
                m["why"] = "import:%s:%s" % (bad.get("module"), (bad.get("error") or "")[:160])
                self.import_fragile.setdefault(bad.get("module"), "%s, %s" % (
                    m["show"].strip().split("\n")[-1].strip(), (bad.get("error") or "")[:160]))
            return "survived", None
        if any(r.get("kind") == "mismatch" and r.get("status") == "missing" for r in recs):
            return "error", None
        kinds = {r.get("kind") for r in recs}
        if kinds & {"mismatch", "module_error", "runner_timeout"}:
            return "killed", first
        if "runner_crash" in kinds:
            return "error", None
        return "survived", None

    def zygote_check(self, root: str, cfg: dict, swap, wall: float):
        z = self.zygotes.get(root)
        if z is None or z == "broken" or z.poll() is not None:
            return None
        d = self.new_run_dir()
        cfg = dict(cfg)
        cfg["out"] = os.path.join(d, "out", "out.jsonl")
        cfg["tmp_base"] = os.path.join(d, "tmp")
        cfg["sys_path"] = [root] + self.extra_path + [self.record_dir]
        cfg.setdefault("packages", [p["name"] for p in self.packages])
        cfg_path = os.path.join(d, "cfg.json")
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh)
        os.chmod(cfg_path, 0o644)
        status = None
        try:
            z.stdin.write((json.dumps({"cfg": cfg_path, "swap": swap, "wall": wall}) + "\n").encode())
            z.stdin.flush()
            status = _readline(z.stdout, wall + 15)
        except Exception:
            status = None
        if status is None:
            try:
                os.killpg(z.pid, signal.SIGKILL)
            except Exception:
                pass
            self.zygotes.pop(root, None)
            shutil.rmtree(d, ignore_errors=True)
            return None
        recs = []
        try:
            with open(cfg["out"], errors="replace") as fh:
                for line in fh:
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        if status.strip() == "timeout":
            recs.append({"kind": "runner_timeout"})
        elif not recs or recs[-1].get("kind") != "end":
            recs.append({"kind": "runner_crash"})
        shutil.rmtree(d, ignore_errors=True)
        return recs

    def start_zygote(self, root: str):
        if not hasattr(os, "fork"):
            return None
        d = self.new_run_dir()
        cfg = {"sys_path": [root] + self.extra_path + [self.record_dir], "packages": [p["name"] for p in self.packages]}
        cfg_path = os.path.join(d, "zygote.json")
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh)
        os.chmod(cfg_path, 0o644)
        env = self.base_env(os.path.join(d, "tmp"), os.pathsep.join([root] + self.extra_path))
        _seal(d)
        kw = {}
        ids = _nobody_ids()
        if ids:
            kw.update(user=ids[0], group=ids[1], extra_groups=[])
        try:
            z = subprocess.Popen([self.py, self.runner_file, "--tg-zygote", cfg_path], cwd=d,
                                 env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 start_new_session=True, **kw)
        except Exception:
            return None
        if (_readline(z.stdout, 120) or "").strip() != "ready":
            try:
                os.killpg(z.pid, signal.SIGKILL)
            except Exception:
                pass
            return None
        self.zygotes[root] = z
        return z

    def stop_zygotes(self) -> None:
        for z in list(self.zygotes.values()):
            if z not in (None, "broken"):
                try:
                    os.killpg(z.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.zygotes = {}

    def start_pool(self) -> None:
        if self.pool_threads:
            return
        self.ensure_workers()
        for root in self.worker_roots:
            t = threading.Thread(target=self._pool_worker, args=(root,), daemon=True)
            t.start()
            self.pool_threads.append(t)
        self.pool_submit(list(self.mutants.values()), foreground=False)

    def stop_pool(self) -> None:
        with self.pool_cv:
            self.pool_stop = True
            self.pool_cv.notify_all()

    def pool_submit(self, mutants: list, foreground: bool, full: bool = False):
        import heapq
        waiter = {"left": len(mutants)} if foreground else None
        stats = None if foreground else self.function_stats()
        with self.pool_cv:
            for m in mutants:
                self.pool_seq += 1
                prio = (0, self.pool_seq) if foreground else (1,) + self.mutant_priority(m, stats)
                heapq.heappush(self.pool_queue, (prio, self.pool_seq, m["id"], full, waiter))
            self.pool_cv.notify_all()
        return waiter

    def pool_wait(self, waiter, timeout: float) -> None:
        end = time.time() + timeout
        with self.pool_cv:
            while waiter["left"] > 0 and time.time() < end and not self.pool_stop:
                self.pool_cv.wait(min(1.0, max(0.05, end - time.time())))

    def pool_pending(self) -> int:
        with self.pool_cv:
            return len(self.pool_queue)

    def pause_pool(self, paused: bool) -> None:
        with self.pool_cv:
            self.pool_paused = paused
            self.pool_cv.notify_all()
            if paused:
                end = time.time() + 30
                while self.pool_busy and time.time() < end:
                    self.pool_cv.wait(0.5)

    def _pool_worker(self, root: str) -> None:
        import heapq
        while True:
            with self.pool_cv:
                while (not self.pool_queue or self.pool_paused) and not self.pool_stop:
                    self.pool_cv.wait(1.0)
                if self.pool_stop:
                    return
                prio, _, mid, full, waiter = heapq.heappop(self.pool_queue)
                skip = waiter is None and mid in self.mutant_status
                if not skip:
                    self.pool_busy += 1
            status = None
            if not skip and self.left() > 20:
                try:
                    status = self.check_mutant(self.mutants[mid], root, full)
                except Exception:
                    status = "error"
            with self.pool_cv:
                if not skip:
                    self.pool_busy -= 1
                if status and status not in ("error", "broken"):
                    current = self.mutant_status.get(mid)
                    if status == "killed" or current is None or full or not self.mutant_full.get(mid):
                        self.mutant_status[mid] = status
                        self.mutant_full[mid] = full
                elif status == "error" and waiter is None and mid not in self.error_retried:
                    self.error_retried.add(mid)
                    self.pool_seq += 1
                    heapq.heappush(self.pool_queue, ((2,) + self.mutant_priority(self.mutants[mid]), self.pool_seq, mid,
                                                     full, None))
                if waiter is not None:
                    waiter["left"] -= 1
                self.pool_cv.notify_all()

    def evaluate_now(self, mutants: list, timeout: float, full: bool = True) -> None:
        if not mutants:
            return
        if not self.pool_threads:
            self.start_pool()
        for m in mutants:
            self.mutant_status.pop(m["id"], None)
            self.mutant_full.pop(m["id"], None)
        self.pool_wait(self.pool_submit(mutants, foreground=True, full=full), timeout)

    def mutation_stats(self) -> dict:
        stats = {}
        for st in list(self.mutant_status.values()):
            stats[st] = stats.get(st, 0) + 1
        return stats

    def fresh_round(self) -> None:
        self.llm.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": self.round_context},
                             {"role": "assistant", "content": "Understood. Send the first round."}]
        self.round_files = set()

    def files_note(self) -> str:
        names = sorted(self.case_sources)
        return "Existing case files (do not reuse these names): " + ", ".join(n + ".py" for n in names[-80:])

    def snapshot(self, label: str) -> None:
        root = os.getenv("TG_SNAPSHOT_DIR")
        if not root:
            return
        try:
            files, _ = self.build_suite()
            self.snap_n = getattr(self, "snap_n", 0) + 1
            d = os.path.join(root, "%03d_%s_%ds_%.4f" % (self.snap_n, label, time.time() - self.t_start, self.llm.spent))
            os.makedirs(d, exist_ok=True)
            for rel, text in files.items():
                with open(os.path.join(d, rel), "w") as fh:
                    fh.write(text)
        except Exception:
            pass

    def mutation_phase(self, until_frac: float) -> None:
        self.refresh_mutants()
        self.start_pool()
        log("[MUT] %d candidate changes over %d files, %s" % (len(self.mutants), len(self.cov_lines),
                                                               self.mutation_stats()))
        for sweep in range(COVERAGE_SWEEPS + 1):
            self.mutation_rounds(until_frac)
            if sweep == COVERAGE_SWEEPS or self.frac() > until_frac - 0.08 or not self.llm.affordable():
                break
            cov = self.coverage_report(detail=True)
            if not cov:
                break
            log("[RUN] coverage second look (sweep %d)" % (sweep + 1))
            self.fresh_round()
            self.converse(self.files_note() + "\n\nCode in the scope that no case "
                          "runs yet is a blind spot; these lines never run:\n\n" + cov +
                          "\n\nAdd cases (in new case files) that reach the in-scope parts of this code through the "
                          "public API, especially error paths and rarely used options. Ignore anything outside the "
                          "task's scope. Reply DONE when finished.", max_turns=3, until_frac=until_frac)
            self.refresh_mutants()
            end = time.time() + 60
            while self.pool_pending() and time.time() < end and self.left() > 60:
                self.pool_wait({"left": 1}, 5)
        self.extra_looks(until_frac)
        if self.kinds == 1 and self.frac() < SECOND_KINDS_FRAC and self.llm.affordable():
            self.kinds = 2
            n = self.refresh_mutants()
            log("[MUT] further kinds of changes: %d new" % n)
            if n:
                self.mutation_rounds(until_frac)
                self.extra_looks(until_frac)

    def extra_looks(self, until_frac: float) -> None:
        looked, rounds, set_aside_look = set(), 0, False
        while rounds < EXTRA_LOOKS and self.frac() < min(until_frac, EXTRA_LOOK_FRAC) and self.llm.affordable():
            alive = [m for m in self.mutants.values() if self.mutant_status.get(m["id"]) == "survived"
                     and m["id"] not in self.skipped and m["id"] not in looked]
            if not alive and not set_aside_look and self.frac() < SET_ASIDE_LOOK_FRAC:
                set_aside_look = True
                alive = [m for m in self.mutants.values() if self.mutant_status.get(m["id"]) == "survived"
                         and m["id"] not in looked]
                for m in alive:
                    self.skipped.discard(m["id"])
            if not alive:
                break
            stats = self.function_stats()
            alive.sort(key=lambda m: self.mutant_priority(m, stats))
            candidates = _pick_batch(alive, MUTANTS_PER_ROUND * 2)
            self.evaluate_now(candidates, min(90.0, max(20.0, self.left() * 0.06)))
            batch = _pick_batch([m for m in candidates if self.mutant_status.get(m["id"]) == "survived"],
                                MUTANTS_PER_ROUND)
            looked.update(m["id"] for m in candidates)
            if not batch:
                continue
            rounds += 1
            self.count_round(batch)
            log("[MUT] extra look %d: %d changes" % (rounds, len(batch)))
            for m in batch:
                self.shown.add(m["id"])
            self.fresh_round()
            replies = self.converse(self.files_note() + "\n\n" + self.mutation_message(batch, [], True) + "\n\n" +
                                    THIRD_LOOK, max_turns=3, until_frac=until_frac, batch_mode=True)
            if not replies:
                break
            for reply in replies:
                for line in re.findall(r"(?im)^\W*SKIP\W*:?(.*)$", reply):
                    self.skipped.update(re.findall(r"\bm\d+\b", line))
            self.evaluate_now([m for m in batch if m["id"] not in self.skipped],
                              min(90.0, max(20.0, self.left() * 0.06)))
            self.note_feedback(batch)
            self.refresh_mutants()
            self.snapshot("extra")
            log("[MUT] extra look %d: %d of %d now detected" % (
                rounds, sum(1 for m in batch if self.mutant_status.get(m["id"]) == "killed"), len(batch)))

    def mutation_rounds(self, until_frac: float) -> None:
        last_batch = []
        stage = "first"
        idle = 0
        retry_rounds = 0
        while self.frac() < until_frac and self.llm.affordable():
            self.drop_import_fragile()
            if stage == "first" and idle >= IDLE_STOP and self.pool_pending() <= max(5, 0.03 * len(self.mutants)):
                log("[MUT] %d rounds in a row without a new detection; moving on" % idle)
                stage, idle = "retry", 0
            if stage == "retry" and retry_rounds >= RETRY_ROUNDS:
                break
            if stage == "first":
                pool = [m for m in self.mutants.values() if (m["id"] not in self.shown or m["id"] in self.doubted)
                        and m["id"] not in self.skipped]
            elif stage == "retry":
                pool = [m for m in self.mutants.values() if m["id"] not in self.skipped and m["id"] not in self.retried]
            else:
                pool = [m for m in self.mutants.values() if m["id"] not in self.retried]
            alive = [m for m in pool if self.mutant_status.get(m["id"]) == "survived"]
            if not alive:
                if self.pool_pending() and self.left() > 60:
                    self.pool_wait({"left": 1}, 10)
                    continue
                if stage == "first":
                    stage = "retry"
                    continue
                break
            stats = self.function_stats()
            alive.sort(key=lambda m: self.mutant_priority(m, stats))
            candidates = _pick_batch(alive, MUTANTS_PER_ROUND * 2)
            self.evaluate_now(candidates, min(90.0, max(20.0, self.left() * 0.06)))
            batch = _pick_batch([m for m in candidates if self.mutant_status.get(m["id"]) == "survived"],
                                MUTANTS_PER_ROUND)
            if not batch:
                continue
            retry_mode = stage != "first"
            for m in batch:
                self.shown.add(m["id"])
                if retry_mode:
                    self.retried.add(m["id"])
                    self.skipped.discard(m["id"])
            if retry_mode:
                retry_rounds += 1
            self.count_round(batch)
            text = self.mutation_message(batch, last_batch, retry_mode)
            self.fresh_round()
            replies = self.converse(self.files_note() + "\n\n" + text, max_turns=3, until_frac=until_frac,
                                    batch_mode=True)
            if not replies:
                break
            for m in batch:
                if m["id"] in self.doubted:
                    self.doubted.discard(m["id"])
                    self.doubt_shown.add(m["id"])
            for reply in replies:
                for line in re.findall(r"(?im)^\W*SKIP\W*:?(.*)$", reply):
                    for mid in re.findall(r"\bm\d+\b", line):
                        m = self.mutants.get(mid)
                        if m is not None and stage == "first" and mid not in self.doubt_shown and self.is_named(m):
                            self.doubted.add(mid)
                        else:
                            self.skipped.add(mid)
            self.evaluate_now([m for m in batch if m["id"] not in self.skipped],
                              min(90.0, max(20.0, self.left() * 0.06)))
            self.note_feedback(batch)
            detected = sum(1 for m in batch if self.mutant_status.get(m["id"]) == "killed")
            idle = 0 if detected else idle + 1
            self.refresh_mutants()
            self.snapshot("round")
            last_batch = batch
        log("[MUT] end: %s, skipped %d, queued %d" % (self.mutation_stats(), len(self.skipped), self.pool_pending()))

    def drop_import_fragile(self) -> None:
        for test_mod, change in list(self.import_fragile.items()):
            if test_mod in self.import_fragile_done:
                continue
            self.import_fragile_done.add(test_mod)
            module = next((n for n in self.case_sources if self.test_module(n) == test_mod), None)
            if module is None or module in self.module_problems:
                continue
            self.module_problems[module] = ["runs library code when the file is imported"]
            for key in self.usable_keys(module):
                self.excluded[key] = "its file runs library code when imported"
            self.notes.append(
                "%s.py was removed from the suite: it runs library code when the file is imported, and under a small "
                "change of the library (%s) the file could not be imported at all, which breaks the whole test run. "
                "Write its cases again in a new file; create library objects, and classes built on library classes, "
                "inside helper functions that the cases call." % (module, change))
            log("[MUT] %s.py dropped: fails to import under a change (%s)" % (module, change[:120]))
            self.rebuild_coverage()

    def mutant_priority(self, m: dict, stats: dict | None = None) -> tuple:
        if stats is None:
            stats = self.function_stats()
        killed, survived = stats.get((m["file"], m["func"]), (0, 0))
        weak = survived >= WEAK_FUNCTION * (killed + survived)
        return (m["id"] not in self.doubted, self.set_aside(m), m["file"] not in self.scope_files,
                self.func_rounds.get((m["file"], m["func"]), 0), not weak, not self.is_named(m),
                m.get("swap") is None, int(m["id"][1:]))

    def is_named(self, m: dict) -> bool:
        words = self.scope_words()
        return any(p in words for p in m["func"].split(".") if p and p != "<module>")

    def function_stats(self) -> dict:
        stats = {}
        for mid, status in list(self.mutant_status.items()):
            m = self.mutants.get(mid)
            if m is None or status not in ("killed", "survived"):
                continue
            k, s = stats.get((m["file"], m["func"]), (0, 0))
            stats[(m["file"], m["func"])] = (k + (status == "killed"), s + (status == "survived"))
        return stats

    def set_aside(self, m: dict) -> bool:
        key = (m["file"], m["func"])
        stats = self.func_feedback.get(key)
        return bool(stats) and stats[0] >= 3 and stats[1] == 0

    def count_round(self, batch: list) -> None:
        for key in {(m["file"], m["func"]) for m in batch}:
            self.func_rounds[key] = self.func_rounds.get(key, 0) + 1

    def note_feedback(self, batch: list) -> None:
        for m in batch:
            key = (m["file"], m["func"])
            stats = self.func_feedback.setdefault(key, [0, 0])
            if m["id"] in self.skipped:
                stats[0] += 1
            elif self.mutant_status.get(m["id"]) == "killed":
                stats[1] += 1

    def scope_words(self) -> set:
        if not hasattr(self, "_scope_words"):
            self._scope_words = set(re.findall(r"[A-Za-z_]\w+", self.scope_text()))
        return self._scope_words

    def function_source(self, m: dict, limit: int = 60) -> str:
        try:
            with open(os.path.join(self.src_root, m["file"]), errors="replace") as fh:
                lines = fh.read().split("\n")
        except OSError:
            return ""
        a, b = m.get("func_span") or (m["line"], m["line"])
        if m.get("module_level") or b - a < 3:
            a, b = max(1, m["line"] - 6), min(len(lines), m["line"] + 6)
        if b - a + 1 > limit:
            a = max(a, m["line"] - limit // 2)
            b = min(b, a + limit - 1)
        return "\n".join("%5d%s %s" % (i, ">" if i == m["line"] else " ", lines[i - 1]) for i in range(a, b + 1))

    def mutation_message(self, batch: list, last_batch: list, retry_mode: bool = False) -> str:
        parts = []
        if last_batch:
            killed = [m["id"] for m in last_batch if self.mutant_status.get(m["id"]) == "killed"]
            alive = [m["id"] for m in last_batch if self.mutant_status.get(m["id"]) == "survived"
                     and m["id"] not in self.skipped]
            parts.append("Result of the previous round: now detected %s; still undetected %s." % (
                ", ".join(killed) or "none", ", ".join(alive) or "none"))
        stats = self.mutation_stats()
        parts.append(MUTATION_INTRO.format(tried=len(self.mutant_status), killed=stats.get("killed", 0),
                                           alive=stats.get("survived", 0)))
        if retry_mode:
            parts.append("Second look: the changes below are still undetected after an earlier round (some were "
                         "set aside as out of scope then). The code around each change is shown (the changed line "
                         "is marked with >). Work out which inputs reach that line and what the change does to the "
                         "result, and write cases for them; SKIP only changes that truly cannot alter in-scope "
                         "behaviour.")
        stats = self.function_stats()
        groups = {}
        for m in batch:
            groups.setdefault((m["file"], m["func"]), []).append(m)
        for (file, func), ms in groups.items():
            killed, survived = stats.get((file, func), (0, 0))
            head = "### %s in %s: %d of %d checked changes are undetected" % (func, file, survived, killed + survived)
            if survived and survived >= WEAK_FUNCTION * (killed + survived):
                head += (" (the cases run this code but do not pin down its results: check its results directly, "
                         "for every input class and option that reaches it)")
            items = [head]
            for m in ms:
                item = "[%s] line %d\n%s" % (m["id"], m["line"], m["show"])
                if m["id"] in self.doubt_shown and m["id"] not in self.retried:
                    item += ("\n  note: you set this change aside before, but `%s` is named in the task, so its "
                             "behaviour is in scope. Find inputs through the public API for which this change shows "
                             "(for a changed default: a call that leaves the argument out). SKIP only if none exists."
                             % m["func"].split(".")[-1])
                items.append(item)
            if retry_mode or file not in self.context_files:
                items.append("  code:\n" + self.function_source(ms[0]))
            parts.append("\n".join(items))
        return "\n\n".join(parts)

    def execute(self) -> None:
        self.setup()
        if not self.packages:
            log("[RUN] no package found; nothing to test")
            return
        context = self.build_context()
        log("[RUN] context %d chars, scope files: %s" % (len(context), ", ".join(self.scope_files[:8])))
        self.llm.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.round_context = context + "\n\n---\n\n" + ROUND_MODE
        items = _scope_items(self.statement)
        lo, hi = max(120, 3 * len(items)), max(300, 6 * len(items))
        size = SMALL_SCOPE if hi <= 300 else LARGE_SCOPE.format(n=len(items), lo=min(lo, 450), hi=min(hi, 900))
        log("[RUN] scope: %d named items; first suite: %s" % (
            len(items), "%d-%d cases" % (min(lo, 450), min(hi, 900)) if hi > 300 else "standard size"))
        self.converse(context + "\n\n---\n\n" + AUTHOR_TASK.format(size=size), max_turns=12, until_frac=0.42)
        self.checkpoint()
        self.refresh_mutants()
        self.start_pool()
        cov = self.coverage_report()
        if cov and self.frac() < 0.5 and self.llm.affordable():
            log("[RUN] coverage feedback: %d items" % (cov.count("\n") + 1))
            self.converse("Coverage of the library by the recorded cases. These parts never run:\n\n" + cov +
                          "\n\nAdd cases (in new case files) that reach the in-scope parts of this code through "
                          "the public API. Ignore anything outside the task's scope. Reply DONE when finished.",
                          max_turns=5, until_frac=0.52)
            self.checkpoint()
        if self.frac() < 0.85 and self.llm.affordable():
            self.mutation_phase(until_frac=0.86)

    def checkpoint(self) -> None:
        if self.left() > 30:
            self.verify_suite(confirm_runs=1)
            self.snapshot("checkpoint")
            log("[RUN] checkpoint: %d usable cases, $%.3f spent" % (
                sum(1 for k in self.records if k not in self.excluded), self.llm.spent))

    def dump_mutants(self) -> None:
        path = os.getenv("TG_DEBUG_MUTANTS")
        if not path:
            return
        try:
            rows = [{"id": m["id"], "file": m["file"], "line": m["line"], "func": m["func"], "kind": m["kind"],
                     "status": self.mutant_status.get(m["id"]), "full": self.mutant_full.get(m["id"]),
                     "shown": m["id"] in self.shown, "skipped": m["id"] in self.skipped,
                     "swap": bool(m.get("swap")), "repl": m["repl"].decode("utf-8", "replace")[:120],
                     "why": m.get("why", ""),
                     "cases": len(self.relevant_keys(m))} for m in self.mutants.values()]
            with open(path, "w") as fh:
                json.dump(rows, fh)
        except Exception:
            pass

    def finalize(self) -> str:
        self.stop_pool()
        self.stop_zygotes()
        self.drop_import_fragile()
        self.dump_mutants()
        files = {}
        try:
            if self.end_left() > 90:
                self.contract_check()
        except Exception:
            log("[CONTRACT] failed: %s" % traceback.format_exc()[-600:])
        try:
            if self.end_left() > 120:
                self.rewrite_check()
        except Exception:
            log("[REWRITE] failed: %s" % traceback.format_exc()[-600:])
        try:
            if self.end_left() > 40:
                files = self.verify_suite(confirm_runs=2)
        except Exception:
            log("[FINAL] verify failed: %s" % traceback.format_exc()[-800:])
        files = files or self.best_files or self.fallback_files()
        n_tests = sum(len(re.findall(r"(?m)^def test_", t)) for t in files.values())
        log("[FINAL] %d files, %d tests, $%.4f over %d calls" % (len(files), n_tests, self.llm.spent, self.llm.calls))
        return self.write_patch(files)

    def fallback_files(self) -> dict:
        if not self.packages:
            return {}
        lines = ['"""Smoke tests for the public API."""', "import importlib", ""]
        for p in self.packages[:1]:
            lines += ["", "def test_package_imports():", "    module = importlib.import_module(%r)" % p["name"],
                      "    assert module is not None", ""]
        return {"test_smoke.py": "\n".join(lines)}

    def write_patch(self, files: dict) -> str:
        tdir = os.path.join(self.repo, self.test_rel)
        existed = os.path.isdir(tdir)
        initial = self.initial_test_entries if self.initial_test_entries is not None else set()
        written = []
        patch = ""
        try:
            if existed:
                for entry in os.listdir(tdir):
                    if entry in initial:
                        continue
                    full = os.path.join(tdir, entry)
                    if os.path.isdir(full):
                        shutil.rmtree(full, ignore_errors=True)
                    else:
                        os.remove(full)
            os.makedirs(tdir, exist_ok=True)
            for rel, text in files.items():
                if rel in initial:
                    continue
                path = os.path.join(tdir, rel)
                with open(path, "w") as fh:
                    fh.write(text if text.endswith("\n") else text + "\n")
                written.append(path)
            idx = os.path.join(self.work, "index.tmp")
            env = dict(os.environ, GIT_INDEX_FILE=idx)
            subprocess.run(["git", "read-tree", "HEAD"], cwd=self.repo, env=env, capture_output=True, timeout=60)
            rels = [os.path.relpath(p, self.repo) for p in written]
            subprocess.run(["git", "add", "-f", "--"] + rels, cwd=self.repo, env=env, capture_output=True, timeout=60)
            r = subprocess.run(["git", "-c", "core.quotepath=off", "diff", "--cached", "--binary", "--no-color",
                                "--no-ext-diff", "HEAD", "--"] + rels, cwd=self.repo, env=env,
                               capture_output=True, timeout=60)
            patch = r.stdout.decode(errors="replace")
        except Exception:
            log("[PATCH] building the diff failed: %s" % traceback.format_exc()[-500:])
        finally:
            for path in written:
                try:
                    os.remove(path)
                except OSError:
                    pass
            if not existed:
                shutil.rmtree(tdir, ignore_errors=True)
        if not patch.strip():
            patch = _manual_patch(self.test_rel, {k: v for k, v in files.items() if k not in initial})
        return patch

def _seal(d: str) -> None:
    try:
        os.chmod(d, 0o555)
    except OSError:
        pass

def _remove_dir(d: str) -> None:
    try:
        os.chmod(d, 0o755)
    except OSError:
        pass
    shutil.rmtree(d, ignore_errors=True)

def _make_readable(root: str) -> None:
    for dirpath, dirs, fs in os.walk(root):
        os.chmod(dirpath, 0o755)
        for f in fs:
            os.chmod(os.path.join(dirpath, f), 0o644)

def _number(text: str) -> str:
    return "\n".join("%5d  %s" % (i + 1, line) for i, line in enumerate(text.split("\n")))

def _outline(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "(could not parse)"
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append("line %d: def %s(...)" % (node.lineno, node.name))
        elif isinstance(node, ast.ClassDef):
            out.append("line %d: class %s" % (node.lineno, node.name))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append("    line %d: def %s(...)" % (sub.lineno, sub.name))
    return "\n".join(out[:150]) or "(no functions or classes)"

def _ranges(lines: list) -> str:
    out = []
    start = prev = None
    for n in sorted(lines):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(str(start) if start == prev else "%d-%d" % (start, prev))
            start = prev = n
    if start is not None:
        out.append(str(start) if start == prev else "%d-%d" % (start, prev))
    return ", ".join(out)

def _walk_no_nested(stmts):
    stack = list(stmts)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))

def _lint_case(fn) -> str:
    if isinstance(fn, ast.AsyncFunctionDef):
        return "case functions must not be async (use asyncio.run inside a plain function)"
    if fn.decorator_list:
        return "case functions must not be decorated"
    a = fn.args
    params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg or a.kwarg or any(p != "tmp_path" for p in params):
        return "a case takes no parameters (or only `tmp_path`)"
    ret = _final_return(fn.body)
    if ret is None:
        return ("the last statement of a case must be `return <expression>` (it may also end a final `with` block or "
                "`try ... finally` without `except`)")
    for node in _walk_no_nested(fn.body):
        if isinstance(node, ast.Return) and node is not ret:
            return "line %d: a case may only return at its end" % node.lineno
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            return "line %d: a case must not be a generator or coroutine" % node.lineno
    return ""

def _final_return_list(body):
    if not body:
        return None
    last = body[-1]
    if isinstance(last, ast.Return):
        return body if last.value is not None else None
    if isinstance(last, ast.With) or (isinstance(last, ast.Try) and last.finalbody and not last.handlers
                                      and not last.orelse):
        return _final_return_list(last.body)
    return None

def _final_return(body):
    found = _final_return_list(body)
    return found[-1] if found is not None else None

def _library_helpers(tree, lib_aliases: set) -> set:
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    used = {k: {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} for k, n in defs.items()}
    tainted = set()
    changed = True
    while changed:
        changed = False
        for k, names in used.items():
            if k not in tainted and names & (lib_aliases | tainted):
                tainted.add(k)
                changed = True
    return tainted

def _import_time_library_use(node, lib_aliases: set, lib_helpers: set) -> str:
    def lib_ref(exprs):
        for e in exprs:
            for x in ast.walk(e):
                if isinstance(x, ast.Name) and x.id in lib_aliases:
                    return x.id
        return ""

    def helper_call(exprs):
        for e in exprs:
            for x in ast.walk(e):
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in lib_helpers:
                    return x.func.id + "()"
        return ""

    def helper_decorator(decorators):
        for d in decorators:
            base = d.func if isinstance(d, ast.Call) else d
            if isinstance(base, ast.Name) and base.id in lib_helpers:
                return base.id
        return ""

    def check_function(fn, owner):
        a = fn.args
        defaults = list(a.defaults) + [d for d in a.kw_defaults if d is not None]
        name = lib_ref(fn.decorator_list) or helper_call(fn.decorator_list) or helper_decorator(fn.decorator_list)
        if name:
            return "the decorator of `%s%s` (`%s`)" % (owner, fn.name, name)
        name = lib_ref(defaults) or helper_call(defaults)
        if name:
            return "a default value of `%s%s` (`%s`)" % (owner, fn.name, name)
        return ""

    def check_class(cls, owner):
        head = list(cls.bases) + [k.value for k in cls.keywords] + list(cls.decorator_list)
        name = lib_ref(head) or helper_call(head) or helper_decorator(cls.decorator_list)
        if name:
            return "class `%s%s` (its bases or decorators use `%s`)" % (owner, cls.name, name)
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                what = check_function(item, owner + cls.name + ".")
            elif isinstance(item, ast.ClassDef):
                what = check_class(item, owner + cls.name + ".")
            else:
                name = lib_ref([item]) or helper_call([item])
                what = "the body of class `%s%s` (`%s`)" % (owner, cls.name, name) if name else ""
            if what:
                return what
        return ""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return check_function(node, "")
    if isinstance(node, ast.ClassDef):
        return check_class(node, "")
    value = getattr(node, "value", None)
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)) and value is not None:
        name = lib_ref([value]) or helper_call([value])
        if name:
            return "this module-level value (`%s`)" % name
    return ""

_PROTOCOL_METHODS = {"__iter__", "__reversed__", "__len__", "__length_hint__", "__hash__", "__eq__", "__ne__", "__lt__",
                     "__le__", "__gt__", "__ge__", "__bool__", "__contains__", "__index__"}

def _counts_calls(fn) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.AugAssign):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr in ("append", "appendleft", "add", "extend", "insert"):
            return True
    return False

_FORBIDDEN_ATTRS = {"__code__", "__closure__", "__globals__", "__dict__", "__defaults__", "__kwdefaults__",
                    "__wrapped__", "__file__", "__spec__", "__loader__", "__cached__", "__path__",
                    "__doc__", "__qualname__", "__module__", "__annotations__", "__mro__", "__bases__",
                    "__subclasses__", "__signature__", "__text_signature__"}
_FORBIDDEN_CALLS = {"id": "object ids differ between runs", "hash": "hash values differ between runs",
                    "repr": "repr() output is not part of the contract", "globals": "process state",
                    "locals": "process state", "breakpoint": "interactive",
                    "vars": "it exposes implementation details (the attribute dictionary)",
                    "dir": "it lists private and imported names, which a refactoring may change",
                    "get_type_hints": "annotations are not behaviour"}
_FORBIDDEN_DOTTED = {
    "time.time": "the real clock", "time.monotonic": "the real clock", "time.perf_counter": "the real clock",
    "time.sleep": "sleeping", "time.process_time": "the real clock", "datetime.now": "the real clock",
    "datetime.utcnow": "the real clock", "datetime.today": "the real clock", "date.today": "the real clock",
    "os.environ": "environment variables", "os.getenv": "environment variables", "os.getcwd": "the working directory",
    "os.getpid": "process state", "random.random": "randomness", "random.randint": "randomness",
    "random.choice": "randomness", "random.shuffle": "randomness", "random.sample": "randomness",
    "random.uniform": "randomness", "inspect.getsource": "source code", "inspect.getsourcelines": "source code",
    "inspect.getfile": "file paths", "sys.modules": "process state", "importlib.reload": "process state",
    "uuid.uuid4": "randomness", "uuid.uuid1": "time and host",
    "inspect.signature": "annotations and other declaration details", "inspect.getfullargspec": "declaration details",
    "inspect.getmembers": "private and imported names", "inspect.getdoc": "docstrings",
    "inspect.getmodule": "module layout", "typing.get_type_hints": "annotations",
}

def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""

def _forbidden(node) -> str:
    if isinstance(node, ast.Attribute):
        a = node.attr
        if a in _FORBIDDEN_ATTRS:
            return "`.%s` is an implementation detail" % a
        if a == "__name__" and ((isinstance(node.value, ast.Call) and _dotted(node.value.func) == "type") or
                                (isinstance(node.value, ast.Attribute) and node.value.attr == "__class__")):
            return "the class name of a result is an implementation detail (compare with the public class instead)"
        if a.startswith("_") and not (a.startswith("__") and a.endswith("__")):
            if not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")):
                return "`.%s` is a private attribute" % a
        d = _dotted(node)
        for k, why in _FORBIDDEN_DOTTED.items():
            if d == k or d.endswith("." + k):
                return "`%s` depends on %s" % (d, why)
    if isinstance(node, ast.Name) and node.id == "__file__":
        return "`__file__` ties the test to file paths"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
        return "`%s()` is not allowed: %s" % (node.func.id, _FORBIDDEN_CALLS[node.func.id])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
            node.func.id in ("getattr", "hasattr", "setattr", "delattr") and len(node.args) >= 2 and \
            isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str) and \
            node.args[1].value.startswith("_"):
        return "`%s(..., %r)` reaches a private or special attribute" % (node.func.id, node.args[1].value)
    if isinstance(node, ast.FormattedValue) and node.conversion == ord("r"):
        return "`!r` formatting uses repr(), which is not part of the contract"
    return ""

def _uses_exception_text(node, name: str) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("str", "repr", "format"):
        return any(isinstance(a, ast.Name) and a.id == name for a in node.args)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == name:
        return node.attr in ("args", "message", "msg", "strerror")
    if isinstance(node, ast.FormattedValue) and isinstance(node.value, ast.Name) and node.value.id == name:
        return True
    return False

def _scope_items(statement: str) -> list:
    import builtins
    import keyword
    m = re.search(r"(?ims)^#{2,3}[ \t]*(?:in[ \t]+)?scope\b[^\n]*\n(.*?)(?=^#{1,3}[ \t]|\Z)", statement)
    if not m:
        return []
    common = set(dir(builtins)) | set(keyword.kwlist)
    names = []
    for tok in re.findall(r"`([^`\n]+)`", m.group(1)):
        for part in re.split(r"[\s,()\[\]{}=:*]+", tok):
            part = part.strip(".")
            if not re.fullmatch(r"[A-Za-z_][\w.]*", part) or re.search(r"\.(py|md|rst|txt|toml|cfg)$", part):
                continue
            name = part.split(".")[-1]
            if len(name) >= 2 and name not in common and not name.startswith("_") and name not in names:
                names.append(name)
    return names

def _owner_top(tree, target):
    line = getattr(target, "lineno", None)
    if line is None:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            if first <= line <= (node.end_lineno or node.lineno):
                return node.name
    return None

def _helper_users(tree, cases) -> dict:
    tops = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    refs = {name: {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and x.id in tops and x.id != name}
            for name, node in tops.items()}
    users = {}
    for c in cases:
        seen, todo = set(), list(refs.get(c, ()))
        while todo:
            h = todo.pop()
            if h in seen:
                continue
            seen.add(h)
            todo += refs.get(h, ())
        for h in seen:
            users.setdefault(h, []).append(c)
    return users

def _functions(tree):
    out = []

    def visit(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + node.name
                out.append((q, node))
                visit(node.body, q + ".")
            elif isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")
    visit(tree.body, "")
    return out

def _stmt_lines(body):
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        yield node.lineno
        for field in ("body", "orelse", "finalbody"):
            sub = getattr(node, field, None)
            if isinstance(sub, list):
                yield from _stmt_lines(sub)
        for h in getattr(node, "handlers", []) or []:
            yield from _stmt_lines(h.body)
        for c in getattr(node, "cases", []) or []:
            yield from _stmt_lines(c.body)

def _parse_reply(reply: str, finish):
    blocks = []
    pos = 0
    pattern = re.compile(r"```([^\n`]*)\n(.*?)\n?```", re.S)
    for m in pattern.finditer(reply):
        info, body = m.group(1).strip(), m.group(2)
        pos = m.end()
        low = info.lower()
        name = None
        mm = re.search(r"(cases_[A-Za-z0-9_]+)\.py", info)
        if mm:
            name = mm.group(1).lower() + ".py"
        lines = body.split("\n")
        first_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first_idx is not None:
            first = lines[first_idx].strip()
            mm = re.fullmatch(r"(?:#\s*)?(?:file(?:name)?:?\s*)?`?(cases_[A-Za-z0-9_]+)\.py`?:?", first)
            if mm:
                name = name or mm.group(1).lower() + ".py"
                del lines[first_idx]
                body = "\n".join(lines)
        is_python = low.startswith("python") or low in ("py", "") or name is not None
        if is_python:
            body = "\n".join(line for line in body.split("\n")
                              if not re.match(r"\s*(SKIP\s*:|SKIP\s+m\d|DONE\s*$)", line))
        if is_python and re.search(r"EXCEPTION_CLASSES\s*=\s*True", info) and \
                not re.search(r"(?m)^EXCEPTION_CLASSES\s*=", body):
            body = "EXCEPTION_CLASSES = True\n" + body
        if name:
            blocks.append(("case", name, body.strip("\n") + "\n"))
        elif low in ("bash", "sh", "shell", "console"):
            if body.strip():
                blocks.append(("bash", None, body.strip()))
        elif is_python and re.search(r"(?m)^def case_\w+\(", body):
            blocks.append(("case-unnamed", None, body.strip("\n") + "\n"))
    truncated = "```" in reply[pos:] or finish == "length"
    done = bool(re.search(r"(?m)^\W*DONE\W*$", reply))
    return blocks, done, truncated

def _test_names(case_source: str) -> dict:
    tree = ast.parse(case_source)
    used = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    names = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("case_"):
            name = "test_" + node.name[len("case_"):]
            while name in used:
                name += "_"
            used.add(name)
            names[node.name] = name
    return names

def _test_module_source(case_source: str, topic: str, names: dict, keep: set, records: dict | None = None) -> str:
    tree = ast.parse(case_source)
    have_math = have_pytest = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            have_math |= any(a.name == "math" and not a.asname for a in node.names)
            have_pytest |= any(a.name == "pytest" and not a.asname for a in node.names)
    header = [ast.Expr(ast.Constant("Regression tests: %s." % topic.replace("_", " ")))]
    if not have_math:
        header.append(ast.Import(names=[ast.alias(name="math")]))
    if not have_pytest:
        header.append(ast.Import(names=[ast.alias(name="pytest")]))
    for rec in (records or {}).values():
        cls = rec.get("cls") if rec.get("status") == "raises" else None
        if cls and "." in cls:
            header.append(ast.Import(names=[ast.alias(name=cls.rpartition(".")[0])]))
    body = []
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.FunctionDef) and node.name.startswith("case_"):
            if node.name not in keep or node.name not in names:
                continue
            rec = records.get(node.name) if records is not None else {"status": "record"}
            if rec is None:
                continue
            body.append(_case_to_test(node, rec, names[node.name]))
            continue
        body.append(node)
    seen_imports, uniq = set(), []
    for node in header:
        key = ast.dump(node)
        if key not in seen_imports:
            seen_imports.add(key)
            uniq.append(node)
    helper = ast.parse(SAME_SRC).body
    module = ast.Module(body=uniq + body + helper, type_ignores=[])
    ast.fix_missing_locations(module)
    text = ast.unparse(module) + "\n"
    compile(text, "<generated>", "exec")
    return text

def _case_to_test(fn: ast.FunctionDef, rec: dict, name: str) -> ast.FunctionDef:
    stmts = [copy.deepcopy(s) for s in fn.body]
    doc = None
    if len(stmts) > 1 and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant) \
            and isinstance(stmts[0].value.value, str):
        doc = stmts.pop(0)
    local_names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    var = "actual"
    while var in local_names:
        var += "_"
    holder = _final_return_list(stmts)
    if holder is stmts:
        ret = stmts.pop()
    else:
        ret = holder[-1]
        holder[-1] = ast.Assign(targets=[ast.Name(var, ast.Store())], value=ret.value)
        ret = ast.Return(ast.Name(var, ast.Load()))
    if rec["status"] == "raises":
        inner = stmts + ([ast.Expr(ret.value)] if holder is stmts else [])
        cls = ast.parse(rec.get("cls") or "Exception", mode="eval").body
        new_body = [ast.With(items=[ast.withitem(
            context_expr=ast.Call(func=ast.Attribute(ast.Name("pytest", ast.Load()), "raises", ast.Load()),
                                  args=[cls], keywords=[]),
            optional_vars=None)], body=inner)]
    else:
        if holder is stmts:
            stmts.append(ast.Assign(targets=[ast.Name(var, ast.Store())], value=ret.value))
        if rec["status"] == "record":
            stmts.append(ast.Return(ast.Name(var, ast.Load())))
        else:
            expected = ast.parse(rec["src"], mode="eval").body
            stmts.append(ast.Assert(test=ast.Call(func=ast.Name("_same_value", ast.Load()),
                                                  args=[ast.Name(var, ast.Load()), expected], keywords=[]), msg=None))
        new_body = stmts
    if doc is not None:
        new_body.insert(0, doc)
    new = ast.FunctionDef(name=name, args=copy.deepcopy(fn.args), body=new_body, decorator_list=[], returns=None,
                          type_comment=None)
    if sys.version_info >= (3, 12):
        new.type_params = []
    return new

def _manual_patch(test_rel: str, files: dict) -> str:
    out = []
    for rel in sorted(files):
        path = "%s/%s" % (test_rel.strip("/"), rel)
        text = files[rel] if files[rel].endswith("\n") else files[rel] + "\n"
        lines = text.split("\n")[:-1]
        out.append("diff --git a/%s b/%s\nnew file mode 100644\n--- /dev/null\n+++ b/%s\n@@ -0,0 +1,%d @@\n" % (
            path, path, path, len(lines)))
        out.append("".join("+" + line + "\n" for line in lines))
    return "".join(out)

def _pick_batch(alive: list, n: int) -> list:
    per_func, lines, funcs, batch = {}, set(), [], []
    for second_pass in (False, True):
        for m in alive:
            k = (m["file"], m["func"])
            if m in batch or per_func.get(k, 0) >= PER_FUNCTION:
                continue
            if k not in per_func and len(funcs) >= FUNCS_PER_ROUND:
                continue
            if not second_pass and (m["file"], m["line"]) in lines:
                continue
            if k not in per_func:
                funcs.append(k)
            per_func[k] = per_func.get(k, 0) + 1
            lines.add((m["file"], m["line"]))
            batch.append(m)
            if len(batch) >= n:
                return batch
    return batch

_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
             ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Div,
             ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult, ast.LShift: ast.RShift, ast.RShift: ast.LShift,
             ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd, ast.BitXor: ast.BitOr}
_LOG_CALL = re.compile(r"(^|\.)(warn|warning|warnings|debug|info|error|exception|critical|log|print|trace)$")
_EXC_SWAP = {"ValueError": "TypeError", "TypeError": "ValueError", "KeyError": "IndexError",
             "IndexError": "KeyError", "AttributeError": "TypeError", "RuntimeError": "ValueError"}
_TYPING_CALLS = {"TypeVar", "NewType", "ParamSpec", "TypeVarTuple", "cast", "overload"}
_ORDER_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_TWINS = {}
for _a, _b in (("min", "max"), ("any", "all"), ("startswith", "endswith"), ("lstrip", "rstrip"), ("ljust", "rjust"),
               ("find", "rfind"), ("index", "rindex"), ("split", "rsplit"), ("partition", "rpartition"),
               ("upper", "lower"), ("floor", "ceil"), ("bisect_left", "bisect_right"),
               ("insort_left", "insort_right"), ("appendleft", "append"), ("popleft", "pop")):
    _TWINS[_a], _TWINS[_b] = _b, _a
_NO_SWAP_CALLS = {"isinstance", "issubclass", "getattr", "setattr", "hasattr", "super", "range", "enumerate"}
_SKIP_FUNCS = {"__repr__", "__str__", "__format__", "__rich_repr__", "__del__", "__hash__"}
_NOT_PACKAGES = {"tests", "test", "testing", "docs", "doc", "examples", "example", "scripts", "tools", "ci", "build",
                 "dist"}
_KEEPS_FUNCTION = {"property", "setter", "getter", "deleter", "staticmethod", "classmethod", "abstractmethod",
                   "cached_property"}

def _contract_variant(source: bytes):
    text = source.decode("utf-8")
    tree = ast.parse(text)
    starts = [0]
    for line in source.split(b"\n"):
        starts.append(starts[-1] + len(line) + 1)

    def off(lineno, col):
        return starts[lineno - 1] + col

    in_fstring = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for sub in ast.walk(node):
                in_fstring.add(id(sub))
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            for sub in ast.walk(node.exc):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value and id(sub) not in in_fstring:
                    edits.append((off(sub.lineno, sub.col_offset), off(sub.end_lineno, sub.end_col_offset),
                                  repr(sub.value + " (reworded)")))
        if isinstance(node, ast.ClassDef):
            is_exc = node.name.endswith(("Error", "Exception")) or any(
                (_dotted(b) or "").split(".")[-1].endswith(("Error", "Exception", "Warning")) for b in node.bases)
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (item.name == "__repr__" or (is_exc and item.name == "__str__")):
                    for sub in _walk_no_nested(item.body):
                        if isinstance(sub, ast.Return) and sub.value is not None:
                            v = sub.value
                            edits.append((off(v.lineno, v.col_offset), off(v.end_lineno, v.end_col_offset),
                                          "(str(%s) + ' ~')" % ast.unparse(v)))
    if not edits:
        return None
    edits.sort(reverse=True)
    out, last = source, None
    for start, end, repl in edits:
        if last is not None and end > last:
            continue
        out = out[:start] + repl.encode("utf-8") + out[end:]
        last = start
    try:
        compile(out, "<variant>", "exec", dont_inherit=True)
    except Exception:
        return None
    return out

_DYNAMIC_NAMES = {"locals", "vars", "eval", "exec", "dir", "globals", "_getframe", "currentframe", "f_locals"}

def _rewrite_variant(source: bytes):
    text = source.decode("utf-8")
    tree = ast.parse(text)
    starts = [0]
    for line in source.split(b"\n"):
        starts.append(starts[-1] + len(line) + 1)
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)} | set(dir(__builtins__))
    edits = []
    for _, fn in _functions(tree):
        inner = [n for stmt in fn.body for n in ast.walk(stmt)]
        if any((isinstance(n, ast.Name) and n.id in _DYNAMIC_NAMES) or
               (isinstance(n, ast.Attribute) and n.attr in _DYNAMIC_NAMES) or
               isinstance(n, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)) for n in inner):
            continue
        a = fn.args
        params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        params |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
        nested = set()
        for n in inner:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nested.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.JoinedStr)):
                for x in ast.walk(n):
                    if isinstance(x, ast.Name):
                        nested.add(x.id)
                    elif isinstance(x, ast.arg):
                        nested.add(x.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                nested.add(n.name)
            elif hasattr(ast, "MatchAs") and isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
                nested.add(n.name)
            elif hasattr(ast, "MatchMapping") and isinstance(n, ast.MatchMapping) and n.rest:
                nested.add(n.rest)
        comp_targets = {x.id for n in inner if isinstance(n, ast.comprehension)
                        for x in ast.walk(n.target) if isinstance(x, ast.Name)}
        assigned = {n.id for n in inner if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        local = {x for x in assigned - params - nested
                 if x not in comp_targets or x in {n.id for n in _bound_outside_comprehensions(fn.body)}}
        if not local:
            continue
        rename = {}
        for name in sorted(local):
            new = "%s_rw" % name
            while new in taken:
                new += "_"
            taken.add(new)
            rename[name] = new
        for n in inner:
            if isinstance(n, ast.Name) and n.id in rename:
                edits.append((starts[n.lineno - 1] + n.col_offset, starts[n.end_lineno - 1] + n.end_col_offset,
                              rename[n.id]))
    if not edits:
        return None
    out = source
    for start, end, new in sorted(set(edits), reverse=True):
        out = out[:start] + new.encode("utf-8") + out[end:]
    try:
        compile(out, "<variant>", "exec", dont_inherit=True)
    except Exception:
        return None
    return out

def _bound_outside_comprehensions(body):
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node
        stack.extend(ast.iter_child_nodes(node))

def _call_index(tree) -> tuple:
    spans = [(q, f.lineno, f.end_lineno or f.lineno) for q, f in _functions(tree)]

    def owner(line):
        best = None
        for q, a, b in spans:
            if a <= line <= b and (best is None or a >= best[1]):
                best = (q, a)
        return best[0] if best else "module level"
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name:
                calls.setdefault(name, []).append(node)
    return calls, owner

def _difference(a: dict, x) -> str:
    if x is None:
        return "; a later run did not finish"
    if x["status"] != a["status"]:
        return "; one run %s, another %s" % (
            "returned a value" if a["status"] == "value" else "raised " + str(a.get("exc")),
            "returned a value" if x["status"] == "value" else "raised " + str(x.get("exc")))
    s, t = a.get("src") or "", x.get("src") or ""
    i = next((k for k in range(min(len(s), len(t))) if s[k] != t[k]), min(len(s), len(t)))
    lo = max(0, i - 50)
    return ("; first difference: one run gave ...%s... and another ...%s... (keep the case's inputs and calls as they are "
            "and leave only the varying part out of the result, such as ids or counters that setup calls return, "
            "times, or process details)" % (s[lo:i + 40], t[lo:i + 40]))

def _default_use_note(index_of_calls, fn, param: str, index, method: bool) -> str:
    calls, owner = index_of_calls
    omit, passes = [], 0
    for node in calls.get(fn.name, ()):
        f = node.func
        pos = -1 if index is None else index - (1 if method and isinstance(f, ast.Attribute) else 0)
        if any(k.arg == param or k.arg is None for k in node.keywords) or \
                any(isinstance(a, ast.Starred) for a in node.args) or (pos >= 0 and len(node.args) > pos):
            passes += 1
        else:
            omit.append("line %d (in %s)" % (node.lineno, owner(node.lineno)))
    if omit:
        return "  the default is used by the calls that leave `%s` out: %s" % (
            param, ", ".join(omit[:6]) + (", ..." if len(omit) > 6 else ""))
    if passes:
        return ("  every call of `%s` in this file passes `%s`; the default only matters when a caller of the public "
                "API calls `%s(...)` without `%s`" % (fn.name, param, fn.name, param))
    return ("  the default is used whenever `%s(...)` is called without `%s`: write a case that calls `%s` with the "
            "other arguments only (or none) and returns a result that depends on `%s`" % (fn.name, param, fn.name, param))

def _generate_mutants(rel: str, source: bytes, covered: set, more: bool = False) -> list:
    call_index = [None]
    text = source.decode("utf-8")
    tree = ast.parse(text)
    line_starts = [0]
    for line in source.split(b"\n"):
        line_starts.append(line_starts[-1] + len(line) + 1)
    lines_text = text.split("\n")
    out = []
    seen = set()
    func_spans = {}
    for qual, fn in _functions(tree):
        func_spans[qual] = (fn.lineno, fn.end_lineno or fn.lineno)
    outer = [None]
    parts = rel[:-3].split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    modname = ".".join(parts)

    def span(node):
        return (line_starts[node.lineno - 1] + node.col_offset,
                line_starts[node.end_lineno - 1] + node.end_col_offset)

    def add(node, replacement: str, stmt: tuple, func: str, kind: str, scope_lines=None, module_level=False,
            swap=None):
        fspan = func_spans.get(func, (node.lineno, node.end_lineno))
        start, end = span(node)
        repl = replacement.encode("utf-8")
        if source[start:end] == repl or (start, end, repl) in seen:
            return
        seen.add((start, end, repl))
        if swap is None:
            swap = outer[0]
        a, b = node.lineno, node.end_lineno
        head = source[line_starts[a - 1]:start].decode("utf-8", errors="replace")
        tail = source[end:max(end, line_starts[b] - 1)].decode("utf-8", errors="replace")
        before = lines_text[a - 1:b][:4]
        after = (head + replacement + tail).split("\n")[:4]
        show = "\n".join(["  was:  " + x.strip() for x in before] + ["  now:  " + x.strip() for x in after])
        out.append({"file": rel, "line": a, "func": func or "<module>", "start": start, "end": end, "repl": repl,
                    "show": show, "stmt_start": stmt[0], "stmt_end": stmt[1], "scope_lines": scope_lines,
                    "module_level": module_level, "kind": kind, "func_span": fspan,
                    "swap": None if module_level or scope_lines else swap})

    def expr_mutations(root, stmt, func, scope_lines=None, module_level=False):
        stack = [root]
        while stack:
            node = stack.pop()
            if isinstance(node, ast.JoinedStr):
                continue
            if isinstance(node, ast.Call) and (_LOG_CALL.search(_dotted(node.func) or "") or
                                               (_dotted(node.func) or "").split(".")[-1] in _TYPING_CALLS):
                continue
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.expr, ast.comprehension, ast.keyword, ast.arguments)):
                    stack.append(child)
            kw = {"scope_lines": scope_lines, "module_level": module_level}
            if isinstance(node, ast.Compare):
                for i, op in enumerate(node.ops):
                    rep = _CMP_SWAP.get(type(op))
                    if rep:
                        new = copy.deepcopy(node)
                        new.ops[i] = rep()
                        add(node, "(%s)" % ast.unparse(new), stmt, func, "compare", **kw)
            elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
                if isinstance(node.op, ast.Mod) and isinstance(node.left, (ast.Constant, ast.JoinedStr)):
                    continue
                new = copy.deepcopy(node)
                new.op = _BIN_SWAP[type(node.op)]()
                add(node, "(%s)" % ast.unparse(new), stmt, func, "arith", **kw)
            elif isinstance(node, ast.BoolOp):
                new = copy.deepcopy(node)
                new.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                add(node, "(%s)" % ast.unparse(new), stmt, func, "bool", **kw)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
                add(node, "(%s)" % ast.unparse(node.operand), stmt, func, "unary", **kw)
            elif isinstance(node, ast.IfExp):
                new = copy.deepcopy(node)
                new.test = ast.UnaryOp(ast.Not(), new.test)
                add(node, "(%s)" % ast.unparse(new), stmt, func, "condition", **kw)
            elif isinstance(node, ast.Call) and (node.keywords or any(isinstance(a, ast.Starred) for a in node.args)):
                removable = [("kw", i) for i in range(len(node.keywords))] + \
                            [("star", i) for i, a in enumerate(node.args) if isinstance(a, ast.Starred)]
                for what, i in removable[:3]:
                    new = copy.deepcopy(node)
                    if what == "kw":
                        del new.keywords[i]
                    else:
                        del new.args[i]
                    add(node, "(%s)" % ast.unparse(new), stmt, func, "argument", **kw)
            elif isinstance(node, ast.Constant):
                v = node.value
                if isinstance(v, bool):
                    add(node, repr(not v), stmt, func, "constant", **kw)
                elif isinstance(v, int):
                    add(node, "(%d)" % (0 if v == 1 else v + 1), stmt, func, "constant", **kw)
                elif isinstance(v, float):
                    add(node, "(%r)" % (v + 1.0), stmt, func, "constant", **kw)
                elif isinstance(v, str):
                    add(node, repr("" if v else "XX"), stmt, func, "constant", **kw)
                elif isinstance(v, bytes):
                    add(node, repr(b"" if v else b"XX"), stmt, func, "constant", **kw)
            if more and not module_level:
                more_mutations(node, stmt, func, kw)

    def more_mutations(node, stmt, func, kw):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _ORDER_OPS:
            for alt in _ORDER_OPS:
                if alt is not type(node.ops[0]):
                    new = copy.deepcopy(node)
                    new.ops[0] = alt()
                    add(node, "(%s)" % ast.unparse(new), stmt, func, "compare-all", **kw)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                                                                    ast.Mod)):
            if not (isinstance(node.op, ast.Mod) and isinstance(node.left, (ast.Constant, ast.JoinedStr))):
                add(node, "(%s)" % ast.unparse(node.left), stmt, func, "operand", **kw)
                add(node, "(%s)" % ast.unparse(node.right), stmt, func, "operand", **kw)
        elif isinstance(node, ast.BoolOp):
            for i in range(len(node.values)):
                rest = node.values[:i] + node.values[i + 1:]
                text = ast.unparse(rest[0]) if len(rest) == 1 else ast.unparse(ast.BoolOp(node.op, rest))
                add(node, "(%s)" % text, stmt, func, "operand", **kw)
        elif isinstance(node, ast.IfExp):
            add(node, "(%s)" % ast.unparse(node.body), stmt, func, "condition-fixed", **kw)
            add(node, "(%s)" % ast.unparse(node.orelse), stmt, func, "condition-fixed", **kw)
        elif isinstance(node, ast.Call):
            name = (_dotted(node.func) or "").split(".")[-1]
            if name in _TWINS:
                target = node.func.attr if isinstance(node.func, ast.Attribute) else name
                new = copy.deepcopy(node)
                if isinstance(new.func, ast.Attribute):
                    new.func.attr = _TWINS[target]
                else:
                    new.func = ast.Name(_TWINS[target], ast.Load())
                add(node, "(%s)" % ast.unparse(new), stmt, func, "twin", **kw)
            positional = [a for a in node.args if not isinstance(a, ast.Starred)]
            if len(positional) == len(node.args) and len(node.args) >= 2 and name not in _NO_SWAP_CALLS:
                for i in range(min(2, len(node.args) - 1)):
                    if ast.dump(node.args[i]) == ast.dump(node.args[i + 1]):
                        continue
                    new = copy.deepcopy(node)
                    new.args[i], new.args[i + 1] = new.args[i + 1], new.args[i]
                    add(node, "(%s)" % ast.unparse(new), stmt, func, "argument-swap", **kw)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            sl = node.slice
            if isinstance(sl, ast.Slice):
                for part in ("lower", "upper"):
                    bound = getattr(sl, part)
                    if bound is not None and not isinstance(bound, ast.Constant):
                        for op in (ast.Add(), ast.Sub()):
                            new = copy.deepcopy(node)
                            setattr(new.slice, part, ast.BinOp(copy.deepcopy(bound), op, ast.Constant(1)))
                            add(node, "(%s)" % ast.unparse(new), stmt, func, "index", **kw)
            elif not isinstance(sl, (ast.Constant, ast.Tuple)) and not (
                    isinstance(sl, ast.UnaryOp) and isinstance(sl.operand, ast.Constant)):
                for op in (ast.Add(), ast.Sub()):
                    new = copy.deepcopy(node)
                    new.slice = ast.BinOp(copy.deepcopy(sl), op, ast.Constant(1))
                    add(node, "(%s)" % ast.unparse(new), stmt, func, "index", **kw)

    def is_docstring(s):
        return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)

    def header_range(s):
        body = getattr(s, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            return s.lineno, max(s.lineno, body[0].lineno - 1)
        return s.lineno, s.end_lineno

    def covered_range(a, b):
        return any(n in covered for n in range(a, b + 1))

    def visit(stmts, func, in_func, skip):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = (func + "." if func else "") + s.name
                fskip = skip or s.name in _SKIP_FUNCS
                first, last = s.body[0].lineno, s.end_lineno or s.lineno
                if not in_func:
                    kept = all((_dotted(d) or "").split(".")[-1] in _KEEPS_FUNCTION for d in s.decorator_list)
                    outer[0] = {"qual": qual, "name": s.name, "modname": modname,
                                "line": s.decorator_list[0].lineno if s.decorator_list else s.lineno} if kept else None
                if not fskip and covered_range(first, last):
                    body_lines = list(range(first, last + 1))
                    params = s.args.posonlyargs + s.args.args
                    method = bool(params) and params[0].arg in ("self", "cls")
                    with_defaults = [(params[len(params) - len(s.args.defaults) + j], d, True)
                                     for j, d in enumerate(s.args.defaults)]
                    with_defaults += [(p, d, False) for p, d in zip(s.args.kwonlyargs, s.args.kw_defaults)
                                      if d is not None]
                    for p, d, positional in with_defaults:
                        n0 = len(out)
                        expr_mutations(d, (s.lineno, s.lineno), qual, scope_lines=body_lines if not in_func else None)
                        if isinstance(d, ast.Constant) and d.value is None:
                            add(d, "0", (s.lineno, s.lineno), qual, "constant", scope_lines=body_lines if not in_func else None)
                        if len(out) > n0:
                            if call_index[0] is None:
                                call_index[0] = _call_index(tree)
                            note = _default_use_note(call_index[0], s, p.arg, params.index(p) if positional else None,
                                                     method)
                            for mm in out[n0:]:
                                if note:
                                    mm["show"] += "\n" + note
                visit(s.body, qual, True, fskip)
                if not in_func:
                    outer[0] = None
                continue
            if isinstance(s, ast.ClassDef):
                visit(s.body, (func + "." if func else "") + s.name, in_func, skip)
                continue
            if skip or is_docstring(s) or isinstance(s, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                                                         ast.Pass, ast.Assert)):
                continue
            if hasattr(ast, "Match") and isinstance(s, ast.Match):
                continue
            if isinstance(s, ast.If) and "TYPE_CHECKING" in ast.unparse(s.test):
                continue
            if isinstance(s, (ast.Assign, ast.AnnAssign)):
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                if any(isinstance(t, ast.Name) and t.id in ("__all__", "__version__", "__slots__", "__author__")
                       for t in targets):
                    continue
            a, b = header_range(s)
            module_level = not in_func
            is_cov = covered_range(a, b) if in_func else bool(covered)
            is_log = isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and \
                _LOG_CALL.search(_dotted(s.value.func) or "")
            if is_cov and not is_log:
                if in_func:
                    if isinstance(s, (ast.Assign, ast.AugAssign, ast.Expr, ast.Raise, ast.Delete)) or (
                            isinstance(s, ast.AnnAssign) and s.value is not None):
                        add(s, "pass", (a, b), func, "delete")
                    if isinstance(s, ast.Return) and s.value is not None and not (
                            isinstance(s.value, ast.Constant) and s.value.value is None):
                        add(s, "return None", (a, b), func, "return")
                    if isinstance(s, ast.Break):
                        add(s, "continue", (a, b), func, "loop")
                    if isinstance(s, ast.Continue):
                        add(s, "break", (a, b), func, "loop")
                    if isinstance(s, (ast.If, ast.While)):
                        add(s.test, "(not (%s))" % ast.unparse(s.test), (a, b), func, "condition")
                    if more and isinstance(s, ast.If):
                        add(s.test, "True", (a, b), func, "condition-fixed")
                        add(s.test, "False", (a, b), func, "condition-fixed")
                    if more and isinstance(s, ast.AugAssign):
                        add(s, "%s = %s" % (ast.unparse(s.target), ast.unparse(s.value)), (a, b), func, "assign")
                    if more and isinstance(s, ast.Try):
                        for h in s.handlers:
                            if h.type is not None:
                                add(h.type, "()", (h.lineno, h.lineno), func, "handler")
                    if isinstance(s, ast.AugAssign) and type(s.op) in _BIN_SWAP:
                        new = copy.deepcopy(s)
                        new.op = _BIN_SWAP[type(s.op)]()
                        add(s, ast.unparse(new), (a, b), func, "arith")
                    if isinstance(s, ast.Raise) and s.exc is not None:
                        target = s.exc.func if isinstance(s.exc, ast.Call) else s.exc
                        if isinstance(target, (ast.Name, ast.Attribute)):
                            name = _dotted(target).split(".")[-1]
                            if name and name[0].isupper():
                                add(target, _EXC_SWAP.get(name, "RuntimeError"), (a, b), func, "exception")
                if not isinstance(s, ast.Raise):
                    for field, value in ast.iter_fields(s):
                        if field in ("body", "orelse", "finalbody", "handlers", "cases", "decorator_list",
                                     "annotation", "returns", "type_comment", "type_params", "targets", "target"):
                            continue
                        for v in (value if isinstance(value, list) else [value]):
                            if isinstance(v, ast.expr):
                                expr_mutations(v, (a, b), func, module_level=module_level)
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(s, field, None)
                if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                    visit(sub, func, in_func, skip)
            for h in getattr(s, "handlers", []) or []:
                visit(h.body, func, in_func, skip)

    visit(tree.body, "", False, False)
    return out

def agent_main(input):
    statement = input.get("problem_statement", "") if isinstance(input, dict) else str(input)
    run = Run(statement)
    try:
        run.execute()
    except Exception:
        log("[RUN] failed: %s" % traceback.format_exc()[-2000:])
    try:
        patch = run.finalize()
    except Exception:
        log("[RUN] finalize failed: %s" % traceback.format_exc()[-2000:])
        patch = _manual_patch(run.test_rel, run.best_files or run.fallback_files())
    return patch

if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--tg-runner":
    _runner_main(sys.argv[2])
elif __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--tg-zygote":
    _zygote_main(sys.argv[2])
