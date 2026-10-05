"""Isolated subprocess runner: records case results, exception classes and line coverage; lists the API."""
from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import signal
import sys
import tempfile
import time
import traceback

_HAS_ITIMER = all(hasattr(signal, n) for n in ("SIGALRM", "setitimer", "ITIMER_REAL"))


class Unrecordable(Exception):
    pass


class _Timeout(BaseException):
    pass


def to_src(v, depth=0):
    if depth > 30:
        raise Unrecordable("nested too deeply")
    if v is None or isinstance(v, bool):
        return repr(v)
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
        return "complex(%s, %s)" % (to_src(v.real), to_src(v.imag))
    if isinstance(v, str):
        return repr(str(v))
    if isinstance(v, (bytes, bytearray)):
        return repr(bytes(v))
    if isinstance(v, tuple):
        it = [to_src(x, depth + 1) for x in v]
        return "(" + ", ".join(it) + ("," if len(it) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(to_src(x, depth + 1) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join("%s: %s" % (to_src(k, depth + 1), to_src(x, depth + 1)) for k, x in v.items()) + "}"
    if isinstance(v, (set, frozenset)):
        it = sorted(to_src(x, depth + 1) for x in v)
        if isinstance(v, frozenset):
            return "frozenset({%s})" % ", ".join(it) if it else "frozenset()"
        return "{%s}" % ", ".join(it) if it else "set()"
    raise Unrecordable("returned %s.%s, not plain data; convert it through the public API" % (
        type(v).__module__, type(v).__qualname__))


def exc_path(cls, packages):
    """Most specific *public* class path for an exception class (falls back to a builtin base)."""
    for k in cls.__mro__:
        if k.__module__ == "builtins":
            return k.__name__
        if k.__name__.startswith("_") or k.__module__.split(".")[0] not in packages:
            continue
        cands = sorted((n for n in list(sys.modules) if n.split(".")[0] in packages
                        and not any(p.startswith("_") for p in n.split("."))), key=lambda n: (n.count("."), n))
        for n in cands:
            if getattr(sys.modules.get(n), k.__name__, None) is k:
                return n + "." + k.__name__
    return "Exception"


class Tracer:
    def __init__(self, root):
        self.root, self.lines = root, set()

    def g(self, frame, event, arg):
        if frame.f_code.co_filename.startswith(self.root):
            self.lines.add((frame.f_code.co_filename, frame.f_lineno))
            return self.l
        return None

    def l(self, frame, event, arg):
        if event == "line":
            self.lines.add((frame.f_code.co_filename, frame.f_lineno))
        return self.l


def _api(names, packages):
    out = []
    for name in names:
        try:
            m = importlib.import_module(name)
        except BaseException as e:
            out.append({"module": name, "error": "%s: %s" % (type(e).__name__, e)})
            continue
        items = []
        allv = getattr(m, "__all__", None)
        for n in (allv if isinstance(allv, (list, tuple)) else sorted(vars(m))):
            if n.startswith("_") or not hasattr(m, n):
                continue
            o = getattr(m, n)
            if inspect.ismodule(o) or (allv is None and getattr(o, "__module__", "").split(".")[0] not in packages
                                       and (callable(o) or inspect.isclass(o))):
                continue
            try:
                sig = str(inspect.signature(o)) if callable(o) else ""
            except (TypeError, ValueError):
                sig = "(...)"
            doc = (inspect.getdoc(o) or "").split("\n")[0][:120] if callable(o) else ""
            kind = "class" if inspect.isclass(o) else "def" if callable(o) else "const"
            entry = {"name": n, "kind": kind, "sig": sig, "doc": doc}
            if inspect.isclass(o):
                ms = []
                for mn, mv in sorted(vars(o).items()):
                    if mn.startswith("_") and mn not in ("__call__", "__iter__", "__len__", "__getitem__",
                                                         "__contains__", "__eq__", "__enter__", "__exit__"):
                        continue
                    f = mv.__func__ if isinstance(mv, (staticmethod, classmethod)) else mv
                    if isinstance(mv, property):
                        ms.append(mn + " (property)")
                    elif callable(f):
                        try:
                            ms.append(mn + str(inspect.signature(f)))
                        except (TypeError, ValueError):
                            ms.append(mn + "(...)")
                entry["members"] = ms
            items.append(entry)
        out.append({"module": name, "file": getattr(m, "__file__", None), "items": items})
    return out


def main(cfg_path):
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    for p in reversed(cfg.get("sys_path", [])):
        sys.path.insert(0, p)
    packages = set(cfg.get("packages") or [])
    out = open(cfg["out"], "w", buffering=1)

    def emit(o):
        out.write(json.dumps(o) + "\n")

    if cfg["mode"] == "api":
        emit({"kind": "api", "data": _api(cfg["modules"], packages)})
        emit({"kind": "end"})
        return
    tracer = Tracer(cfg["trace_root"]) if cfg.get("trace_root") else None
    if _HAS_ITIMER:
        signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_Timeout()))
    base_cwd = os.getcwd()
    for modname in cfg["modules"]:
        try:
            mod = importlib.import_module(modname)
        except BaseException as e:
            emit({"kind": "module_error", "module": modname, "error": "%s: %s" % (type(e).__name__, e),
                  "trace": traceback.format_exc()[-1200:]})
            continue
        names = [n for n in vars(mod) if n.startswith("case_") and callable(getattr(mod, n))]
        if cfg.get("only"):
            names = [n for n in names if "%s::%s" % (modname, n) in cfg["only"]]
        emit({"kind": "module_info", "module": modname, "found": len(names), "file": getattr(mod, "__file__", None)})
        if cfg.get("reverse"):
            names.reverse()
        for n in names:
            fn = getattr(mod, n)
            key = "%s::%s" % (modname, n)
            tmp = tempfile.mkdtemp(prefix="case-")
            os.chdir(tmp)
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                import pathlib
                kwargs["tmp_path"] = pathlib.Path(tmp)
            rec = {"kind": "case", "key": key}
            t0 = time.perf_counter()
            if _HAS_ITIMER:
                signal.setitimer(signal.ITIMER_REAL, float(cfg.get("timeout", 5)))
            if tracer:
                tracer.lines = set()
                sys.settrace(tracer.g)
            try:
                val = fn(**kwargs)
                rec["status"] = "value"
            except _Timeout:
                rec["status"], rec["error"] = "error", "timed out"
            except Exception as e:
                rec.update(status="raises", exc=type(e).__name__, msg=str(e)[:300])
                try:
                    rec["cls"] = exc_path(type(e), packages)
                except Exception:
                    rec["cls"] = "Exception"
            except BaseException as e:
                rec["status"], rec["error"] = "error", "raised %s (cannot be asserted)" % type(e).__name__
            finally:
                sys.settrace(None)
                if _HAS_ITIMER:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                os.chdir(base_cwd)
            rec["secs"] = round(time.perf_counter() - t0, 4)
            if rec["status"] == "value":
                try:
                    src = to_src(val)
                    if len(src) > 8000:
                        raise Unrecordable("result too large (%d chars); return a summary" % len(src))
                    rec["src"] = src
                except Unrecordable as u:
                    rec["status"], rec["error"] = "error", str(u)
                except Exception as e:
                    rec["status"], rec["error"] = "error", "unrecordable: %s" % type(e).__name__
            if tracer:
                root = cfg["src_root"]
                rl = []
                for f, ln in tracer.lines:
                    try:
                        rl.append([os.path.relpath(f, root), ln])
                    except ValueError:          # Windows: different drive / unrelated path
                        continue
                rec["lines"] = sorted(rl)
            emit(rec)
    emit({"kind": "end"})


if __name__ == "__main__":
    main(sys.argv[1])
