"""
regression_runtime.py -- runtime half of rtgen (the regression-test generator).

It is copied next to the generated tests and is used in two ways:

  * by the generator (``collect`` / ``verify`` sub-commands) to discover the public
    API of a module, synthesise inputs, run them and record the *observable
    behaviour* (return value, raised exception, stdout/stderr, mutation of the
    arguments, public state of the instance);
  * by the generated pytest file to replay every recorded case and compare.

Design rules that keep the tests stable under behaviour-preserving refactoring:
  - only the public API is touched (no underscore names, private attributes are
    never recorded);
  - values are compared through a canonical encoding (sets sorted, memory
    addresses removed, exact types and exact float reprs kept);
  - cases whose outcome differs between two independent runs (different hash
    seed, reversed order) are discarded as non-deterministic / order dependent.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import collections.abc
import contextlib
import dataclasses
import doctest
import enum
import functools
import importlib
import inspect
import io
import json
import os
import random
import re
import shutil
import signal
import sys
import tempfile
import threading
import types
import typing

P = inspect.Parameter

STEP_TIMEOUT = 5.0
MAX_DEPTH = 8
MAX_ITEMS = 200
MAX_TEXT = 20000
OMIT = object()
_ROOT = ""

DANGEROUS = re.compile(
    r"(^|_)(exit|quit|system|popen|spawn|serve|server|forever|shutdown|daemon|download|"
    r"upload|sleep|listen|main|cli|install|uninstall|kill|login)(_|$)",
    re.I,
)
DUNDERS = {
    "__len__", "__str__", "__repr__", "__bool__", "__iter__", "__contains__", "__getitem__",
    "__setitem__", "__delitem__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__add__", "__sub__", "__mul__", "__truediv__", "__neg__", "__abs__", "__hash__",
    "__call__", "__enter__", "__exit__", "__next__", "__radd__", "__rmul__",
}


def set_root(root: str) -> None:
    global _ROOT
    _ROOT = os.path.abspath(root)
    if _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)


# --------------------------------------------------------------------------- #
# Encoding of *inputs* (JSON friendly, exactly reversible)
# --------------------------------------------------------------------------- #
def enc(v):
    if v is None:
        return ["none"]
    if isinstance(v, enum.Enum):
        return ["enum", type(v).__module__, type(v).__qualname__, v.name]
    t = type(v)
    if t is bool:
        return ["bool", v]
    if t is int:
        return ["int", str(v)]
    if t is float:
        return ["float", repr(v)]
    if t is complex:
        return ["complex", repr(v)]
    if t is str:
        return ["str", v]
    if t is bytes:
        return ["bytes", v.hex()]
    if t is list:
        return ["list", [enc(x) for x in v]]
    if t is tuple:
        return ["tuple", [enc(x) for x in v]]
    if t is set:
        return ["set", [enc(x) for x in v]]
    if t is frozenset:
        return ["frozenset", [enc(x) for x in v]]
    if t is dict:
        return ["dict", [[enc(k), enc(x)] for k, x in v.items()]]
    raise TypeError(f"cannot encode {t!r}")


def dec(e):
    tag = e[0]
    if tag == "none":
        return None
    if tag == "enum":
        obj = importlib.import_module(e[1])
        for part in e[2].split("."):
            obj = getattr(obj, part)
        return obj[e[3]]
    if tag == "bool":
        return bool(e[1])
    if tag == "int":
        return int(e[1])
    if tag == "float":
        return float(e[1])
    if tag == "complex":
        return complex(e[1])
    if tag == "str":
        return e[1]
    if tag == "bytes":
        return bytes.fromhex(e[1])
    if tag == "list":
        return [dec(x) for x in e[1]]
    if tag == "tuple":
        return tuple(dec(x) for x in e[1])
    if tag == "set":
        return {dec(x) for x in e[1]}
    if tag == "frozenset":
        return frozenset(dec(x) for x in e[1])
    if tag == "dict":
        return {dec(k): dec(x) for k, x in e[1]}
    raise ValueError(tag)


def dec_call(call):
    return [dec(a) for a in call["args"]], {k: dec(v) for k, v in call["kwargs"].items()}


# --------------------------------------------------------------------------- #
# Canonical form of *outcomes*
# --------------------------------------------------------------------------- #
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _qn(t):
    return getattr(t, "__qualname__", getattr(t, "__name__", str(t)))


def _jk(x):
    return json.dumps(x, sort_keys=True)


def _public_attrs(v):
    d = {}
    if hasattr(v, "__dict__"):
        try:
            d.update(vars(v))
        except TypeError:
            pass
    for klass in type(v).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for s in slots:
            if s in ("__dict__", "__weakref__"):
                continue
            try:
                d[s] = getattr(v, s)
            except AttributeError:
                pass
    return {k: d[k] for k in sorted(d) if isinstance(k, str) and not k.startswith("_")}


def canon(v, depth=0, seen=frozenset()):
    t = type(v)
    if v is None:
        return ["none"]
    if isinstance(v, enum.Enum):
        return ["enum", _qn(t), v.name]
    if t is bool:
        return ["bool", v]
    if isinstance(v, int):
        return ["int" if t is int else "int:" + _qn(t), str(int(v))]
    if isinstance(v, float):
        return ["float" if t is float else "float:" + _qn(t), repr(float(v))]
    if isinstance(v, complex):
        return ["complex", repr(complex(v))]
    if isinstance(v, str):
        return ["str" if t is str else "str:" + _qn(t), str.__str__(v)]
    if isinstance(v, (bytes, bytearray)):
        return [_qn(t), bytes(v).hex()]
    if depth > MAX_DEPTH:
        return ["deep", _qn(t)]
    if id(v) in seen:
        return ["cycle"]
    seen = seen | {id(v)}
    d = depth + 1
    if isinstance(v, (list, tuple)):
        items = [canon(x, d, seen) for x in list(v)[:MAX_ITEMS]]
        tag = "list" if t is list else "tuple" if t is tuple else _qn(t)
        return [tag, items] + ([len(v)] if len(v) > MAX_ITEMS else [])
    if isinstance(v, dict):
        items = [[canon(k, d, seen), canon(x, d, seen)] for k, x in list(v.items())[:MAX_ITEMS]]
        return ["dict" if t is dict else _qn(t), items]
    if isinstance(v, (set, frozenset)):
        items = sorted((canon(x, d, seen) for x in v), key=_jk)[:MAX_ITEMS]
        return ["set" if t is set else "frozenset" if t is frozenset else _qn(t), items]
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        flds = {f.name: getattr(v, f.name, None) for f in dataclasses.fields(v)}
        return ["dataclass", _qn(t), canon(flds, d, seen)]
    if inspect.isgenerator(v) or (
        isinstance(v, collections.abc.Iterator) and t.__module__ in ("builtins", "itertools")
    ):
        items = []
        for x in v:
            items.append(canon(x, d, seen))
            if len(items) >= MAX_ITEMS:
                break
        return ["iter", items]
    if isinstance(v, type):
        return ["type", _qn(v)]
    if isinstance(v, BaseException):
        return ["exc", _qn(t), str(v)]
    if inspect.ismodule(v):
        return ["module", v.__name__]
    if inspect.isroutine(v) or isinstance(v, functools.partial):
        return ["callable", getattr(v, "__qualname__", _qn(t))]
    r = None
    if t.__repr__ is not object.__repr__:
        try:
            r = _ADDR.sub("", repr(v))[:MAX_TEXT]
        except Exception:
            r = "<unreprable>"
    attrs = _public_attrs(v)
    if attrs or r is None:
        return ["obj", _qn(t), canon(attrs, d, seen)] + ([r] if r is not None else [])
    return ["repr", _qn(t), r]


# --------------------------------------------------------------------------- #
# Input pools
# --------------------------------------------------------------------------- #
INTS = [3, 0, 1, -1, 2, 10, -7, 255]
FLOATS = [2.5, 0.0, 1.0, -1.5, 0.1, 100.25]
STRS = ["hello", "", "a", "Hello World", " padded ", "a,b,c", "ÄÖü", "123", "line1\nline2"]
BOOLS = [True, False]
BYTES = [b"abc", b""]
LISTS_INT = [[3, 1, 2], [], [1], [1, 1, 2, 2, 3], [-1, 0, 5]]
LISTS_STR = [["b", "a", "c"], [], ["a"], ["x", "x", "y"]]
LISTS = LISTS_INT + LISTS_STR[1:3] + [STRS[0]]
DICTS = [{"a": 1, "b": 2}, {}, {"a": 1}, {"b": 2, "a": 3}]
GENERIC = [3, 0, 1, -1, "abc", "", [1, 2, 3], [], None, True, 2.5, {"a": 1}]

_NAME_RULES = [
    (re.compile(r"^(is_|has_|use_|allow_|enable|flag$|strict$|reverse$|verbose$|inplace$|"
                r"ignore_case$|case_sensitive$|descending$|ascending$|recursive$|force$|dry_run$)"), BOOLS),
    (re.compile(r"(^|_)(name|text|string|str|word|sentence|line|key|prefix|suffix|sep|separator|"
                r"delimiter|label|title|msg|message|pattern|path|filename|file|s|char|ch|token|"
                r"query|code|id|url|email|phrase)$"), STRS),
    (re.compile(r"(^|_)(n|i|j|k|x|y|z|num|number|count|idx|index|size|length|len|width|height|"
                r"limit|depth|start|stop|step|offset|total|amount|age|year|month|day|value|val|v|"
                r"a|b|c|max|min|lo|hi|low|high|capacity|base|exp|power|threshold|rate|price|qty|"
                r"quantity|weight|score|level|radius|degree)$"), INTS + FLOATS[:3]),
    (re.compile(r"(^|_)(items|lst|list|values|xs|ys|seq|sequence|data|arr|array|numbers|nums|"
                r"elements|iterable|collection|args|lines|words|rows|tokens|names|keys|stack|"
                r"queue|vals|points|records|entries)$"), LISTS),
    (re.compile(r"(^|_)(d|dct|dict|mapping|config|options|kwargs|params|record|table|counts|"
                r"env|settings|headers|obj)$"), DICTS),
]
_STR_ANN = {"int": int, "str": str, "float": float, "bool": bool, "bytes": bytes,
            "list": list, "dict": dict, "tuple": tuple, "set": set, "None": None}


EXTRA_INTS: list = []
EXTRA_STRS: list = []


def harvest_literals(mod):
    """Boundary candidates from the module source: compared constants (+-1), small ints, short strings."""
    global EXTRA_INTS, EXTRA_STRS
    EXTRA_INTS, EXTRA_STRS = [], []
    try:
        tree = ast.parse(inspect.getsource(mod))
    except (OSError, TypeError, SyntaxError):
        return
    cmp_ints, other_ints, strs = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for sub in [node.left] + node.comparators:
                if isinstance(sub, ast.Constant):
                    v = sub.value
                    if type(v) is int and abs(v) < 10**6:
                        cmp_ints += [v, v - 1, v + 1]
                    elif type(v) is str and len(v) <= 20:
                        strs.append(v)
        elif isinstance(node, ast.Constant):
            v = node.value
            if type(v) is int and abs(v) < 10**4:
                other_ints += [v, v + 1]
            elif type(v) is str and 0 < len(v) <= 12 and v.isprintable():
                strs.append(v)

    def uniq(seq, n):
        out = []
        for x in seq:
            if x not in out:
                out.append(x)
        return out[:n]

    EXTRA_INTS = uniq(cmp_ints + other_ints, 14)
    EXTRA_STRS = uniq(strs, 8)


def _with_extras(pool):
    vals = [x for x in pool if x is not None]
    if vals and all(type(x) is int for x in vals):
        return pool + EXTRA_INTS
    if vals and all(type(x) is str for x in vals):
        return pool + EXTRA_STRS
    return pool


def _hashable(x):
    return isinstance(x, (int, str, float, bytes, bool, tuple, type(None)))


def _clean(pool):
    seen, out = set(), []
    for v in pool:
        try:
            k = json.dumps(enc(v))
        except TypeError:
            continue
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _pool_from_ann(ann, depth=0):
    if ann is P.empty or depth > 4:
        return None
    if isinstance(ann, str):
        if ann.strip() not in _STR_ANN:
            return None
        ann = _STR_ANN[ann.strip()]
    if ann is None or ann is type(None):
        return [None]
    if ann is bool:
        return list(BOOLS)
    if ann is int:
        return list(INTS)
    if ann is float:
        return list(FLOATS) + [1, 0]
    if ann is str:
        return list(STRS)
    if ann is bytes:
        return list(BYTES)
    if ann is typing.Any or ann is object:
        return list(GENERIC)
    if ann in (list, set, frozenset, tuple, dict):
        return {list: LISTS_INT, set: [{3, 1, 2}, set()], frozenset: [frozenset({1, 2})],
                tuple: [(3, 1, 2), ()], dict: DICTS}[ann]
    if inspect.isclass(ann) and issubclass(ann, enum.Enum):
        return list(ann)
    origin, args = typing.get_origin(ann), typing.get_args(ann)
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        pool = []
        for a in args:
            pool += _pool_from_ann(a, depth + 1) or []
        return pool or None
    if origin is typing.Literal:
        return list(args)
    seq_origins = (list, collections.abc.Sequence, collections.abc.Iterable,
                   collections.abc.MutableSequence, collections.abc.Collection)
    if origin in seq_origins:
        el = (_pool_from_ann(args[0], depth + 1) if args else None) or INTS
        if args and args[0] is int:
            return [list(x) for x in LISTS_INT]
        if args and args[0] is str:
            return [list(x) for x in LISTS_STR]
        return [el[:3], [], el[:1], el[:2] + el[:1]]
    if origin in (set, frozenset, collections.abc.Set, collections.abc.MutableSet):
        el = [x for x in ((_pool_from_ann(args[0], depth + 1) if args else None) or INTS) if _hashable(x)]
        mk = frozenset if origin is frozenset else set
        return [mk(el[:3]), mk(), mk(el[:1])]
    if origin is tuple:
        if args and args[-1] is not Ellipsis:
            pools = [_pool_from_ann(a, depth + 1) or GENERIC for a in args]
            return [tuple(p[0] for p in pools), tuple(p[min(1, len(p) - 1)] for p in pools)]
        el = (_pool_from_ann(args[0], depth + 1) if args else None) or INTS
        return [tuple(el[:3]), ()]
    if origin in (dict, collections.abc.Mapping, collections.abc.MutableMapping):
        ks = [k for k in ((_pool_from_ann(args[0], depth + 1) if args else None) or STRS) if _hashable(k)]
        vs = (_pool_from_ann(args[1], depth + 1) if len(args) > 1 else None) or INTS
        if len(ks) < 2:
            ks = ks + ["k2"]
        return [{ks[0]: vs[0], ks[1]: vs[min(1, len(vs) - 1)]}, {}, {ks[0]: vs[0]}]
    return None


def _pool_like(d):
    if isinstance(d, bool):
        return list(BOOLS)
    if isinstance(d, int):
        return list(INTS)
    if isinstance(d, float):
        return list(FLOATS)
    if isinstance(d, str):
        return list(STRS)
    if isinstance(d, bytes):
        return list(BYTES)
    if isinstance(d, list):
        return list(LISTS)
    if isinstance(d, tuple):
        return [(3, 1, 2), ()]
    if isinstance(d, dict):
        return list(DICTS)
    if isinstance(d, (set, frozenset)):
        return [{3, 1, 2}, set()]
    return None


def pool_for(p, ann):
    pool = _pool_from_ann(ann)
    if pool is not None and p.default is None:
        pool = pool + [None]
    if pool is None and p.default is not P.empty:
        if p.default is None:
            pool = [None, 3, "hello", [3, 1, 2]]
        else:
            pool = _pool_like(p.default)
            if pool is not None:
                pool = pool + [p.default]
    if pool is None:
        for rx, pl in _NAME_RULES:
            if rx.search(p.name):
                pool = list(pl)
                break
    if pool is None:
        pool = list(GENERIC)
    return _clean(_with_extras(pool)) or [None]


def _hints(fn):
    try:
        return typing.get_type_hints(inspect.unwrap(fn))
    except Exception:
        return {}


def make_specs(params, hints):
    specs = []
    for p in params:
        if p.kind is P.VAR_POSITIONAL:
            pool, has_def = [(), (3, 1), ("a",)], True
        elif p.kind is P.VAR_KEYWORD:
            pool, has_def = [{}, {"extra": 1}], True
        else:
            pool = pool_for(p, hints.get(p.name, p.annotation))
            has_def = p.default is not P.empty
        specs.append({"name": p.name, "kind": p.kind, "has_default": has_def, "pool": pool})
    return specs


def build_call(specs, values, kwstyle=False):
    args, kwargs, positional_ok = [], {}, True
    for s, v in zip(specs, values):
        k = s["kind"]
        if v is OMIT:
            if k in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD):
                positional_ok = False
            continue
        if k is P.VAR_POSITIONAL:
            if not positional_ok:
                return None
            args.extend(v)
        elif k is P.VAR_KEYWORD:
            kwargs.update(v)
        elif k is P.KEYWORD_ONLY:
            kwargs[s["name"]] = v
        elif k is P.POSITIONAL_ONLY:
            if not positional_ok:
                return None
            args.append(v)
        else:
            if positional_ok and not kwstyle:
                args.append(v)
            else:
                kwargs[s["name"]] = v
                positional_ok = False if kwstyle else positional_ok
    try:
        return {"args": [enc(a) for a in args], "kwargs": {n: enc(x) for n, x in kwargs.items()}}
    except TypeError:
        return None


def gen_calls(specs, rng, max_cases):
    pools = [([OMIT] if s["has_default"] else []) + list(s["pool"]) for s in specs]
    base = [p[0] for p in pools]
    cands = [(base, False)]
    n_rand = max(2, max_cases // 4) if specs else 0
    oat = []
    for i, p in enumerate(pools):
        for v in p[1:]:
            c = list(base)
            c[i] = v
            oat.append((c, False))
    limit = max(1, max_cases - n_rand - 1)
    if len(oat) > limit:
        oat = rng.sample(oat, limit)
    cands += oat
    for _ in range(n_rand):
        cands.append(([rng.choice(p) for p in pools], rng.random() < 0.25))
    seen, out = set(), []
    for values, kw in cands:
        c = build_call(specs, values, kw)
        if c is None:
            continue
        k = _jk(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def doc_calls(fn, name):
    out = []
    try:
        examples = doctest.DocTestParser().get_examples(inspect.getdoc(fn) or "")
    except Exception:
        return out
    for ex in examples:
        try:
            node = ast.parse(ex.source.strip(), mode="eval").body
        except SyntaxError:
            continue
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name):
            continue
        if any(k.arg is None for k in node.keywords):
            continue
        try:
            args = [ast.literal_eval(a) for a in node.args]
            kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            out.append({"args": [enc(a) for a in args], "kwargs": {k: enc(v) for k, v in kwargs.items()}})
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Public API discovery
# --------------------------------------------------------------------------- #
def public_names(mod):
    allv = getattr(mod, "__all__", None)
    if allv is not None:
        return [n for n in allv if isinstance(n, str) and hasattr(mod, n)]
    out = []
    for n in sorted(vars(mod)):
        if n.startswith("_"):
            continue
        v = getattr(mod, n)
        if inspect.ismodule(v):
            continue
        if (inspect.isclass(v) or callable(v)) and getattr(v, "__module__", None) != mod.__name__:
            continue
        out.append(n)
    return out


def _immutable_const(v):
    if v is None or isinstance(v, (int, float, str, bytes, complex)):
        return True
    if isinstance(v, (tuple, frozenset)):
        return all(_immutable_const(x) for x in v)
    return False


def _in_repo(klass):
    m = sys.modules.get(klass.__module__)
    f = getattr(m, "__file__", None)
    return bool(f) and os.path.abspath(f).startswith(_ROOT)


def class_members(cls):
    methods, props, seen = [], [], set()
    for klass in cls.__mro__:
        if klass is object or not _in_repo(klass):
            continue
        for n, raw in vars(klass).items():
            if n in seen:
                continue
            seen.add(n)
            if (n.startswith("_") and n not in DUNDERS) or DANGEROUS.search(n):
                continue
            if isinstance(raw, property):
                if raw.fget is not None:
                    props.append(n)
            elif isinstance(raw, (staticmethod, classmethod)) or inspect.isfunction(raw):
                methods.append(n)
    return sorted(methods), sorted(props)


def sig_of(fn):
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    out = []
    for p in sig.parameters.values():
        out.append([p.name, p.kind.name, ["empty"] if p.default is P.empty else canon(p.default)])
    return out


def describe_api(mod):
    d = {"functions": {}, "classes": {}, "consts": {}}
    for name in public_names(mod):
        obj = getattr(mod, name)
        if inspect.isclass(obj):
            info = {}
            if issubclass(obj, enum.Enum):
                info["members"] = [[m.name, canon(m.value)] for m in obj]
            else:
                info["init"] = sig_of(obj)
                methods, props = class_members(obj)
                info["methods"] = {m: sig_of(getattr(obj, m)) for m in methods}
                info["properties"] = props
            d["classes"][name] = info
        elif callable(obj):
            d["functions"][name] = sig_of(obj)
        elif _immutable_const(obj):
            d["consts"][name] = canon(obj)
    return json.loads(json.dumps(d))


def _subset(exp, act, path, probs):
    if isinstance(exp, dict):
        if not isinstance(act, dict):
            probs.append(f"{path}: expected a mapping")
            return
        for k, v in exp.items():
            if k not in act:
                probs.append(f"{path}.{k}: missing from the public API")
            else:
                _subset(v, act[k], f"{path}.{k}", probs)
    elif exp != act:
        probs.append(f"{path}: expected {json.dumps(exp)[:300]} but got {json.dumps(act)[:300]}")


def check_api(expected, actual):
    probs = []
    _subset(expected, json.loads(json.dumps(actual)), "api", probs)
    return probs


# --------------------------------------------------------------------------- #
# Safe execution of one case
# --------------------------------------------------------------------------- #
class StepTimeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise StepTimeout()


def _can_alarm():
    return hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()


@contextlib.contextmanager
def _guard():
    import socket
    import subprocess

    saved = []

    def block(obj, name):
        if not hasattr(obj, name):
            return
        old = getattr(obj, name)

        def denied(*a, **k):
            raise PermissionError(f"{name} is disabled by the regression harness")

        try:
            setattr(obj, name, denied)
            saved.append((obj, name, old))
        except (TypeError, AttributeError):
            pass

    for n in ("remove", "unlink", "rmdir", "removedirs", "system", "kill", "killpg", "_exit",
              "execv", "execvp", "fork"):
        block(os, n)
    block(shutil, "rmtree")
    block(subprocess, "Popen")
    block(socket.socket, "connect")
    block(socket, "create_connection")
    old_in = sys.stdin
    sys.stdin = io.StringIO("")
    try:
        yield
    finally:
        for obj, name, old in reversed(saved):
            setattr(obj, name, old)
        sys.stdin = old_in


async def _await(a):
    return await a


def _invoke(fn, a, k, ignore):
    out, err = io.StringIO(), io.StringIO()
    rec = {}
    if _can_alarm():
        signal.setitimer(signal.ITIMER_REAL, STEP_TIMEOUT)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                val = fn()
                if inspect.isawaitable(val):
                    val = asyncio.run(_await(val))
                rec["ret"] = canon(val)
                rec["exc"] = None
            except StepTimeout:
                raise
            except BaseException as e:  # noqa: BLE001 - exceptions are observable behaviour
                val = None
                rec["ret"] = ["none"]
                rec["exc"] = [_qn(type(e)), "" if ignore else str(e)[:2000]]
    finally:
        if _can_alarm():
            signal.setitimer(signal.ITIMER_REAL, 0)
    rec["out"] = out.getvalue()[:MAX_TEXT]
    rec["err"] = err.getvalue()[:MAX_TEXT]
    try:
        rec["args_after"] = canon([list(a), k])
    except Exception:
        rec["args_after"] = ["uncanon"]
    return rec, val


def _run(case, ignore):
    mod = importlib.import_module(case["module"])
    cls_name = case.get("cls")
    inst, res = None, []
    if cls_name:
        cls = getattr(mod, cls_name)
        a, k = dec_call(case["ctor"])
        rec, val = _invoke(lambda: cls(*a, **k), a, k, ignore)
        rec["step"] = "__init__"
        rec["ret"] = ["none"]
        if rec["exc"] is None:
            inst = val
            rec["state"] = canon(inst)
        res.append(rec)
        if inst is None:
            return res
    target = inst if cls_name else mod
    for st in case["steps"]:
        a, k = dec_call(st)
        name = st["name"]
        if st.get("mode") == "get":
            fn = lambda: getattr(target, name)  # noqa: E731
        else:
            fn = lambda: getattr(target, name)(*a, **k)  # noqa: E731
        rec, _ = _invoke(fn, a, k, ignore)
        rec["step"] = name
        if inst is not None:
            rec["state"] = canon(inst)
        res.append(rec)
    return res


def run_case(case, ignore_messages=False):
    """Execute a case; returns the list of per-step outcomes or None on timeout."""
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="rt_")
    old_handler = None
    if _can_alarm():
        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    try:
        os.chdir(tmp)
        random.seed(1234)
        with _guard():
            res = _run(case, ignore_messages)
        return json.loads(json.dumps(res))
    except StepTimeout:
        return None
    finally:
        os.chdir(cwd)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
        shutil.rmtree(tmp, ignore_errors=True)


def explain_diff(case, actual):
    exp = case["expected"]
    if actual is None:
        return f"{case['id']}: timed out"
    for i, (e, a) in enumerate(zip(exp, actual)):
        for key in e:
            if e[key] != a.get(key):
                si = i - (1 if case.get("cls") else 0)
                call = case["steps"][si] if 0 <= si < len(case["steps"]) else case.get("ctor")
                return (f"{case['id']}\n  step {i} ({e.get('step')}) differs in '{key}'\n"
                        f"  call:     {json.dumps(call)[:400]}\n"
                        f"  expected: {json.dumps(e[key])[:600]}\n  actual:   {json.dumps(a.get(key))[:600]}")
    return f"{case['id']}: expected {len(exp)} step results, got {len(actual)}"


# --------------------------------------------------------------------------- #
# Case construction
# --------------------------------------------------------------------------- #
def mkcase(module, cls, ctor, steps, label):
    return {"label": label, "module": module, "cls": cls, "ctor": ctor, "steps": steps}


def function_cases(mod, name, fn, rng, max_cases):
    if inspect.isasyncgenfunction(fn):
        return []
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return []
    specs = make_specs(list(sig.parameters.values()), _hints(fn))
    calls = doc_calls(fn, name) + gen_calls(specs, rng, max_cases)
    seen, out = set(), []
    for c in calls:
        k = _jk(c)
        if k not in seen:
            seen.add(k)
            out.append(mkcase(mod.__name__, None, None, [dict(name=name, mode="call", **c)], name))
    return out


def _method_specs(cls, name):
    raw = inspect.getattr_static(cls, name)
    fn = getattr(cls, name)
    params = list(inspect.signature(fn).parameters.values())
    if (not isinstance(raw, (staticmethod, classmethod)) and params
            and params[0].kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)):
        params = params[1:]
    return make_specs(params, _hints(fn))


def class_cases(mod, name, cls, rng, max_cases, ignore):
    if inspect.isabstract(cls) or issubclass(cls, (BaseException, enum.Enum)):
        return []
    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return []
    specs = make_specs(list(sig.parameters.values()), _hints(cls.__init__))
    cases, working = [], []
    for call in gen_calls(specs, rng, 8):
        case = mkcase(mod.__name__, name, call, [], f"{name}.__init__")
        res = run_case(case, ignore)
        if res is None:
            continue
        cases.append(case)
        if res[0]["exc"] is None:
            working.append(call)
    if not working:
        return cases
    methods, props = class_members(cls)
    steps_pool = []
    per_method = max(6, max_cases // 2)
    for m in methods:
        try:
            mspecs = _method_specs(cls, m)
        except (ValueError, TypeError):
            continue
        calls = gen_calls(mspecs, rng, per_method)
        steps_pool.append((m, "call", calls))
        for i, c in enumerate(calls):
            ctor = working[i % len(working)]
            cases.append(mkcase(mod.__name__, name, ctor, [dict(name=m, mode="call", **c)], f"{name}.{m}"))
    for p in props:
        steps_pool.append((p, "get", [{"args": [], "kwargs": {}}]))
        cases.append(mkcase(mod.__name__, name, working[0],
                            [dict(name=p, mode="get", args=[], kwargs={})], f"{name}.{p}"))
    if steps_pool:
        for _ in range(max(12, max_cases)):
            ctor = rng.choice(working)
            steps = []
            for _ in range(rng.randint(3, 7)):
                m, mode, calls = rng.choice(steps_pool)
                # bias towards the baseline-ish calls so consecutive steps share arguments
                r = rng.random()
                pick = calls[:1] if r < 0.5 else calls[:4] if r < 0.8 else calls
                steps.append(dict(name=m, mode=mode, **rng.choice(pick)))
            cases.append(mkcase(mod.__name__, name, ctor, steps, f"{name}.sequence"))
    return cases


# --------------------------------------------------------------------------- #
# CLI used by the generator
# --------------------------------------------------------------------------- #
def collect(root, module_name, seed, max_cases, ignore):
    set_root(root)
    mod = importlib.import_module(module_name)
    harvest_literals(mod)
    rng = random.Random(f"{seed}:{module_name}")
    api = describe_api(mod)
    cases = []
    for name in public_names(mod):
        obj = getattr(mod, name)
        if DANGEROUS.search(name):
            continue
        try:
            if inspect.isclass(obj):
                cases += class_cases(mod, name, obj, rng, max_cases, ignore)
            elif callable(obj):
                cases += function_cases(mod, name, obj, rng, max_cases)
        except Exception as e:  # discovery problems must never abort the module
            print(f"[rtgen] skipped {module_name}.{name}: {e!r}", file=sys.stderr)
    counters, final = {}, []
    for c in cases:
        res = run_case(c, ignore)
        if res is None:
            continue
        n = counters.get(c["label"], 0)
        counters[c["label"]] = n + 1
        c["id"] = f"{module_name}::{c['label']}#{n}"
        c["expected"] = res
        final.append(c)
    return {"module": module_name, "api": api, "cases": final}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--root", required=True)
    c.add_argument("--module", required=True)
    c.add_argument("--seed", default="0")
    c.add_argument("--max-cases", type=int, default=30)
    c.add_argument("--ignore-messages", action="store_true")
    c.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--root", required=True)
    v.add_argument("--cases", required=True)
    v.add_argument("--ignore-messages", action="store_true")
    v.add_argument("--out", required=True)
    ns = ap.parse_args(argv)
    if ns.cmd == "collect":
        data = collect(ns.root, ns.module, ns.seed, ns.max_cases, ns.ignore_messages)
        with open(ns.out, "w", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        set_root(ns.root)
        with open(ns.cases, encoding="utf-8") as f:
            data = json.load(f)
        kept = []
        for case in reversed(data["cases"]):  # different order + different hash seed
            if run_case(case, ns.ignore_messages) == case["expected"]:
                kept.append(case["id"])
        with open(ns.out, "w", encoding="utf-8") as f:
            json.dump({"kept": kept}, f)


if __name__ == "__main__":
    main()
