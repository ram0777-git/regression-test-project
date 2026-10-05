"""Deterministic cases (via regression_runtime), LLM reply parsing, and case-file validation (lint)."""
from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.join(HERE, "regression_runtime.py")
sys.path.insert(0, HERE)
import regression_runtime as rt  # noqa: E402

HELPERS = '''
def _plain(v, depth=0):
    if v is None or isinstance(v, (bool, int, float, str, bytes, complex)):
        return v
    if depth > 6:
        return "<deep>"
    if isinstance(v, (list, tuple)):
        return type(v)(_plain(x, depth + 1) for x in v) if type(v) in (list, tuple) else [_plain(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {_plain(k, depth + 1): _plain(x, depth + 1) for k, x in v.items()} if type(v) is dict else "<mapping>"
    if isinstance(v, (set, frozenset)):
        return v if all(isinstance(x, (int, str, float, bool, bytes, tuple, type(None))) for x in v) else sorted(map(str, v))
    return "<%s>" % type(v).__name__


def _obs(fn):
    try:
        return ["ok", _plain(fn())]
    except Exception as exc:
        return ["raises", type(exc).__name__]


def _state(obj):
    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        return None
    return {k: _plain(v) for k, v in sorted(d.items()) if isinstance(k, str) and not k.startswith("_")}
'''

ALLOWED_EXTRA = {"pytest"}
FORBIDDEN_MODULES = {"socket", "subprocess", "urllib", "http", "requests", "ftplib", "smtplib", "asyncio.subprocess",
                     "multiprocessing", "ctypes", "threading"}
FORBIDDEN_CALLS = {"id", "hash", "repr", "eval", "exec", "compile", "globals", "locals", "vars", "input",
                   "breakpoint", "__import__", "open"}
CLOCK = {("time", "time"), ("time", "monotonic"), ("time", "perf_counter"), ("time", "sleep"),
         ("datetime", "now"), ("datetime", "utcnow"), ("date", "today"), ("os", "getenv"), ("os", "getcwd"),
         ("os", "getpid"), ("os", "system"), ("os", "remove"), ("shutil", "rmtree"), ("uuid", "uuid4"),
         ("random", "random"), ("random", "randint"), ("random", "choice"), ("random", "shuffle")}
PRIVATE_DUNDER = {"__dict__", "__code__", "__closure__", "__globals__", "__slots__", "__wrapped__"}


def literal(v):
    r = repr(v)
    try:
        back = ast.literal_eval(r)
    except Exception:
        return None
    if back != v or type(back) is not type(v):
        return None
    if any(isinstance(x, float) and not math.isfinite(x) for x in _flat(v)):
        return None
    return r


def _flat(v):
    if isinstance(v, (list, tuple, set, frozenset)):
        for x in v:
            yield from _flat(x)
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from _flat(k)
            yield from _flat(x)
    else:
        yield v


def collect_runtime(py, src_root, module, seed, max_cases, work, timeout=600):
    out = os.path.join(work, "rt_%s_%s.json" % (module.replace(".", "_"), seed))
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    env.pop("OPENROUTER_API_KEY", None)
    try:
        p = subprocess.run([py, RUNTIME, "collect", "--root", src_root, "--module", module, "--seed", str(seed),
                            "--max-cases", str(max_cases), "--ignore-messages", "--out", out],
                           cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if p.returncode != 0 or not os.path.exists(out):
        return []
    with open(out) as fh:
        return json.load(fh).get("cases", [])


def _call_src(target, call, pre, counter, mutable_names):
    args, kwargs = rt.dec_call(call)
    parts = []
    for a in args:
        r = literal(a)
        if r is None:
            return None
        if isinstance(a, (list, dict, set)) and a:
            nm = "a%d" % counter[0]
            counter[0] += 1
            pre.append("    %s = %s" % (nm, r))
            mutable_names.append(nm)
            parts.append(nm)
        else:
            parts.append(r)
    for k, v in kwargs.items():
        r = literal(v)
        if r is None or not k.isidentifier():
            return None
        parts.append("%s=%s" % (k, r))
    return "%s(%s)" % (target, ", ".join(parts))


def runtime_case_file(module, cases, names, per_label, contract, tag):
    """Translate regression_runtime cases into a case file (only in-scope labels)."""
    used, funcs, counts = set(), [], {}
    for c in cases:
        label = c["label"]
        top = label.split(".")[0]
        if names and top not in names and label.split(".")[-1] not in names:
            continue
        if counts.get(label, 0) >= per_label:
            continue
        pre, muts, counter = [], [], [0]
        if c.get("cls"):
            ctor = _call_src(c["cls"], c["ctor"], pre, counter, muts)
            if ctor is None:
                continue
            obs = []
            ok = True
            for st in c["steps"]:
                if st.get("mode") == "get":
                    obs.append("_obs(lambda: obj.%s)" % st["name"])
                else:
                    s = _call_src("obj." + st["name"], st, pre, counter, muts)
                    if s is None:
                        ok = False
                        break
                    obs.append("_obs(lambda: %s)" % s)
            if not ok:
                continue
            if contract.state:
                obs.append("_state(obj)")
            body = pre + ["    obj = %s" % ctor, "    return [%s]" % ", ".join(obs or ["_state(obj)"])] \
                if c["steps"] or contract.state else pre + ["    return _plain(%s)" % ctor]
            used.add(c["cls"])
        else:
            fn = c["steps"][0]["name"]
            s = _call_src(fn, c["steps"][0], pre, counter, muts)
            if s is None:
                continue
            ret = "[_plain(%s), %s]" % (s, ", ".join(muts)) if muts else "_plain(%s)" % s
            body = pre + ["    return %s" % ret]
            used.add(fn)
        counts[label] = counts.get(label, 0) + 1
        nm = "case_auto_%s_%d" % (re.sub(r"\W", "_", label), counts[label])
        funcs.append("def %s():\n%s\n" % (nm, "\n".join(body)))
    if not funcs:
        return None
    head = "from %s import %s\n" % (module, ", ".join(sorted(used)))
    return '"""Deterministic cases (%s) for %s."""\n%s%s\n\n%s' % (tag, module, head, HELPERS, "\n\n".join(funcs))


BLOCK = re.compile(r"```([^\n`]*)\n(.*?)\n?```", re.S)


def parse_reply(text):
    files, shells = [], []
    for info, body in BLOCK.findall(text or ""):
        info = info.strip()
        m = re.search(r"(cases_\w+\.py)", info)
        if m:
            files.append((m.group(1), body))
        elif info.split()[:1] in (["bash"], ["sh"], ["shell"]):
            shells.append(body)
    skips = set()
    for line in re.findall(r"(?im)^\W*SKIP\W*:?(.*)$", text or ""):
        skips |= set(re.findall(r"\bm\d+\b", line))
    done = bool(re.search(r"(?m)^\W*DONE\W*$", text or ""))
    return files, shells, skips, done


def _returns(fn):
    out, stack = [], list(fn.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(n, ast.Return):
            out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def lint(source, packages, stdlib, contract, lib_names=None):
    """Returns (module_problems, case_problems, case_names)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ["syntax error: %s (line %s)" % (e.msg, e.lineno)], {}, []
    mprob, cprob, cases = [], {}, []
    allowed = set(stdlib) | ALLOWED_EXTRA | set(packages)
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and node.level:
                mprob.append("line %d: relative import" % node.lineno)
            for m in mods:
                root = m.split(".")[0]
                if m == "__future__":
                    continue
                if root not in allowed or m in FORBIDDEN_MODULES or root in FORBIDDEN_MODULES:
                    mprob.append("line %d: import of `%s` is not allowed" % (node.lineno, m))
                if any(p.startswith("_") for p in m.split(".")):
                    mprob.append("line %d: private module `%s`" % (node.lineno, m))
                if root in packages and contract.allowed_imports and not any(
                        m == a or m.startswith(a + ".") or a.startswith(m + ".") for a in contract.allowed_imports):
                    mprob.append("line %d: the task allows imports only from %s" % (
                        node.lineno, ", ".join(contract.allowed_imports)))
            for a in node.names:
                if isinstance(node, ast.ImportFrom) and a.name.startswith("_"):
                    mprob.append("line %d: private name `%s`" % (node.lineno, a.name))
                if mods and mods[0].split(".")[0] in packages:
                    aliases.add((a.asname or a.name).split(".")[0])
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.decorator_list and any(isinstance(n, ast.Name) and n.id in aliases
                                           for d in node.decorator_list for n in ast.walk(d)):
                mprob.append("line %d: decorator runs library code at import time" % node.lineno)
            if node.name.startswith("test"):
                mprob.append("line %d: helper names may not start with `test`" % node.lineno)
            if node.name.startswith("case_"):
                cases.append(node.name)
                a = node.args
                params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
                rets = _returns(node)
                if isinstance(node, ast.AsyncFunctionDef):
                    cprob[node.name] = "case must not be async"
                elif params not in ([], ["tmp_path"]) or a.vararg or a.kwarg:
                    cprob[node.name] = "case may only take `tmp_path`"
                elif not node.body or not isinstance(node.body[-1], ast.Return) or node.body[-1].value is None \
                        or len(rets) != 1:
                    cprob[node.name] = "last statement must be the only `return <expr>`"
                elif any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)) for n in ast.walk(node)):
                    cprob[node.name] = "case must not be a generator/coroutine"
            continue
        if isinstance(node, ast.ClassDef):
            if any(isinstance(n, ast.Name) and n.id in aliases for b in node.bases + node.decorator_list
                   for n in ast.walk(b)):
                mprob.append("line %d: module-level class uses library objects; define it inside a helper" % node.lineno)
            if node.name.startswith("Test"):
                mprob.append("line %d: class names may not start with `Test`" % node.lineno)
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            try:
                ast.literal_eval(node.value)
                continue
            except Exception:
                pass
        mprob.append("line %d: only imports, literal constants, helpers and cases at module level" % node.lineno)
    if len(set(cases)) != len(cases):
        mprob.append("a case is defined twice")
    tops = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    users = {}
    for c in cases:
        seen, stack = set(), [c]
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in tops:
                continue
            seen.add(cur)
            stack += [n.id for n in ast.walk(tops[cur]) if isinstance(n, ast.Name)]
        for h in seen:
            users.setdefault(h, set()).add(c)

    def problem_of(n):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                return "calls `%s()` (non-deterministic or implementation detail)" % f.id
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in CLOCK:
                return "uses `%s.%s` (clock/env/randomness/side effect)" % (f.value.id, f.attr)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) and f.attr in ("now", "utcnow", "today"):
                return "reads the current date/time"
        if isinstance(n, ast.Attribute):
            if n.attr in PRIVATE_DUNDER:
                return "inspects `%s` (implementation detail)" % n.attr
            if n.attr.startswith("_") and not n.attr.startswith("__") and not (
                    isinstance(n.value, ast.Name) and n.value.id in ("self", "cls")):
                return "accesses private attribute `%s`" % n.attr
            if n.attr == "environ":
                return "reads the environment"
        if isinstance(n, ast.Name) and n.id == "__file__":
            return "uses __file__"
        if isinstance(n, ast.FormattedValue) and n.conversion == ord("r"):
            return "uses !r formatting (repr is not contractual)"
        return None
    for top in tree.body:
        if not isinstance(top, (ast.FunctionDef, ast.ClassDef)):
            continue
        for n in ast.walk(top):
            p = problem_of(n)
            if not p and not contract.messages and isinstance(n, ast.ExceptHandler) and n.name:
                for s in ast.walk(n):
                    if isinstance(s, ast.Name) and s.id == n.name and not (isinstance(n.type, ast.Name)):
                        pass
                    if (isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id in ("str", "format")
                            and any(isinstance(a, ast.Name) and a.id == n.name for a in s.args)) or \
                            (isinstance(s, ast.Attribute) and isinstance(s.value, ast.Name) and s.value.id == n.name
                             and s.attr in ("args", "msg", "message", "strerror")):
                        p = "uses the text of a caught exception (messages are not contractual)"
                        break
            if p:
                text = "line %d: %s" % (getattr(n, "lineno", 0), p)
                for c in users.get(top.name, ()):
                    cprob.setdefault(c, text)
    return mprob[:10], cprob, cases
