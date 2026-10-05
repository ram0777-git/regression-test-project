#!/usr/bin/env python3
"""
agent.py -- autonomous, contract-aware regression test generator.

    python agent.py REPO --task statement.txt [--tests-dir tests] [--max-rounds 6] [--max-mutants 150] [--no-llm]

Round 0: contract (deterministic + LLM JSON), ranked repository context, deterministic cases, LLM cases.
Round 1: record + emit pytest, coverage feedback.
Round 2+: mutation analysis -> prioritized survivors -> targeted cases (LLM, or deterministic re-sampling) -> repeat.
Final: determinism, refactor-tolerance (renamed locals / reworded messages), minimization, run inside the repo,
patch hygiene, optional patch file.  Only new test files under the task's test directory are ever written.

Benchmark entry point: ``agent_main(input)`` (see its docstring).
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # exec()'d or embedded without __file__
    HERE = ""


def _locate_here():
    cands = [HERE, os.getcwd(), os.path.dirname(os.path.abspath(sys.argv[0] or "."))] + list(sys.path)
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "casegen.py")) and os.path.isfile(os.path.join(c, "recorder.py")):
            return os.path.abspath(c)
    return os.path.abspath(HERE or os.getcwd())


HERE = _locate_here()
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import casegen  # noqa: E402
import contract as contract_mod  # noqa: E402
import emit  # noqa: E402
import mutation_engine as me  # noqa: E402
from llm_client import LLM  # noqa: E402

PY = sys.executable
RECORDER = os.path.join(HERE, "recorder.py")
T0 = time.time()
NOT_PACKAGES = {"tests", "test", "testing", "docs", "doc", "examples", "example", "scripts", "benchmarks",
                "build", "dist", "venv", ".venv", "env", "site-packages", "node_modules", "regression_tests",
                "migrations"}
SKIP_FILES = {"setup", "conftest", "__main__", "noxfile", "fabfile", "manage", "wsgi", "asgi"}
SKIP_SNAPSHOT = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
                 ".nox", "node_modules", ".venv", "venv", ".idea", ".vscode", ".eggs", ".hypothesis"}
ARTIFACT_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"}
DYNAMIC_CALLS = {"locals", "vars", "eval", "exec", "dir", "globals", "compile", "__import__"}
FAIL_RX = re.compile(r"^(?:FAILED|ERROR) (\S+\.py(?:::\S+)?)", re.M)


def log(msg):
    sys.stdout.write("[%5.0fs] %s\n" % (time.time() - T0, msg))
    sys.stdout.flush()


def dbg(msg):
    if os.environ.get("TG_DEBUG"):
        log(msg)


SYSTEM_PROMPT = """You are an expert Python test engineer writing a regression suite that pins down the
CONTRACT of a library: every behaviour change in scope must make a test fail, while a behaviour-preserving
refactoring must keep every test passing.

You never write expected values or pytest files. You write CASE FUNCTIONS; the tooling runs each case on the
current library, records the result, and emits a pytest test asserting it.

Write case files as fenced blocks whose info string is `python` followed by a NEW file name:

```python cases_parsing.py
from mylib import parse, Parser

def case_parse_plain_number():
    return parse("12")

def case_parser_state_between_calls():
    p = Parser(strict=True)
    first = p.feed("a=1")
    return [first, p.feed("b=2"), p.count]

def case_parse_rejects_empty():
    return parse("")          # raising cases become `with pytest.raises(<exact class>)`
```

Rules for case files
1. Every case is a module-level function `case_<what>()` with no parameters (it may take `tmp_path` for a
   temporary directory). Its LAST statement is the only `return <expr>`. No generators, no async.
2. The returned value must be plain data: None, bool, int, float, str, bytes, tuple, list, dict, set, frozenset
   (nested). Convert library objects to plain data through their PUBLIC API (attributes, methods, list(...),
   sorted(...)). Do not return objects, functions or iterators.
3. Import only the standard library and the library's PUBLIC modules (no leading underscores, no relative
   imports, no pytest). Respect any import restriction in the task.
4. At module level only imports, literal constants, helper functions/classes and `case_` functions. Helper names
   must not start with `test`. Add the line `EXCEPTION_CLASSES = True` only if the exact exception CLASS is part
   of the contract for the cases in that file.
5. Never touch private names (leading underscore), never call id/hash/repr/eval/exec/open/globals/vars, and do
   not use the clock, randomness, environment variables, network, subprocesses, threads, or files outside
   `tmp_path`.
6. Do not use the TEXT of caught exceptions (str(exc), exc.args) unless the task says messages are contractual.
   A case that should raise simply lets the exception propagate; to observe several outcomes in one case use a
   helper that returns the exception CLASS NAME.
7. Never pin behaviour the task lists as out of scope or as not part of the contract.
8. Prefer many small, sharp cases: boundaries (0, 1, -1, empty, max, off-by-one), every branch, defaults and
   keyword arguments, invalid inputs, argument mutation, state across several calls on one object, and ordering
   only where the contract defines it (do not depend on set/dict iteration order otherwise).
9. Cases must be deterministic and fast (well under a second).

Reply protocol
- Output the case files as fenced blocks (several files are fine) plus at most a few lines of notes.
- When mutants (ids like m12) are shown to you, write cases that make them fail. If you are certain a mutant is
  equivalent (cannot change observable behaviour) write a line `SKIP: m12, m31`.
- Write a line `DONE` on its own when no further cases would improve the suite.
"""

ANALYSIS_SYSTEM = """You analyse a task statement for a Python library and return ONE JSON object, nothing else.
Keys: scope {functions, classes, methods: lists of names that exist in the library}, in_scope_behaviors,
out_of_scope_behaviors, edge_cases, invalid_inputs, boundary_conditions, stateful_sequences, interaction_cases,
hygiene_rules, likely_bug_patterns, documentation_sources (all lists of short strings) and assertion_policy
{messages, stdout, state, ordering, exception_classes: booleans, false only when the statement says the
behaviour is NOT part of the contract}. Never invent names that are not in the provided symbol list."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root):
    """relative path -> content hash for every file that is not VCS/cache noise."""
    snap = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_SNAPSHOT]
        for f in fns:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            try:
                if os.path.islink(p):
                    snap[rel] = "link:" + os.readlink(p)
                    continue
                st = os.stat(p)
                snap[rel] = ("big:%d:%d" % (st.st_size, st.st_mtime_ns)) if st.st_size > 8 * 1024 * 1024 \
                    else sha1_file(p)
            except OSError:
                snap[rel] = "unreadable"
    return snap


def artifact_dirs(root):
    found = set()
    for dp, dns, _ in os.walk(root):
        dns[:] = [d for d in dns if d not in {".git", ".hg", ".svn", "node_modules", ".venv", "venv", ".tox"}]
        for d in list(dns):
            if d in ARTIFACT_DIRS:
                found.add(os.path.join(dp, d))
                dns.remove(d)
    return found


def stdlib_names():
    names = set(getattr(sys, "stdlib_module_names", ()) or ())
    names |= {"__future__", "abc", "argparse", "ast", "base64", "bisect", "builtins", "calendar", "collections",
              "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal", "enum", "fractions", "functools",
              "heapq", "itertools", "json", "math", "operator", "os", "pathlib", "re", "statistics", "string",
              "textwrap", "types", "typing", "unittest", "uuid", "io", "sys", "time", "random"}
    return names


def safe_tests_dir(d):
    d = (d or "tests").strip().replace("\\", "/")
    if os.path.isabs(d) or re.match(r"^[A-Za-z]:", d):
        return "tests"
    parts = [p for p in d.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return "tests"
    return "/".join(parts)


def load_statement(task):
    if not task:
        return ""
    try:
        if "\n" not in task and len(task) < 4096 and os.path.isfile(task):
            return read_text(task)
    except (OSError, ValueError):
        pass
    return task


def find_src_root(repo):
    src = os.path.join(repo, "src")
    if os.path.isdir(src):
        for d in sorted(os.listdir(src)):
            if os.path.isfile(os.path.join(src, d, "__init__.py")):
                return src
    return repo


def discover_modules(src_root, skip_rel=()):
    """Importable public modules below src_root: [{'name': dotted, 'rel': posix path}]."""
    skip = {s.strip("/") for s in skip_rel}
    mods = []
    for dp, dns, fns in os.walk(src_root):
        rel_dir = os.path.relpath(dp, src_root)
        rparts = [] if rel_dir == "." else rel_dir.split(os.sep)
        keep = []
        for d in sorted(dns):
            sub = os.path.join(dp, d)
            relsub = os.path.relpath(sub, src_root).replace(os.sep, "/")
            if d in NOT_PACKAGES or d.startswith((".", "_")) or d.endswith((".egg-info", ".dist-info")):
                continue
            if relsub in skip or not os.path.isfile(os.path.join(sub, "__init__.py")):
                continue
            keep.append(d)
        dns[:] = keep
        for f in sorted(fns):
            if not f.endswith(".py"):
                continue
            stem = f[:-3]
            if stem in SKIP_FILES or stem.startswith("test_") or stem.endswith("_test"):
                continue
            if stem.startswith("_") and stem != "__init__":
                continue
            if stem == "__init__":
                if not rparts:
                    continue
                name = ".".join(rparts)
            else:
                name = ".".join(rparts + [stem])
            mods.append({"name": name, "rel": os.path.relpath(os.path.join(dp, f), src_root).replace(os.sep, "/")})
    return mods


def scan_symbols(path, is_init):
    try:
        tree = ast.parse(read_text(path))
    except (SyntaxError, ValueError):
        return set()
    syms = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            syms.add(node.name)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        syms.add(sub.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and is_init:
            for a in node.names:
                syms.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    syms.add(t.id)
    return {s for s in syms if not s.startswith("_")}


def select_modules(mods, contract, symbols, scope_names):
    by_name = {m["name"]: m for m in mods}
    chosen = {}
    for tok in contract.modules:
        parts = tok.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in by_name:
                chosen[cand] = by_name[cand]
                break
    if scope_names:
        for m in mods:
            if symbols.get(m["name"], set()) & scope_names:
                chosen[m["name"]] = m
    return [chosen[k] for k in sorted(chosen)] if chosen else list(mods)


def drop_functions(source, names):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                              and n.name in names)]
    return ast.unparse(tree) + "\n"


def norm_id(nodeid):
    f, _, t = nodeid.partition("::")
    return os.path.basename(f) + (("::" + t.split("::")[-1].split("[")[0]) if t else "")


def make_patch(files):
    """Unified diff that adds the given {relpath: text} files (all new)."""
    out = []
    for rel in sorted(files):
        text = files[rel]
        if not text:
            continue
        lines = text.splitlines(keepends=True)
        body = []
        for ln in lines:
            body.append("+" + ln if ln.endswith("\n") else "+" + ln + "\n\\ No newline at end of file\n")
        out.append("diff --git a/%s b/%s\nnew file mode 100644\n--- /dev/null\n+++ b/%s\n@@ -0,0 +1,%d @@\n%s"
                   % (rel, rel, rel, len(lines), "".join(body)))
    return "".join(out)


# --------------------------------------------------------------------------- #
# behaviour-preserving source transformations (refactor-tolerance checks)
# --------------------------------------------------------------------------- #
def refactor_reformat(src):
    return ast.unparse(ast.parse(src)) + "\n", 1


class _Reword(ast.NodeTransformer):
    def __init__(self):
        self.n = 0

    def visit_Raise(self, node):
        self.generic_visit(node)
        exc = node.exc
        if isinstance(exc, ast.Call) and exc.args:
            a = exc.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                exc.args[0] = ast.Constant(a.value + " (reworded)")
                self.n += 1
            elif isinstance(a, ast.JoinedStr):
                a.values.append(ast.Constant(" (reworded)"))
                self.n += 1
        return node


def refactor_reword(src):
    tree = ast.parse(src)
    t = _Reword()
    t.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", t.n


def refactor_rename(src):
    """Rename function-local variables consistently (parameters and anything dynamic are left alone)."""
    tree = ast.parse(src)
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    taken |= {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    taken |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            taken |= {(a.asname or a.name).split(".")[0] for a in n.names}
    match_t = getattr(ast, "Match", ())
    done, changed = set(), 0
    for fn in list(ast.walk(tree)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or id(fn) in done:
            continue
        sub = list(ast.walk(fn))
        for s in sub:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                done.add(id(s))
        if any(isinstance(s, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.Import, ast.ImportFrom) + (
                (match_t,) if match_t else ())) for s in sub):
            continue
        if any(isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id in DYNAMIC_CALLS for s in sub):
            continue
        if any(isinstance(s, ast.JoinedStr) and any(
                isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value.rstrip().endswith("=")
                for v in s.values) for s in sub):
            continue
        argn = {s.arg for s in sub if isinstance(s, ast.arg)}
        stores = {s.id for s in sub if isinstance(s, ast.Name) and isinstance(s.ctx, ast.Store)}
        stores |= {s.name for s in sub if isinstance(s, ast.ExceptHandler) and s.name}
        cand = {n for n in stores if n not in argn and not n.startswith("__") and n != "_"}
        if not cand:
            continue
        mapping = {}
        for n in sorted(cand):
            new = n + "_rt"
            while new in taken or new in mapping.values():
                new += "_"
            mapping[n] = new
            taken.add(new)
        for s in sub:
            if isinstance(s, ast.Name) and s.id in mapping:
                s.id = mapping[s.id]
            elif isinstance(s, ast.ExceptHandler) and s.name in mapping:
                s.name = mapping[s.name]
        changed += len(mapping)
    return ast.unparse(tree) + "\n", changed


# --------------------------------------------------------------------------- #
# the agent
# --------------------------------------------------------------------------- #
class Agent:
    def __init__(self, repo, statement="", tests_dir=None, max_rounds=6, max_mutants=150, timeout=1800.0,
                 use_llm=True, model=None, budget=1.0, jobs=None, seed=0, report_path=None, patch_path=None,
                 apply=True, keep_work=False, src_root=None, target_score=95.0):
        self.repo = os.path.abspath(repo)
        if not os.path.isdir(self.repo):
            raise SystemExit("repository not found: %s" % repo)
        self.statement = statement or ""
        self.contract = contract_mod.parse_statement(self.statement)
        if tests_dir:
            self.contract.tests_dir = tests_dir
        self.tests_dir = safe_tests_dir(self.contract.tests_dir)
        self.contract.tests_dir = self.tests_dir
        self.max_rounds = max(1, int(max_rounds))
        self.max_mutants = max(1, int(max_mutants))
        self.timeout = float(timeout)
        self.t0 = time.time()
        self.deadline = self.t0 + self.timeout
        self.use_llm = use_llm
        self.model = model
        self.budget = budget
        self.jobs = int(jobs) if jobs else max(1, min(4, os.cpu_count() or 1))
        self.seed = int(seed)
        self.report_path = report_path
        self.patch_path = patch_path
        self.apply = apply
        self.keep_work = keep_work
        self.src_root_override = src_root
        self.target_score = float(target_score)
        self.stdlib = stdlib_names()
        self.audit = []
        self.cases = {}          # stem -> {"source", "origin", "round"}
        self.recs = {}           # stem -> {case name -> record}
        self.keep = {}           # stem -> set(case names)
        self.covered = {}        # rel file -> set(lines)
        self.kills = {}          # (stem, case name) -> set(mutant ids)
        self.file_to_stem = {}
        self.stage_texts = {}
        self.names_cache = {}
        self.evaluated = []
        self.all_mutants = []
        self.func_stats = {}
        self.suite_secs = 1.0
        self.mut_fast = False
        self.llm = None
        self.llm_fail = 0
        self.counter = 0
        self.final_files = {}
        self.hygiene = {"ok": None, "problems": [], "removed": []}
        self.patch = ""
        self.ok = False
        self.runner = None
        self.work = None
        self.api_text = ""
        self.packages = []
        self.scope = []
        self.known_names = set()
        self.scope_names = set()
        self.excluded = set()
        self.rel_to_module = {}
        self.symbols = {}
        self.llm_cases_used = False
        self.last_run = {}

    # ------------------------------------------------------------------ util
    def left(self):
        return self.deadline - time.time()

    def note(self, what, **kw):
        rec = {"what": what, "t": round(time.time() - self.t0, 1)}
        rec.update(kw)
        self.audit.append(rec)
        return rec

    def clean_env(self):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("OPENROUTER_API_KEY", None)
        return env

    def guarded(self, name, fn, *a, **k):
        t_phase = time.time()
        try:
            return fn(*a, **k)
        except Exception as e:          # a failing phase must never prevent producing the suite
            tb = traceback.format_exc()
            log("PHASE FAILED %s: %r" % (name, e))
            dbg(tb)
            self.note("phase-error", phase=name, error=repr(e), trace=tb[-1500:])
            return None
        finally:
            log("phase %s took %.1fs (%.0fs of deadline left)" % (name, time.time() - t_phase, self.left()))

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.snap0 = snapshot(self.repo)
        self.artifacts0 = artifact_dirs(self.repo)
        self.work = tempfile.mkdtemp(prefix="tgwork-")
        self.case_dir = os.path.join(self.work, "cases")
        self.stage_dir = os.path.join(self.work, "stage")
        self.scratch = os.path.join(self.work, "scratch")
        for d in (self.case_dir, self.stage_dir, self.scratch, os.path.join(self.work, "rec"),
                  os.path.join(self.work, "collect"), os.path.join(self.work, "mut")):
            os.makedirs(d, exist_ok=True)
        self.pytest_ini = os.path.join(self.work, "pytest.ini")
        write_text(self.pytest_ini, "[pytest]\naddopts =\n")
        log("repo=%s tests_dir=%s timeout=%ds llm=%s" % (self.repo, self.tests_dir, self.timeout, self.use_llm))

    def discover(self):
        self.src_root = os.path.abspath(self.src_root_override) if self.src_root_override else find_src_root(self.repo)
        tests_rel = os.path.relpath(os.path.join(self.repo, *self.tests_dir.split("/")), self.src_root)
        self.tests_rel = tests_rel.replace(os.sep, "/")
        mods = discover_modules(self.src_root, skip_rel=[self.tests_rel])
        self.symbols = {m["name"]: scan_symbols(os.path.join(self.src_root, m["rel"]),
                                                m["rel"].endswith("__init__.py")) for m in mods}
        self.known_names = set().union(*self.symbols.values()) if self.symbols else set()
        self.packages = sorted({m["name"].split(".")[0] for m in mods})
        self.rel_to_module = {m["rel"]: m["name"] for m in mods}
        self.recompute_scope(mods)
        self.compute_excluded()
        self.note("discovery", src_root=self.src_root, modules=len(mods), scope=[m["name"] for m in self.scope],
                  names=sorted(self.scope_names)[:40], excluded=sorted(self.excluded))
        log("discovered %d modules; in scope: %s" % (len(mods), ", ".join(m["name"] for m in self.scope) or "-"))
        self.all_mods = mods

    def recompute_scope(self, mods=None):
        mods = mods if mods is not None else self.all_mods
        self.scope_names = set(self.contract.names) & self.known_names
        self.scope = select_modules(mods, self.contract, self.symbols, self.scope_names)

    def compute_excluded(self):
        names = set()
        for s in self.contract.out_of_scope:
            for tok in re.findall(r"`([A-Za-z_][\w.]*)(?:\(\))?`", s):
                for p in tok.split("."):
                    if p and not p.startswith("_"):
                        names.add(p)
        self.excluded = names - set(self.contract.names)

    def is_excluded_label(self, label):
        parts = label.split(".")
        return bool(self.excluded) and (parts[0] in self.excluded or parts[-1] in self.excluded)

    # ------------------------------------------------------------------ recorder plumbing
    @staticmethod
    def scrub(text):
        """Remove anything that looks like a secret before it reaches the audit log / report."""
        text = text or ""
        for k, v in os.environ.items():
            if v and len(v) >= 6 and any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                text = text.replace(v, "<redacted>")
        return re.sub(r"(?i)(api[_-]?key|token|secret|password|authorization)(\s*[=:]\s*)\S+", r"\1\2<redacted>", text)

    def run_recorder(self, cfg, hashseed=0, timeout=300.0):
        self.counter += 1
        cfg = dict(cfg)
        cfg["out"] = os.path.join(self.work, "rec", "out%d.jsonl" % self.counter)
        cfg_path = os.path.join(self.work, "rec", "cfg%d.json" % self.counter)
        write_text(cfg_path, json.dumps(cfg))
        env = self.clean_env()
        env["PYTHONHASHSEED"] = str(hashseed)
        crashed, err, rc = False, "", 0
        try:
            p = subprocess.run([PY, RECORDER, cfg_path], cwd=self.scratch, env=env, capture_output=True, text=True,
                               timeout=max(5.0, min(timeout, self.left())))
            rc = p.returncode
            crashed = rc != 0
            err = self.scrub((p.stderr or "")[-600:])
        except subprocess.TimeoutExpired:
            # the subprocess timeout is the only watchdog on platforms without SIGALRM/setitimer (Windows)
            crashed, err, rc = True, "recorder timed out", -1
        rows = []
        try:
            for line in read_text(cfg["out"]).splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        self.last_run = {"returncode": rc, "crashed": crashed, "stderr": err, "rows": len(rows)}
        if crashed:
            self.note("recorder-exit", returncode=rc, stderr=err[-300:], rows=len(rows))
        return rows, crashed, err

    @staticmethod
    def parse_rows(rows):
        recs, errs, ended, api = {}, {}, False, None
        for r in rows:
            k = r.get("kind")
            if k == "case":
                recs[r["key"]] = r
            elif k == "module_error":
                errs[r["module"]] = r
            elif k == "end":
                ended = True
            elif k == "api":
                api = r.get("data")
        return recs, errs, ended, api

    def case_names(self, stem):
        if stem not in self.names_cache:
            try:
                tree = ast.parse(self.cases[stem]["source"])
                self.names_cache[stem] = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                                          and n.name.startswith("case_")]
            except SyntaxError:
                self.names_cache[stem] = []
        return self.names_cache[stem]

    def record(self, stems, trace=False, hashseed=0, reverse=False, only=None):
        cfg = {"mode": "run", "modules": list(stems), "sys_path": [self.case_dir, self.src_root],
               "packages": self.packages, "timeout": 5}
        if only:
            cfg["only"] = sorted(only)
        if reverse:
            cfg["reverse"] = True
        if trace:
            cfg["trace_root"] = self.src_root
            cfg["src_root"] = self.src_root
        # expected keys come from the case files we wrote, not from what the recorder says it ran
        expected = set(self.all_case_keys(stems))
        if only:
            expected &= set(only)
        rows, crashed, err = self.run_recorder(cfg, hashseed, timeout=max(60.0, 6.0 * len(expected)))
        first = dict(getattr(self, "last_run", {}) or {})
        recs, errs, ended, _ = self.parse_rows(rows)
        # a module that failed to import legitimately yields no cases: that is a module_error, not "missing"
        missing = [k for k in sorted(expected) if k not in recs and k.split("::")[0] not in errs]
        # `end` alone proves nothing: it may arrive with zero (or too few) case records
        if missing:
            reason = ("no end record" if not ended else "end record but no case records" if not recs
                      else "end record but incomplete case records")
            self.note("recorder-recover", reason=reason, expected=len(expected), returned=len(recs),
                      missing=len(missing), first_missing=missing[:5], returncode=first.get("returncode"),
                      stderr=(first.get("stderr") or "")[-300:], module_errors=sorted(errs))
            log("recorder: %s; expected=%d returned=%d missing=%d first=%s rc=%s" % (
                reason, len(expected), len(recs), len(missing), missing[:3], first.get("returncode")))
            recovered, failed = 0, 0
            for key in missing[:max(40, min(len(missing), 400))]:      # isolate the culprit(s) one by one
                if self.left() < 5:
                    break
                c2 = dict(cfg)
                c2["only"] = [key]
                c2.pop("reverse", None)
                r2, crashed2, err2 = self.run_recorder(c2, hashseed, timeout=20.0)
                rr, ee, _, _ = self.parse_rows(r2)
                errs.update(ee)
                if key in rr:
                    recs[key] = rr[key]
                    recovered += 1
                elif key.split("::")[0] in ee:
                    continue                                        # genuine module_error: keep it as such
                else:
                    failed += 1
                    why = "crashed or hung (rc=%s) %s" % ((getattr(self, "last_run", {}) or {}).get("returncode"),
                                                          self.scrub(err2)[-120:])
                    recs[key] = {"kind": "case", "key": key, "status": "error", "error": why.strip()}
            still = [k for k in expected if k not in recs and k.split("::")[0] not in errs]
            self.note("recorder-recover-done", expected=len(expected), recovered=recovered, failed=failed,
                      unattempted=len(still), first_unattempted=still[:5])
        return recs, errs

    def all_case_keys(self, stems):
        return ["%s::%s" % (s, n) for s in stems for n in self.case_names(s)]

    def stabilise(self, stems):
        """Record twice (different hash seed and order); keep only reproducible, fast cases."""
        a, errs = self.record(stems, trace=True, hashseed=0)
        b, _ = self.record(stems, hashseed=4242, reverse=True)
        good, dropped = {}, {}
        for key, ra in a.items():
            stem, _, name = key.partition("::")
            st = ra.get("status")
            if st not in ("value", "raises"):
                dropped.setdefault(stem, []).append((name, ra.get("error") or st))
                continue
            rb = b.get(key)
            same = bool(rb) and rb.get("status") == st
            if same and st == "value":
                same = rb.get("src") == ra.get("src")
            if same and st == "raises":
                same = rb.get("cls") == ra.get("cls") and (not self.contract.messages or rb.get("msg") == ra.get("msg"))
            if not same:
                dropped.setdefault(stem, []).append((name, "not reproducible across runs"))
                continue
            if max(ra.get("secs", 0), rb.get("secs", 0)) > 3.0:
                dropped.setdefault(stem, []).append((name, "too slow"))
                continue
            good.setdefault(stem, {})[name] = ra
            for f, ln in ra.get("lines") or []:
                self.covered.setdefault(str(f).replace(os.sep, "/"), set()).add(ln)
        for mod, e in errs.items():
            dropped.setdefault(mod, []).append(("<module>", e.get("error", "import error")))
        return good, dropped

    def register_case_file(self, stem, source, origin, rnd):
        self.cases[stem] = {"source": source, "origin": origin, "round": rnd}
        self.recs.pop(stem, None)
        self.keep.pop(stem, None)
        self.names_cache.pop(stem, None)
        write_text(os.path.join(self.case_dir, stem + ".py"), source)

    def activate(self, stems):
        """Record the new case files, keep reproducible cases, verify on the original source."""
        good, dropped = self.stabilise(stems)
        n_good = sum(len(v) for v in good.values())
        n_drop = sum(len(v) for v in dropped.values())
        n_exp = len(self.all_case_keys(stems))
        self.note("activate", stems=list(stems), expected=n_exp, reproducible=n_good, dropped=n_drop,
                  sample_dropped=[(m, n, str(w)[:80]) for m, v in list(dropped.items())[:2] for n, w in v[:2]])
        problems = []
        for stem in stems:
            self.recs[stem] = good.get(stem, {})
            self.keep[stem] = set(self.recs[stem])
            for name, why in dropped.get(stem, []):
                problems.append("%s: %s" % (name, str(why)[:200]))
        before = self.count_tests()
        self.verify_stage(hashseeds=(0,))
        after = self.count_tests()
        if before and after < before:
            self.note("activate-verify", before=before, after=after,
                      hint="pytest verification removed cases; see 'verify' notes")
        return problems

    # ------------------------------------------------------------------ pytest plumbing
    def run_pytest(self, paths, roots, cwd, hashseed=0, timeout=600.0, isolated=True, basetemp=None, fast=False):
        env = self.clean_env()
        env["PYTHONHASHSEED"] = str(hashseed)
        parts = list(roots)
        if env.get("PYTHONPATH"):
            parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", "--tb=no", "-rfE",
               "--disable-warnings"]
        if isolated:
            cmd += ["-c", self.pytest_ini]
        if basetemp:
            cmd += ["--basetemp", basetemp]
        if fast:
            cmd += ["-x"]
        cmd += list(paths)
        t = time.time()
        if self.left() < 3:                       # global deadline reached: never start another subprocess
            return {"status": "timeout", "failed": set(), "secs": 0.0, "out": "agent deadline reached"}
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                               timeout=max(5.0, min(timeout, self.left() + 5)))
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "failed": set(), "secs": time.time() - t, "out": ""}
        out = (p.stdout or "") + (p.stderr or "")
        rc = p.returncode
        status = {0: "pass", 1: "fail", 2: "collect_error", 5: "no_tests"}.get(rc, "error")
        return {"status": status, "failed": {norm_id(m) for m in FAIL_RX.findall(out)},
                "secs": time.time() - t, "out": out[-3000:]}

    def to_cases(self, ids, file_map=None):
        """normalised pytest ids -> {(stem, case name or None)}"""
        fm = file_map or self.file_to_stem
        res = set()
        for i in ids:
            f, _, t = i.partition("::")
            stem = fm.get(f)
            if stem is None:
                continue
            name = ("case_" + t[len("test_"):]) if t.startswith("test_") else None
            res.add((stem, name))
        return res

    def remove_cases(self, bad):
        for stem, name in bad:
            if name is None:
                self.keep[stem] = set()
            else:
                self.keep.get(stem, set()).discard(name)

    def build_stage(self):
        texts, self.file_to_stem = {}, {}
        for stem in sorted(self.cases):
            recs = self.recs.get(stem) or {}
            names = {n for n in (self.keep.get(stem) or set()) if n in recs}
            if not names:
                continue
            try:
                text = emit.module_source(self.cases[stem]["source"], stem, recs, names, self.contract)
            except Exception as e:
                self.note("emit-error", stem=stem, error=repr(e)[:300])
                self.keep[stem] = set()
                continue
            if text:
                fname = emit.test_file_name(stem)
                texts[fname] = text
                self.file_to_stem[fname] = stem
        shutil.rmtree(self.stage_dir, ignore_errors=True)
        os.makedirs(self.stage_dir, exist_ok=True)
        for fname, text in texts.items():
            write_text(os.path.join(self.stage_dir, fname), text)
        self.stage_texts = texts
        return texts

    def count_tests(self):
        return sum(len({n for n in (self.keep.get(s) or set()) if n in (self.recs.get(s) or {})}) for s in self.cases)

    def verify_stage(self, hashseeds=(0,), max_iter=4):
        for _ in range(max_iter):
            texts = self.build_stage()
            if not texts:
                return False
            bad, unknown = set(), False
            for hs in hashseeds:
                r = self.run_pytest([self.stage_dir], [self.src_root], self.scratch, hashseed=hs,
                                    timeout=max(60.0, 40 * self.suite_secs + 30))
                if r["status"] == "pass":
                    self.suite_secs = max(0.3, r["secs"])
                    continue
                found = self.to_cases(r["failed"])
                if found:
                    bad |= found
                else:
                    unknown = True
            if unknown and not bad:                      # cannot attribute: test the files one by one
                for fname in list(texts):
                    r = self.run_pytest([os.path.join(self.stage_dir, fname)], [self.src_root], self.scratch,
                                        timeout=120.0)
                    if r["status"] != "pass":
                        bad |= self.to_cases(r["failed"]) or {(self.file_to_stem[fname], None)}
                if not bad:
                    return True
            if not bad:
                return True
            self.note("verify", dropped=len(bad), sample=sorted(map(str, bad))[:5])
            self.remove_cases(bad)
        self.build_stage()
        r = self.run_pytest([self.stage_dir], [self.src_root], self.scratch, timeout=300.0)
        return r["status"] == "pass"

    # ------------------------------------------------------------------ LLM plumbing
    def llm_ok(self):
        return self.llm is not None and self.llm.available()

    def _llm_result(self, ok):
        if ok:
            self.llm_fail = 0
            return
        self.llm_fail += 1
        if self.llm_fail >= (1 if self.llm.calls == 0 else 2):
            self.llm.dead = True
            self.note("llm", report="LLM disabled after repeated failures; continuing deterministically")
            log("LLM unavailable -> deterministic fallback")

    def llm_json(self, system, prompt):
        if not self.llm_ok():
            return None
        try:
            data = self.llm.ask_json(system, prompt)
        except Exception as e:
            log("[llm] error %s" % type(e).__name__)
            data = None
        self._llm_result(data is not None)
        return data

    def llm_ask(self, text):
        if not self.llm_ok():
            return None
        try:
            reply = self.llm.ask(text)
        except Exception as e:
            log("[llm] error %s" % type(e).__name__)
            reply = None
        self._llm_result(reply is not None)
        return reply

    def init_llm(self):
        self.llm = LLM(model=self.model, budget=self.budget, deadline=self.deadline - 30, log=log,
                       enabled=self.use_llm)
        self.llm.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if not self.llm_ok():
            log("LLM not available (%s) -> deterministic mode" %
                ("disabled" if not self.use_llm else "no OPENROUTER_API_KEY"))
            self.note("llm", report="not used: %s" % ("disabled by flag" if not self.use_llm else "no API key"))

    # ------------------------------------------------------------------ phases
    def api_listing(self):
        mods = [m["name"] for m in self.scope]
        rows, _, _ = self.run_recorder({"mode": "api", "modules": mods, "sys_path": [self.src_root],
                                        "packages": self.packages}, timeout=120.0)
        _, _, _, api = self.parse_rows(rows)
        lines = []
        for mod in api or []:
            if mod.get("error"):
                continue
            lines.append("module %s" % mod["module"])
            for it in mod.get("items", []):
                if self.scope_names and it["name"] not in self.scope_names:
                    continue
                if self.is_excluded_label(it["name"]):
                    continue
                lines.append("  %s %s%s  # %s" % (it["kind"], it["name"], it.get("sig", ""), it.get("doc", "")))
                for mem in it.get("members", [])[:25]:
                    lines.append("      .%s" % mem)
        self.api_text = "\n".join(lines)[:12000]

    def contract_analysis(self):
        if not self.llm_ok():
            return
        prompt = ("Task statement:\n%s\n\nSymbols that exist in the library: %s\n\nDeterministic policy: %s"
                  % (self.statement[:8000], ", ".join(sorted(self.known_names))[:4000], self.contract.policy()))
        data = self.llm_json(ANALYSIS_SYSTEM, prompt)
        if data is None:
            return
        notes = contract_mod.merge_llm(self.contract, data, self.known_names)
        self.recompute_scope()
        self.compute_excluded()
        self.note("contract", policy=self.contract.policy(), notes=notes[:10], names=sorted(self.scope_names))

    def per_label(self):
        n = sum(len(self.symbols.get(m["name"], ())) for m in self.scope) or 1
        return max(6, min(30, 1500 // n))

    def deterministic(self, tag, rnd, seed, max_cases, per_label, names, mods=None):
        mods = list(mods) if mods else [m["name"] for m in self.scope]
        if not mods:
            return []

        def one(mod):
            return mod, casegen.collect_runtime(PY, self.src_root, mod, seed, max_cases,
                                                os.path.join(self.work, "collect"),
                                                timeout=max(30.0, min(600.0, self.left() / 2)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs) as ex:
            results = list(ex.map(one, mods))
        stems = []
        for mod, cases in results:
            cases = [c for c in cases if not self.is_excluded_label(c.get("label", ""))]
            if not cases:
                continue
            src = casegen.runtime_case_file(mod, cases, sorted(names), per_label, self.contract, tag)
            if not src:
                continue
            stem = "cases_%s%s" % (mod.replace(".", "_"), "" if tag == "det" else "_" + tag)
            self.register_case_file(stem, src, "det" if tag == "det" else "fb", rnd)
            stems.append(stem)
        return stems

    def deterministic_round(self):
        stems = self.deterministic("det", 0, self.seed, 40, self.per_label(), self.scope_names)
        if not stems:
            self.note("deterministic", report="no cases generated")
            return
        problems = self.activate(stems)
        self.note("deterministic", report="round 0: %d case files, %d tests kept" % (len(stems), self.count_tests()),
                  dropped=len(problems))
        log("deterministic cases: %d tests kept in %d files" % (self.count_tests(), len(stems)))

    def source_context(self, budget=70000):
        parts, used = [], 0
        order = sorted(self.scope, key=lambda m: os.path.getsize(os.path.join(self.src_root, m["rel"])))
        for m in order:
            text = read_text(os.path.join(self.src_root, m["rel"]))
            numbered = "\n".join("%4d  %s" % (i + 1, l) for i, l in enumerate(text.splitlines()))
            if used + len(numbered) > budget:
                numbered = numbered[: max(0, budget - used)] + "\n   ... (truncated)"
            parts.append("### %s\n```python\n%s\n```" % (m["rel"], numbered))
            used += len(numbered)
            if used >= budget:
                break
        return "\n\n".join(parts)

    def existing_tests_hint(self):
        out = []
        for dp, dns, fns in os.walk(self.repo):
            dns[:] = [d for d in dns if d not in SKIP_SNAPSHOT and d not in ("node_modules",)]
            for f in sorted(fns):
                if f.startswith("test_") and f.endswith(".py"):
                    out.append(os.path.relpath(os.path.join(dp, f), self.repo))
        return out[:40]

    def context_prompt(self):
        covered = sum(len(v) for v in self.covered.values())
        det = ", ".join("%s(%d)" % (s, len(self.keep.get(s) or ())) for s in sorted(self.cases)) or "none"
        hints = "\n".join("- %s: %s" % (k, "; ".join(v[:6])) for k, v in self.contract.hints.items())
        ex = self.existing_tests_hint()
        return ("# Task statement\n%s\n\n# Contract analysis\n%s\n%s\n\n# Public API in scope\n%s\n\n"
                "# Imports allowed from the library: %s\n\n# Already generated deterministically (%d lines covered): %s\n\n"
                "# Existing tests in the repository (do not duplicate their files): %s\n\n# Source\n%s\n\n"
                "Write case files that complement the deterministic ones: boundaries, every branch, invalid input, "
                "state sequences and interactions that the statement describes. Use new file names cases_<topic>.py."
                % (self.statement.strip()[:8000], self.contract.summary(), hints, self.api_text or "(unavailable)",
                   ", ".join(self.packages), covered, det[:1500], ", ".join(ex) or "none", self.source_context()))

    def ingest_reply(self, reply, rnd):
        files, shells, skips, done = casegen.parse_reply(reply)
        if shells:
            self.note("llm", report="ignored %d shell block(s) from the model (never executed)" % len(shells))
        stems, problems = [], []
        for name, body in files[:8]:
            stem = "cases_llm%d_%s" % (rnd, name[len("cases_"):-3])
            mprob, cprob, cases = casegen.lint(body, set(self.packages), self.stdlib, self.contract, self.known_names)
            if mprob:
                problems.append("%s: %s" % (name, "; ".join(mprob[:5])))
                continue
            src = body
            if cprob:
                src = drop_functions(body, set(cprob))
                for c, why in cprob.items():
                    problems.append("%s::%s: %s" % (name, c, why))
            if not [c for c in cases if c not in cprob]:
                problems.append("%s: no valid case functions" % name)
                continue
            self.register_case_file(stem, src, "llm", rnd)
            stems.append(stem)
        return stems, skips, done, problems

    def handle_reply(self, reply, rnd):
        """Ingest -> record -> verify; one repair turn for rejected material. Returns (stems, skips)."""
        stems, skips, done, problems = self.ingest_reply(reply, rnd)
        problems += self.activate(stems) if stems else []
        if problems and self.llm_ok():
            fix = self.llm_ask("Some of your cases were rejected or dropped:\n- " + "\n- ".join(problems[:25]) +
                               "\n\nResend ONLY corrected case files (new files or the same names). Do not repeat "
                               "cases that were accepted.")
            if fix:
                s2, k2, _, p2 = self.ingest_reply(fix, rnd)
                if s2:
                    self.activate(s2)
                stems = sorted(set(stems) | set(s2))
                skips |= k2
        self.llm_cases_used = self.llm_cases_used or bool(stems)
        return stems, skips

    def llm_round0(self):
        if not self.llm_ok():
            return
        reply = self.llm_ask(self.context_prompt())
        if not reply:
            return
        stems, _ = self.handle_reply(reply, 0)
        self.note("llm", report="round 0: %d LLM case files, %d tests total" % (len(stems), self.count_tests()))
        log("LLM cases: %d files; tests now %d" % (len(stems), self.count_tests()))

    # ------------------------------------------------------------------ mutation analysis
    def prepare_mutants(self):
        out, start = [], 0
        for m in self.scope:
            try:
                text = read_text(os.path.join(self.src_root, m["rel"]))
                ms = me.generate(m["rel"], text, set(self.scope_names), True, self.covered.get(m["rel"], set()), start)
            except Exception as e:
                self.note("mutation", report="skipped %s: %r" % (m["rel"], e))
                continue
            start += len(ms)
            out += ms
        out = [m for m in out if m.in_scope and not self.is_excluded_label(m.func)]
        random.Random(self.seed).shuffle(out)
        self.all_mutants = out

    def run_suite(self, root):
        if self.left() < 15:
            return "error", set()
        r = self.run_pytest([self.stage_dir], [root], self.scratch, hashseed=0,
                            timeout=max(30.0, 10 * self.suite_secs + 20), basetemp=root + "_bt", fast=self.mut_fast)
        if r["status"] == "pass":
            return "survived", set()
        if r["status"] == "fail":
            return "killed", set(r["failed"]) or {"<unknown>"}
        if r["status"] == "timeout":
            return "killed", {"<timeout>"}
        if r["status"] == "collect_error":
            return "killed", set(r["failed"]) or {"<collection>"}
        if not getattr(self, "mut_error_sample", None):
            self.mut_error_sample = "status=%s: %s" % (r["status"], r["out"][-800:])
        return "error", set()

    def ensure_runner(self):
        if self.runner is None:
            self.runner = me.MutationRunner(self.src_root, os.path.join(self.work, "mut"), self.jobs, self.run_suite,
                                            ignore=me.make_ignore(self.src_root, [self.tests_rel]))

    def evaluate(self, mutants):
        self.ensure_runner()
        self.runner.run(mutants, log=dbg)
        for m in mutants:
            if m.status == "killed":
                for kid in m.killers:
                    f, _, t = kid.partition("::")
                    stem = self.file_to_stem.get(f)
                    if stem and t.startswith("test_"):
                        self.kills.setdefault((stem, "case_" + t[5:]), set()).add(m.id)
        known = {m.id for m in self.evaluated}
        self.evaluated += [m for m in mutants if m.id not in known]
        stats = {}
        for m in self.evaluated:
            if m.status in ("killed", "survived"):
                k, s = stats.get((m.file, m.func), (0, 0))
                stats[(m.file, m.func)] = (k + (m.status == "killed"), s + (m.status == "survived"))
        self.func_stats = stats

    def mut_summary(self):
        c = {"killed": 0, "survived": 0, "equivalent": 0, "error": 0}
        for m in self.evaluated:
            if m.status in c:
                c[m.status] += 1
        denom = c["killed"] + c["survived"]
        return {"mutants_tried": len(self.evaluated), "killed": c["killed"], "survived": c["survived"],
                "equivalent": c["equivalent"], "errors": c["error"],
                "score": round(100.0 * c["killed"] / denom, 1) if denom else 100.0}

    def survivors(self):
        return [m for m in self.evaluated if m.status == "survived"]

    def survivor_text(self, ms):
        parts = []
        for m in ms:
            try:
                src = read_text(os.path.join(self.src_root, m.file)).splitlines()
            except OSError:
                src = []
            lo, hi = max(0, m.line - 6), min(len(src), m.line + 5)
            ctx = "\n".join("%4d  %s" % (i + 1, src[i]) for i in range(lo, hi))
            parts.append("### %s  %s:%d in `%s`  (%s, %s)\n  original: %s\n  mutated:  %s\n%s"
                         % (m.id, m.file, m.line, m.func, m.kind, "covered" if m.covered else "NOT covered by any case",
                            m.original, m.mutated, ctx))
        return "\n\n".join(parts)

    def llm_feedback(self, rnd, targets):
        prompt = ("Round %d. The current suite does NOT detect these mutants (each is a small behaviour change in "
                  "the library). Write targeted case files that make them fail, using only public behaviour.\n\n%s\n\n"
                  "Current score: %s. Mark certain equivalents with `SKIP: <ids>`; write `DONE` if nothing more helps."
                  % (rnd, self.survivor_text(targets), self.mut_summary()["score"]))
        reply = self.llm_ask(prompt)
        if not reply:
            return 0
        stems, skips = self.handle_reply(reply, rnd)
        ids = {m.id for m in self.evaluated if m.status == "survived"}
        for m in self.evaluated:
            if m.id in skips and m.id in ids:
                m.status = "equivalent"
        added = sum(len(self.keep.get(s) or ()) for s in stems)
        self.note("llm", report="fb round %d: %d LLM case files, %d tests, %d marked equivalent"
                  % (rnd, len(stems), added, len(skips & ids)))
        return added

    def det_feedback(self, rnd, survivors):
        by_mod = {}
        for m in survivors:
            top = m.func.split(".")[0] if m.func and m.func != "<module>" else ""
            mod = self.rel_to_module.get(m.file)
            if top and not top.startswith("_") and mod:
                by_mod.setdefault(mod, set()).add(top)
        if not by_mod:
            self.note("deterministic", report="fb round %d: survivors are in private/unreachable code" % rnd)
            return 0
        new = []
        seed = self.seed + 1000 * rnd + 7
        for mod, names in sorted(by_mod.items()):
            stems = self.deterministic("fb%d" % rnd, rnd, seed, 90 + 30 * rnd, 120, names, mods=[mod])
            new += stems
        if not new:
            self.note("deterministic", report="fb round %d: no new cases could be generated" % rnd)
            return 0
        self.activate(new)
        before = {m.id for m in survivors}
        self.evaluate(survivors)
        # keep only the new tests that actually killed a former survivor
        useful = {}
        for (stem, name), ids in self.kills.items():
            if stem in new and ids & before:
                useful.setdefault(stem, set()).add(name)
        total = 0
        for stem in new:
            self.keep[stem] = useful.get(stem, set()) & set(self.recs.get(stem, {}))
            total += len(self.keep[stem])
        self.verify_stage(hashseeds=(0,))
        killed_now = sum(1 for m in survivors if m.status == "killed")
        self.note("deterministic", report="fb round %d: re-sampled %d modules, kept %d tests, %d survivors killed"
                  % (rnd, len(by_mod), total, killed_now))
        log("fb round %d: kept %d new tests, killed %d survivors" % (rnd, total, killed_now))
        return total

    def mutation_rounds(self):
        if not self.count_tests():
            self.note("mutation", report="no tests to analyse")
            return
        self.prepare_mutants()
        if not self.all_mutants:
            self.note("mutation", report="no mutants in scope")
            return
        self.mut_fast = (self.suite_secs * min(self.max_mutants, len(self.all_mutants)) / max(1, self.jobs)
                         > 0.3 * self.left())
        func_stats = {}
        ranked = sorted(self.all_mutants, key=lambda m: -me.priority(m, func_stats))
        pool = ranked[: self.max_mutants]
        reserve = max(60.0, 0.25 * self.timeout)
        log("mutation analysis: %d mutants in scope, running %d" % (len(self.all_mutants), len(pool)))
        self.evaluate(pool)
        s = self.mut_summary()
        self.note("mutation round", round=1, **s)
        if s["killed"] + s["survived"] == 0:
            sample = getattr(self, "mut_error_sample", None) or "no mutant produced a usable result (deadline?)"
            log("mutation analysis produced no usable results (%d error); first problem: %s" % (s["errors"], sample[-400:]))
            self.note("mutation", report="no usable mutant results; skipping feedback rounds", sample=sample[-800:])
            shutil.rmtree(os.path.join(self.work, "mut"), ignore_errors=True)
            self.runner = None
            return
        log("round 1: score %.1f%% (%d killed, %d survived)" % (s["score"], s["killed"], s["survived"]))
        for rnd in range(2, self.max_rounds + 1):
            surv = self.survivors()
            if not surv or s["score"] >= self.target_score or self.left() < reserve:
                break
            surv.sort(key=lambda m: -me.priority(m, self.func_stats))
            targets = surv[:12]
            for t in targets:
                t.shown += 1
            added = self.llm_feedback(rnd, targets) if self.llm_ok() else 0
            if added:
                self.evaluate(self.survivors())
            else:
                self.det_feedback(rnd, self.survivors())
            s = self.mut_summary()
            self.note("mutation round", round=rnd, **s)
            log("round %d: score %.1f%% (%d killed, %d survived)" % (rnd, s["score"], s["killed"], s["survived"]))
        if not self.contract.messages:
            # exception-message wording is declared non-contractual: such mutants are unobservable by contract
            for m in self.survivors():
                if m.kind == "constant" and m.original.lstrip().startswith("raise ") \
                        and m.mutated.lstrip()[:1] in ("'", '"'):
                    m.status = "equivalent"
            self.note("mutation", report="message-only survivors classified equivalent under the contract")
        shutil.rmtree(os.path.join(self.work, "mut"), ignore_errors=True)
        self.runner = None

    # ------------------------------------------------------------------ refactor tolerance, minimisation
    def refactor_checks(self):
        if not self.count_tests():
            return
        kinds = [("reformat", refactor_reformat), ("rename-locals", refactor_rename)]
        if not self.contract.messages:
            kinds.append(("reword-messages", refactor_reword))
        rels = [m["rel"] for m in self.scope]
        ignore = me.make_ignore(self.src_root, [self.tests_rel])
        for label, fn in kinds:
            if self.left() < 40:
                break
            dest = os.path.join(self.work, "refactor_" + label)
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(self.src_root, dest, ignore=ignore, symlinks=True)
            changed = 0
            for rel in rels:
                path = os.path.join(dest, rel)
                try:
                    new, n = fn(read_text(path))
                    compile(new, rel, "exec")
                except Exception:
                    continue
                write_text(path, new)
                changed += n
            self.build_stage()
            r = self.run_pytest([self.stage_dir], [dest], self.scratch, timeout=max(60.0, 20 * self.suite_secs + 30))
            bad = self.to_cases(r["failed"]) if r["status"] != "pass" else set()
            if r["status"] not in ("pass", "fail") or (r["status"] == "fail" and not bad):
                bad = set()           # cannot attribute -> do not guess
            self.remove_cases(bad)
            self.note("refactor", kind=label, transformed=changed, status=r["status"], brittle=len(bad),
                      sample=sorted(map(str, bad))[:5])
            log("refactor check %-16s %s (%d sites): %d brittle tests removed" % (label, r["status"], changed, len(bad)))
            shutil.rmtree(dest, ignore_errors=True)

    def minimise(self):
        total = self.count_tests()
        limit = 2500
        if total <= limit and self.suite_secs <= 0.5 * self.contract.suite_limit:
            self.note("minimise", report="not needed (%d tests, %.1fs)" % (total, self.suite_secs))
            return
        entries = []
        for stem in self.cases:
            for name in self.keep.get(stem, set()):
                label = re.sub(r"_\d+$", "", name)
                entries.append((len(self.kills.get((stem, name), ())), stem, name, label))
        entries.sort()
        per_label = {}
        for _, _, _, label in entries:
            per_label[label] = per_label.get(label, 0) + 1
        removed = 0
        for kills, stem, name, label in entries:
            if total - removed <= limit * 0.8 and self.suite_secs <= 0.5 * self.contract.suite_limit:
                break
            if kills == 0 and per_label[label] > 1:
                self.keep[stem].discard(name)
                per_label[label] -= 1
                removed += 1
        self.note("minimise", report="removed %d zero-kill tests (%d -> %d)" % (removed, total, total - removed))

    # ------------------------------------------------------------------ final verification, hygiene, patch
    def write_to_repo(self):
        tdir = os.path.join(self.repo, *self.tests_dir.split("/"))
        files, self.repo_file_to_stem = {}, {}
        for fname in sorted(self.stage_texts):
            target, k = fname, 1
            while os.path.exists(os.path.join(tdir, target)):
                k += 1
                target = fname[:-3] + "_gen%d.py" % k
            path = os.path.abspath(os.path.join(tdir, target))
            if os.path.commonpath([path, os.path.abspath(tdir)]) != os.path.abspath(tdir):
                continue
            write_text(path, self.stage_texts[fname])
            rel = "%s/%s" % (self.tests_dir, target)
            files[rel] = self.stage_texts[fname]
            self.repo_file_to_stem[target] = self.file_to_stem[fname]
        return files

    def remove_repo_files(self, files):
        for rel in files:
            try:
                os.remove(os.path.join(self.repo, *rel.split("/")))
            except OSError:
                pass

    def finalize(self):
        ok = self.verify_stage(hashseeds=(0, 1, 4242), max_iter=3)
        if not ok or not self.stage_texts:
            self.note("final", report="no verified tests")
            return
        if not self.apply:
            self.final_files = {"%s/%s" % (self.tests_dir, f): t for f, t in self.stage_texts.items()}
            self.note("final", report="apply disabled; %d files staged only" % len(self.final_files))
            self.ok = True
            return
        for attempt in range(3):
            files = self.write_to_repo()
            paths = [os.path.join(self.repo, *rel.split("/")) for rel in files]
            bad, unknown = set(), False
            for hs in (0, 7, 4242):
                r = self.run_pytest(paths, [self.src_root], self.repo, hashseed=hs, isolated=False,
                                    timeout=max(120.0, 40 * self.suite_secs + 60))
                if r["status"] == "pass":
                    continue
                found = self.to_cases(r["failed"], self.repo_file_to_stem)
                if found:
                    bad |= found
                else:
                    unknown = True
            self.note("final", attempt=attempt + 1, files=len(files), failing=len(bad), unknown_failure=unknown)
            if not bad and not unknown:
                self.final_files = files
                self.ok = True
                return
            self.remove_repo_files(files)
            if bad:
                self.remove_cases(bad)
                if not self.verify_stage(hashseeds=(0,), max_iter=2):
                    return
            else:
                return
        self.note("final", report="could not obtain a clean run inside the repository")

    def cleanup_artifacts(self):
        removed = []
        for d in artifact_dirs(self.repo) - self.artifacts0:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(os.path.relpath(d, self.repo))
        now = snapshot(self.repo)
        allowed_prefix = self.tests_dir + "/"
        for rel in sorted(set(now) - set(self.snap0)):
            if rel.startswith(allowed_prefix) and rel.endswith(".py") and rel in self.final_files:
                continue
            try:
                os.remove(os.path.join(self.repo, *rel.split("/")))
                removed.append(rel)
            except OSError:
                pass
        return removed

    def hygiene_check(self):
        removed = self.cleanup_artifacts()
        now = snapshot(self.repo)
        problems = []
        for rel, h in self.snap0.items():
            if rel not in now:
                problems.append("deleted: " + rel)
            elif now[rel] != h:
                problems.append("modified: " + rel)
        new = sorted(set(now) - set(self.snap0))
        for rel in new:
            if not (rel.startswith(self.tests_dir + "/") and rel.endswith(".py")):
                problems.append("unexpected new file: " + rel)
        if self.apply and sorted(new) != sorted(self.final_files):
            problems.append("new files differ from the delivered set")
        for rel, text in self.final_files.items():
            try:
                ast.parse(text)
            except SyntaxError as e:
                problems.append("syntax error in %s: %s" % (rel, e.msg))
        if os.path.isdir(os.path.join(self.repo, ".git")):
            st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=self.repo,
                                capture_output=True, text=True)
            dirty = [l[3:] for l in st.stdout.splitlines()
                     if l[3:] not in set(self.snap0) and not l[3:].startswith(self.tests_dir + "/")]
            for d in dirty:
                if d.endswith(".pyc") or "__pycache__" in d or ".pytest_cache" in d:
                    problems.append("stray artifact: " + d)
        self.hygiene = {"ok": not problems, "problems": problems, "removed": removed, "new_files": new}
        self.note("hygiene", ok=not problems, problems=problems[:10], removed=removed[:10])
        log("patch hygiene: %s" % ("clean" if not problems else "; ".join(problems[:5])))
        if problems:
            self.ok = False

    def build_patch(self):
        self.patch = make_patch(self.final_files)
        if not self.patch:
            return
        tmp = tempfile.mkdtemp(prefix="tgpatch-")
        try:
            pf = os.path.join(tmp, "p.patch")
            write_text(pf, self.patch)
            chk = subprocess.run(["git", "apply", "--check", pf], cwd=tmp, capture_output=True, text=True)
            self.note("patch", applies_cleanly=chk.returncode == 0, files=sorted(self.final_files),
                      error=(chk.stderr or "")[-300:] if chk.returncode else "")
        except OSError as e:
            self.note("patch", error=repr(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if self.patch_path:
            write_text(self.patch_path, self.patch)

    # ------------------------------------------------------------------ orchestration
    def run(self):
        try:
            self.setup()
            self.guarded("discover", self.discover)
            if not getattr(self, "scope", None):
                self.note("discovery", report="no importable modules found")
            else:
                self.guarded("llm-init", self.init_llm)
                self.guarded("api", self.api_listing)
                self.guarded("contract", self.contract_analysis)
                self.guarded("deterministic", self.deterministic_round)
                self.guarded("llm-round0", self.llm_round0)
                self.guarded("verify", self.verify_stage)
                self.guarded("mutation", self.mutation_rounds)
                self.guarded("refactor", self.refactor_checks)
                self.guarded("minimise", self.minimise)
            self.guarded("finalize", self.finalize)
            self.guarded("hygiene", self.hygiene_check)
            self.guarded("patch", self.build_patch)
        finally:
            if self.work and not self.keep_work:
                shutil.rmtree(self.work, ignore_errors=True)
            if self.work:
                try:
                    self.cleanup_artifacts()
                except Exception:
                    pass
        return self.result()

    def result(self):
        ms = self.mut_summary()
        survivors = [{"id": m.id, "file": m.file, "line": m.line, "func": m.func, "kind": m.kind,
                      "original": m.original, "mutated": m.mutated} for m in self.survivors()[:30]]
        tests = {os.path.basename(rel): len(re.findall(r"(?m)^def test_", t)) for rel, t in self.final_files.items()}
        report = dict(ms)
        report.update({
            "tests": sum(tests.values()), "test_files": tests, "tests_dir": self.tests_dir,
            "contract": self.contract.policy(), "scope": [m["name"] for m in self.scope] if self.scope else [],
            "llm": {"used": bool(self.llm and self.llm.calls), "calls": self.llm.calls if self.llm else 0,
                    "cost_usd": round(self.llm.spent, 4) if self.llm else 0.0,
                    "cases_used": self.llm_cases_used},
            "hygiene": self.hygiene, "survivors": survivors, "suite_seconds": round(self.suite_secs, 2),
            "elapsed_seconds": round(time.time() - self.t0, 1), "source_modified": any(
                p.startswith(("modified:", "deleted:")) for p in self.hygiene.get("problems", [])),
        })
        return {"ok": bool(self.ok and self.final_files and self.hygiene.get("ok")), "patch": self.patch,
                "files": dict(self.final_files), "report": report, "audit": self.audit}


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def agent_main(input):
    """Benchmark entry point.

    ``input`` is a dict (or a JSON string of one) carrying the repository path and the problem statement under
    any of the usual key names, or a plain string, which is taken as the problem statement for the repository in
    the current directory.  Optional keys: tests_dir, timeout, max_rounds, max_mutants, use_llm / no_llm, model,
    budget, apply (False = do not write into the repository, only return the patch), report_path, patch_path.
    Returns {"success", "patch", "files", "tests_dir", "report", "audit"}.
    """
    data = input
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8", "replace")
    if isinstance(data, str):
        s = data.strip()
        parsed = None
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
            except ValueError:
                parsed = None
        data = parsed if isinstance(parsed, dict) else {"problem_statement": input}
    if not isinstance(data, dict):
        raise TypeError("agent_main expects a dict, a JSON string or a problem statement string")

    def pick(*keys, default=None):
        for k in keys:
            v = data.get(k)
            if v not in (None, ""):
                return v
        return default

    repo = pick("repo", "repo_path", "repo_dir", "repository", "repository_path", "path", "root", "project_dir",
                "workdir", "cwd", default=os.getcwd())
    statement = load_statement(str(pick("problem_statement", "statement", "task", "task_description", "problem",
                                        "prompt", "description", "instructions", default="")))
    use_llm = not bool(pick("no_llm", default=False)) and bool(pick("use_llm", default=True))
    agent = Agent(repo, statement, tests_dir=pick("tests_dir", "test_dir"),
                  max_rounds=int(pick("max_rounds", default=6)), max_mutants=int(pick("max_mutants", default=150)),
                  timeout=float(pick("timeout", "time_limit", default=1800)), use_llm=use_llm,
                  model=pick("model"), budget=float(pick("budget", default=1.0)), jobs=pick("jobs"),
                  seed=int(pick("seed", default=0)), report_path=pick("report_path", "report"),
                  patch_path=pick("patch_path", "patch_file"), apply=bool(pick("apply", default=True)),
                  src_root=pick("src_root"))
    res = agent.run()
    if agent.report_path:
        write_text(agent.report_path, json.dumps({"report": res["report"], "audit": res["audit"],
                                                  "ok": res["ok"]}, indent=2, default=str))
    return {"success": res["ok"], "patch": res["patch"], "files": sorted(res["files"]),
            "tests_dir": agent.tests_dir, "report": res["report"], "audit": res["audit"]}


def default_report_path():
    base = HERE if not os.path.abspath(os.getcwd()).startswith(HERE) else tempfile.gettempdir()
    return os.path.join(base, "agent_report.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Contract-aware regression test generator (pytest, tests only).")
    ap.add_argument("repo", help="path to the repository")
    ap.add_argument("--task", "--problem", dest="task", default="",
                    help="problem statement: a file path or the literal text")
    ap.add_argument("--tests-dir", help="override the test directory (default: from the task, else 'tests')")
    ap.add_argument("--max-rounds", type=int, default=6, help="max mutation/feedback rounds (default 6)")
    ap.add_argument("--max-mutants", type=int, default=150, help="max mutants evaluated (default 150)")
    ap.add_argument("--timeout", type=float, default=1800, help="overall wall-clock budget in seconds")
    ap.add_argument("--no-llm", action="store_true", help="deterministic generation only")
    ap.add_argument("--model", help="OpenRouter model id (default: $TG_MODEL or openai/gpt-5-mini)")
    ap.add_argument("--budget", type=float, default=1.0, help="LLM spend limit in USD")
    ap.add_argument("--jobs", type=int, help="parallel workers (default min(4, cpus))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", help="write the JSON report here")
    ap.add_argument("--patch", help="write the patch (new test files only) here")
    ap.add_argument("--no-apply", action="store_true", help="do not write into the repository; only report/patch")
    ap.add_argument("--src-root", help="directory that must be on sys.path (default: auto)")
    ap.add_argument("--keep-work", action="store_true", help="keep the temporary work directory")
    ns = ap.parse_args(argv)

    agent = Agent(ns.repo, load_statement(ns.task), tests_dir=ns.tests_dir, max_rounds=ns.max_rounds,
                  max_mutants=ns.max_mutants, timeout=ns.timeout, use_llm=not ns.no_llm, model=ns.model,
                  budget=ns.budget, jobs=ns.jobs, seed=ns.seed, report_path=ns.report, patch_path=ns.patch,
                  apply=not ns.no_apply, keep_work=ns.keep_work, src_root=ns.src_root)
    res = agent.run()
    rp = ns.report or default_report_path()
    write_text(rp, json.dumps({"report": res["report"], "audit": res["audit"], "ok": res["ok"]}, indent=2,
                              default=str))
    r = res["report"]
    log("tests=%d files=%d mutation score=%.1f%% (%d/%d tried) hygiene=%s report=%s" % (
        r["tests"], len(res["files"]), r["score"], r["killed"], r["mutants_tried"],
        "ok" if r["hygiene"].get("ok") else "FAILED", rp))
    if not res["files"]:
        log("no tests were produced")
        return 2
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
