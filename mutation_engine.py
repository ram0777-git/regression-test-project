"""Mutant generation with metadata, prioritisation, and parallel execution recording per-test killers."""
from __future__ import annotations

import ast
import concurrent.futures
import copy
import dataclasses
import os
import shutil

CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
       ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Div,
       ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult, ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd,
       ast.LShift: ast.RShift, ast.RShift: ast.LShift}
TWINS = {"min": "max", "max": "min", "any": "all", "all": "any", "startswith": "endswith",
         "endswith": "startswith", "lower": "upper", "upper": "lower", "lstrip": "rstrip", "rstrip": "lstrip",
         "floor": "ceil", "ceil": "floor", "append": "extend"}
EXC_SWAP = {"ValueError": "TypeError", "TypeError": "ValueError", "KeyError": "IndexError",
            "IndexError": "KeyError", "LookupError": "ValueError"}
QUIET = {"print", "debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
SKIP_FIELDS = {"annotation", "returns", "type_comment", "decorator_list"}


@dataclasses.dataclass
class Mutant:
    id: str
    file: str
    line: int
    func: str
    kind: str
    original: str
    mutated: str
    index: int
    variant: str
    cls: str = ""
    in_scope: bool = False
    named: bool = False
    covered: bool = False
    status: str = "pending"          # pending | killed | survived | error | skipped | equivalent
    killers: list = dataclasses.field(default_factory=list)
    shown: int = 0
    priority: float = 0.0

    def meta(self):
        d = dataclasses.asdict(self)
        d["killers"] = d["killers"][:5]
        return d


def walk(tree):
    """DFS list of (node, parent, field, qualname, in_function) with a stable order."""
    out = []

    def rec(node, parent, field, qual, infn):
        out.append((node, parent, field, qual, infn))
        q, f = qual, infn
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            q = (qual + "." if qual else "") + node.name
            f = infn or not isinstance(node, ast.ClassDef)
        for name, val in ast.iter_fields(node):
            if name in SKIP_FIELDS:
                continue
            for child in (val if isinstance(val, list) else [val]):
                if isinstance(child, ast.AST):
                    rec(child, node, name, q, f)
    rec(tree, None, None, "", False)
    return out


def _is_doc(node, parent):
    return isinstance(parent, ast.Expr) and isinstance(node, ast.Constant) and isinstance(node.value, str)


def variants(node, parent, field):
    v = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
        v.append(("compare", "flip"))
        if isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            v.append(("boundary", "reverse"))
    elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in BIN:
        v.append(("arith", "swap"))
        if isinstance(node, ast.BinOp):
            v.append(("operand", "left"))
    elif isinstance(node, ast.BoolOp):
        v.append(("bool", "flip"))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
        v.append(("unary", "remove"))
    elif isinstance(node, ast.Constant) and not _is_doc(node, parent) and not isinstance(
            parent, (ast.JoinedStr, ast.FormattedValue)):
        kind = "default" if isinstance(parent, ast.arguments) else "constant"
        if isinstance(node.value, bool) or isinstance(node.value, (int, float)) or \
                (isinstance(node.value, str) and len(node.value) < 40) or (node.value is None and kind == "default"):
            v.append((kind, "change"))
    elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
        v.append(("branch", "negate"))
        if isinstance(node, ast.If) and node.orelse:
            v.append(("branch", "swap"))
    elif isinstance(node, ast.Return) and node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None):
        v.append(("return", "none"))
    elif isinstance(node, ast.Call):
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name in TWINS and name != "append":
            v.append(("twin", TWINS[name]))
    elif isinstance(node, (ast.Break, ast.Continue)):
        v.append(("loop", "swap"))
    elif isinstance(node, ast.Raise) and node.exc is not None:
        e = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(e, ast.Name) and e.id in EXC_SWAP:
            v.append(("exception", EXC_SWAP[e.id]))
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        f = node.value.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if name not in QUIET and len(parent.body if hasattr(parent, "body") else [1, 1]) > 1:
            v.append(("statement", "delete"))
    return v


def _replace(parent, field, old, new):
    val = getattr(parent, field)
    if isinstance(val, list):
        setattr(parent, field, [new if x is old else x for x in val])
    else:
        setattr(parent, field, new)


def apply(tree, index, kind, variant):
    tree = copy.deepcopy(tree)
    node, parent, field, _, _ = walk(tree)[index]
    if kind == "compare":
        node.ops[0] = CMP[type(node.ops[0])]()
    elif kind == "boundary":
        node.ops[0] = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}[type(node.ops[0])]()
    elif kind == "arith":
        node.op = BIN[type(node.op)]()
    elif kind == "operand":
        _replace(parent, field, node, node.left)
    elif kind == "bool":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "unary":
        _replace(parent, field, node, node.operand)
    elif kind in ("constant", "default"):
        x = node.value
        node.value = (not x) if isinstance(x, bool) else 0 if x is None else (x + 1) if isinstance(x, (int, float)) \
            else ("" if x else "x")
    elif kind == "branch" and variant == "negate":
        node.test = ast.UnaryOp(ast.Not(), node.test)
    elif kind == "branch":
        node.body, node.orelse = node.orelse, node.body
    elif kind == "return":
        node.value = ast.Constant(None)
    elif kind == "twin":
        if isinstance(node.func, ast.Name):
            node.func.id = variant
        else:
            node.func.attr = variant
    elif kind == "loop":
        _replace(parent, field, node, ast.Continue() if isinstance(node, ast.Break) else ast.Break())
    elif kind == "exception":
        e = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        e.id = variant
    elif kind == "statement":
        _replace(parent, field, node, ast.Pass())
    return ast.fix_missing_locations(tree)


def generate(rel, text, scope_names, scope_file, covered_lines, start_id=0):
    tree = ast.parse(text)
    base = ast.unparse(tree)
    lines = text.splitlines()
    out, seen = [], set()
    for idx, (node, parent, field, qual, infn) in enumerate(walk(tree)):
        if not hasattr(node, "lineno"):
            continue
        if not infn and not (isinstance(parent, (ast.Assign, ast.AnnAssign)) or isinstance(node, ast.Assign)):
            if not isinstance(parent, ast.arguments):
                continue
        for kind, var in variants(node, parent, field):
            try:
                new = ast.unparse(apply(tree, idx, kind, var))
            except Exception:
                continue
            if new == base or hash(new) in seen:
                continue
            seen.add(hash(new))
            try:
                compile(new, rel, "exec")
                mutated_node = walk(ast.parse(new))[idx][0] if kind not in ("operand", "unary", "loop", "statement") \
                    else None
                mut_txt = ast.unparse(mutated_node) if mutated_node is not None else "(%s %s)" % (kind, var)
            except Exception:
                continue
            parts = qual.split(".") if qual else []
            named = any(p in scope_names for p in parts)
            cls = parts[0] if parts and parts[0][:1].isupper() else ""
            m = Mutant(id="m%d" % (start_id + len(out) + 1), file=rel, line=node.lineno, func=qual or "<module>",
                       cls=cls, kind=kind, original=(lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""),
                       mutated=mut_txt[:200], index=idx, variant=var, named=named,
                       in_scope=named or (scope_file and not scope_names) or (scope_file and not any(
                           p.startswith("_") for p in parts) and bool(parts)),
                       covered=node.lineno in covered_lines)
            out.append(m)
    return out


def priority(m, func_stats):
    k, s = func_stats.get((m.file, m.func), (0, 0))
    weak = s / max(1, k + s)
    score = 0.0
    score += 40 if m.named else 20 if m.in_scope else 0
    score += 25 if m.covered else -50
    score += 15 * weak
    score += {"boundary": 10, "compare": 10, "branch": 9, "arith": 8, "default": 8, "constant": 6, "return": 7,
              "exception": 6, "bool": 7, "unary": 6, "twin": 6, "statement": 5, "operand": 5, "loop": 5}.get(m.kind, 3)
    score -= 12 * m.shown
    if m.func.split(".")[-1].startswith("_"):
        score -= 5
    return score


class MutationRunner:
    """k private copies of the source root; one mutant per copy at a time."""

    def __init__(self, src_root, work, jobs, run_suite):
        self.src_root, self.run_suite, self.jobs = src_root, run_suite, max(1, jobs)
        self.copies = []
        for i in range(self.jobs):
            d = os.path.join(work, "mut%d" % i)
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(src_root, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            self.copies.append(d)
        self.trees = {}

    def _one(self, slot, m):
        root = self.copies[slot]
        path = os.path.join(root, m.file)
        with open(os.path.join(self.src_root, m.file), encoding="utf-8") as fh:
            original = fh.read()
        if m.file not in self.trees:
            self.trees[m.file] = ast.parse(original)
        try:
            new = ast.unparse(apply(self.trees[m.file], m.index, m.kind, m.variant))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new)
            status, failed = self.run_suite(root)
        finally:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(original)
        return status, failed

    def run(self, mutants, log=print):
        free = list(range(self.jobs))
        import threading
        lock = threading.Lock()

        def task(m):
            with lock:
                slot = free.pop()
            try:
                status, failed = self._one(slot, m)
            except Exception as e:
                status, failed = "error", set()
                log("[mut] %s error: %s" % (m.id, e))
            finally:
                with lock:
                    free.append(slot)
            m.status = status
            m.killers = sorted(failed)
            return m
        with concurrent.futures.ThreadPoolExecutor(self.jobs) as ex:
            list(ex.map(task, mutants))
