"""Case files -> ordinary, self-contained pytest modules with literal expected values."""
from __future__ import annotations

import ast
import copy
import re
import sys

SAME_SRC = '''
def _same_value(actual, expected):
    """Exact type-aware comparison; floats compared with a tiny tolerance."""
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
        return math.isclose(actual, expected, rel_tol=1e-09, abs_tol=1e-12)
    if isinstance(expected, complex):
        return isinstance(actual, complex) and _same_value(actual.real, expected.real) and _same_value(actual.imag, expected.imag)
    if isinstance(expected, (str, bytes)):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (list, tuple)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(_same_value(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and len(actual) == len(expected) and all(k in actual and _same_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (set, frozenset)):
        return isinstance(actual, (set, frozenset)) and actual == expected
    return actual == expected
'''


def test_file_name(module: str) -> str:
    return "test_rt_" + (module[len("cases_"):] if module.startswith("cases_") else module) + ".py"


def test_name(case: str) -> str:
    return "test_" + case[len("case_"):]


def _case_to_test(fn, rec, contract, pins_classes):
    stmts = [copy.deepcopy(s) for s in fn.body]
    doc = None
    if len(stmts) > 1 and isinstance(stmts[0], ast.Expr) and isinstance(getattr(stmts[0], "value", None), ast.Constant) \
            and isinstance(stmts[0].value.value, str):
        doc = stmts.pop(0)
    ret = stmts.pop()
    var = "actual"
    while any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(fn)):
        var += "_"
    if rec["status"] == "raises":
        cls = rec.get("cls") if (contract.exception_classes or pins_classes) else "Exception"
        kws = []
        if contract.messages and rec.get("msg"):
            kws = [ast.keyword("match", ast.Call(ast.Attribute(ast.Name("re", ast.Load()), "escape", ast.Load()),
                                                 [ast.Constant(rec["msg"])], []))]
        body = [ast.With(items=[ast.withitem(ast.Call(
            ast.Attribute(ast.Name("pytest", ast.Load()), "raises", ast.Load()),
            [ast.parse(cls or "Exception", mode="eval").body], kws))], body=stmts + [ast.Expr(ret.value)])]
    else:
        body = stmts + [ast.Assign([ast.Name(var, ast.Store())], ret.value),
                        ast.Assert(ast.Call(ast.Name("_same_value", ast.Load()),
                                            [ast.Name(var, ast.Load()), ast.parse(rec["src"], mode="eval").body], []))]
    if doc is not None:
        body.insert(0, doc)
    new = ast.FunctionDef(name=test_name(fn.name), args=copy.deepcopy(fn.args), body=body, decorator_list=[],
                          returns=None, type_comment=None)
    if sys.version_info >= (3, 12):
        new.type_params = []
    return new


def module_source(case_source: str, module: str, records: dict, keep: set, contract) -> str | None:
    """records: case name -> record. Returns None when no test survives."""
    tree = ast.parse(case_source)
    pins = bool(re.search(r"(?m)^EXCEPTION_CLASSES\s*=\s*True\b", case_source))
    header = [ast.Expr(ast.Constant("Regression tests (%s), generated from the public contract." %
                                    module.replace("cases_", "").replace("_", " "))),
              ast.Import([ast.alias("math")]), ast.Import([ast.alias("re")]), ast.Import([ast.alias("pytest")])]
    body, n = [], 0
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EXCEPTION_CLASSES"
                                                for t in node.targets):
            continue
        if isinstance(node, ast.FunctionDef) and node.name.startswith("case_"):
            rec = records.get(node.name)
            if rec is None or node.name not in keep:
                continue
            if rec["status"] == "raises" and "." in (rec.get("cls") or ""):
                header.append(ast.Import([ast.alias(rec["cls"].rpartition(".")[0])]))
            body.append(_case_to_test(node, rec, contract, pins))
            n += 1
            continue
        body.append(node)
    if not n:
        return None
    seen, uniq = set(), []
    for h in header:
        k = ast.dump(h)
        if k not in seen:
            seen.add(k)
            uniq.append(h)
    mod = ast.Module(body=uniq + body + ast.parse(SAME_SRC).body, type_ignores=[])
    ast.fix_missing_locations(mod)
    text = ast.unparse(mod) + "\n"
    compile(text, test_file_name(module), "exec")
    return text
