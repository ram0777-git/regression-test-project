"""Task statement -> structured Contract (deterministic parse + validated LLM merge)."""
from __future__ import annotations

import dataclasses
import re

OUT_WORDS = ("not part of", "out of scope", "not in scope", "excluded", "non-goal", "unstable",
             "need not", "do not test", "don't test", "not contractual", "unspecified", "may change")


@dataclasses.dataclass
class Contract:
    statement: str = ""
    tests_dir: str = "tests"
    modules: list = dataclasses.field(default_factory=list)
    names: list = dataclasses.field(default_factory=list)          # functions/classes/methods in scope
    in_scope: list = dataclasses.field(default_factory=list)
    out_of_scope: list = dataclasses.field(default_factory=list)
    hints: dict = dataclasses.field(default_factory=dict)           # edge_cases, invalid_inputs, ...
    allowed_imports: list = dataclasses.field(default_factory=list)
    hygiene: list = dataclasses.field(default_factory=list)
    messages: bool = False            # exact exception messages contractual?
    exception_classes: bool = True    # exception *types* contractual?
    ordering: bool = True
    stdout: bool = False
    state: bool = True                # public attributes of instances
    randomness: bool = False
    timestamps: bool = False
    identity: bool = False
    suite_limit: float = 60.0

    def policy(self) -> dict:
        return {k: getattr(self, k) for k in ("messages", "exception_classes", "ordering", "stdout", "state",
                                              "randomness", "timestamps", "identity")}

    def summary(self) -> str:
        return ("scope names: %s\nmodules: %s\nin scope: %s\nout of scope: %s\nassertion policy: %s" % (
            ", ".join(self.names) or "(whole package)", ", ".join(self.modules) or "-",
            "; ".join(self.in_scope[:12]) or "-", "; ".join(self.out_of_scope[:12]) or "-", self.policy()))


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*[-*]\s+|\n{2,}", text) if s.strip()]


def split_scope(statement: str):
    """Returns (in_text, out_text): heading-delimited 'not part of the contract' sections + out sentences."""
    keep, out, skipping = [], [], False
    for line in statement.split("\n"):
        if re.match(r"\s*(#+\s|[A-Z][\w /-]{2,60}:\s*$)", line):
            skipping = any(w in line.lower() for w in OUT_WORDS)
        (out if skipping else keep).append(line)
    in_text = "\n".join(keep)
    moved = [s for s in _sentences(in_text) if any(w in s.lower() for w in OUT_WORDS)]
    return in_text, "\n".join(out + moved)


def parse_statement(statement: str) -> Contract:
    c = Contract(statement=statement)
    for rx in (r"[Aa]dd(?:s)? (?:new )?files only under `([^`]+)`", r"only under `([^`]+)`",
               r"[Tt]est(?:s)? (?:directory|dir|folder)[^`\n]*`([^`]+)`", r"under `([^`]*tests?[^`]*)`"):
        m = re.search(rx, statement)
        if m:
            c.tests_dir = m.group(1).strip().strip("/") or "tests"
            break
    in_text, out_text = split_scope(statement)
    c.out_of_scope = _sentences(out_text)[:40]
    c.in_scope = [s for s in _sentences(in_text) if len(s) < 400][:60]
    names, mods = [], []
    for tok in re.findall(r"`([A-Za-z_][\w.]*)(?:\(\))?`", in_text):
        if "/" in tok or tok.endswith((".py", ".md", ".rst", ".txt")):
            continue
        parts = tok.split(".")
        if len(parts) > 1:
            mods.append(tok)
        names += [p for p in parts if p and not p.startswith("_")]
    c.names = list(dict.fromkeys(n for n in names if n not in ("pytest", "tests", "test", "None", "True", "False")))
    c.modules = list(dict.fromkeys(mods))
    for m in re.finditer(r"[Ii]mport(?:ed)? (?:only )?from `([A-Za-z_][\w.]*)`", statement):
        c.allowed_imports.append(m.group(1))
    low_in, low_out = in_text.lower(), out_text.lower()
    if re.search(r"(exact|error|exception) (message|text)s?[^.]{0,40}(is|are) (part of|contractual|pinned)", low_in) \
            or "message text is part of" in low_in:
        c.messages = True
    if "message" in low_out or "wording" in low_out:
        c.messages = False
    if re.search(r"exception (class|type)s?[^.]{0,30}(not part|unspecified)|which exception", low_out):
        c.exception_classes = False
    if re.search(r"\border(ing)?\b", low_out):
        c.ordering = False
    if re.search(r"\b(stdout|printed|prints|output written)\b", low_in) and "stdout" not in low_out \
            and "print" not in low_out:
        c.stdout = True
    if re.search(r"internal state|attributes?", low_out):
        c.state = False
    c.randomness = bool(re.search(r"random[^.]{0,40}(is|are) part of", low_in))
    c.timestamps = bool(re.search(r"timestamp[^.]{0,40}(is|are) part of", low_in))
    c.identity = bool(re.search(r"\bidentity\b[^.]{0,40}(is|are) part of", low_in))
    m = re.search(r"(?:finish|run|complete)s? in under (\d+(?:\.\d+)?) seconds", statement)
    if m:
        c.suite_limit = float(m.group(1))
    c.hygiene = [s for s in _sentences(statement) if re.search(r"\b(must|never|do not|don't|only)\b", s, re.I)][:20]
    return c


ANALYSIS_KEYS = ("in_scope_behaviors", "out_of_scope_behaviors", "edge_cases", "invalid_inputs",
                 "boundary_conditions", "stateful_sequences", "interaction_cases", "hygiene_rules",
                 "likely_bug_patterns", "documentation_sources")


def merge_llm(c: Contract, data: dict, known_names: set) -> list:
    """Merge the LLM analysis; names are accepted only if they exist in the repository. Returns notes."""
    notes = []
    if not isinstance(data, dict):
        return ["analysis was not a JSON object"]
    scope = data.get("scope") if isinstance(data.get("scope"), dict) else {}
    for key in ("functions", "classes", "methods"):
        for n in scope.get(key) or []:
            if not isinstance(n, str):
                continue
            leaf = n.split(".")[-1].strip("()")
            if leaf in known_names and leaf not in c.names:
                c.names.append(leaf)
            elif leaf not in known_names:
                notes.append("ignored invented name %r" % n)
    for key in ANALYSIS_KEYS:
        vals = [str(v)[:300] for v in (data.get(key) or []) if isinstance(v, (str, int, float))][:25]
        if key == "out_of_scope_behaviors":
            c.out_of_scope += [v for v in vals if v not in c.out_of_scope]
        elif key == "in_scope_behaviors":
            c.in_scope += [v for v in vals if v not in c.in_scope]
        elif vals:
            c.hints[key] = vals
    pol = data.get("assertion_policy") if isinstance(data.get("assertion_policy"), dict) else {}
    # The LLM may only make the policy *stricter* (fewer assertions) unless the statement itself says so.
    for k in ("messages", "stdout", "state", "ordering", "exception_classes"):
        if pol.get(k) is False and getattr(c, k):
            setattr(c, k, False)
            notes.append("policy %s -> False (LLM analysis)" % k)
    return notes