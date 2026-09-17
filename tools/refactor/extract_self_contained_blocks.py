"""Find and extract self-contained blocks from outsized functions, in bulk.

A block is a contiguous run of statements in any statement list of the
function (the body, or the body of an if/for/while/with/try/handler) that:
  * has no return/yield, and no break/continue that leaves the block,
  * defines no nested function/class and reads none defined outside it,
  * reads at most MAX_PARAMS names bound earlier in the function,
  * binds at most MAX_RETURNS names read after it, each bound by a direct,
    unconditional assignment inside the block (so the helper can return it),
  * uses no `await` unless the function is async (then the helper is async).
Extraction is a pure move: the block's lines become the helper's body, the
helper takes the params and returns the returns, and the call replaces the
block. Comments and formatting travel with the lines.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

MAX_PARAMS = 9
MAX_RETURNS = 2
MIN_LINES = 25
MAX_LINES = 400
MAX_SPAN = 80

ESCAPES = (ast.Return, ast.Yield, ast.YieldFrom)


def names(node, ctx):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


class _Events:
    """Name events in evaluation order, with comprehension and handler scopes.

    Emits ("load", name), ("store", name), and for except handlers the name
    is stored on entry and deleted on exit. Comprehension targets bind a
    scope of their own and are neither function stores nor free reads.
    """

    def __init__(self):
        self.events = []
        self.comp_scopes = []
        self.conditional = 0  # inside a body that may not run
        self.bound_stack = [set()]  # names definitely bound on the current path
        self.free = set()

    def _cond(self, stmts):
        self.conditional += 1
        self.bound_stack.append(set(self.bound_stack[-1]))
        for st in stmts:
            self.visit(st)
        self.bound_stack.pop()
        self.conditional -= 1

    def _bound_in_comp(self, name):
        return any(name in scope for scope in self.comp_scopes)

    def load(self, name):
        if not self._bound_in_comp(name):
            self.events.append(("load", name))
            if name not in self.bound_stack[-1]:
                self.free.add(name)

    def store(self, name):
        if self.comp_scopes:
            self.comp_scopes[-1].add(name)
        else:
            self.events.append(("maybe_store" if self.conditional else "store", name))
            self.bound_stack[-1].add(name)

    def delete(self, name):
        self.events.append(("delete", name))
        for scope in self.bound_stack:
            scope.discard(name)

    def visit(self, node):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self.load(node.id)
            elif isinstance(node.ctx, ast.Store):
                self.store(node.id)
            else:  # Del
                self.load(node.id)
            return
        if isinstance(node, (ast.Assign,)):
            self.visit(node.value)
            for t in node.targets:
                self.visit(t)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.visit(node.value)
            if node.value is not None or isinstance(node.target, ast.Name):
                self.visit(node.target)
            return
        if isinstance(node, ast.AugAssign):
            # target is read, then written
            if isinstance(node.target, ast.Name):
                self.load(node.target.id)
            else:
                self.visit(node.target)
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self.store(node.target.id)
            return
        if isinstance(node, ast.NamedExpr):
            self.visit(node.value)
            self.visit(node.target)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self.visit(node.iter)
            self.conditional += 1
            self.visit(node.target)
            self.conditional -= 1
            self._cond(node.body)
            self._cond(node.orelse)
            return
        if isinstance(node, ast.While):
            self.visit(node.test)
            self._cond(node.body)
            self._cond(node.orelse)
            return
        if isinstance(node, ast.If):
            self.visit(node.test)
            self._cond(node.body)
            self._cond(node.orelse)
            return
        if isinstance(node, ast.Match):
            self.visit(node.subject)
            for case in node.cases:
                self._cond([case.pattern] + ([case.guard] if case.guard else []) + case.body)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.visit(item.context_expr)
                if item.optional_vars is not None:
                    self.visit(item.optional_vars)
            for st in node.body: self.visit(st)
            return
        if isinstance(node, COMPREHENSIONS):
            self.comp_scopes.append(set())
            gens = node.generators
            # the first iterable is evaluated in the enclosing scope
            for k, g in enumerate(gens):
                if k == 0:
                    self.comp_scopes.pop()
                    self.visit(g.iter)
                    self.comp_scopes.append(set())
                else:
                    self.visit(g.iter)
                self.visit(g.target)
                for cond in g.ifs: self.visit(cond)
            if isinstance(node, ast.DictComp):
                self.visit(node.key); self.visit(node.value)
            else:
                self.visit(node.elt)
            self.comp_scopes.pop()
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                self.store((a.asname or a.name).split(".")[0])
            return
        if isinstance(node, SCOPES):
            # a nested scope: its free names are reads of ours; its own
            # bindings are not ours. Defaults/decorators evaluate here.
            if isinstance(node, ast.Lambda):
                for d in node.args.defaults + node.args.kw_defaults:
                    if d is not None: self.visit(d)
                inner = _Events(); inner.visit(node.body)
                own = {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
                for kind, name in inner.events:
                    if kind == "load" and name not in own: self.load(name)
                return
            for d in getattr(node, "decorator_list", []): self.visit(d)
            if hasattr(node, "args"):
                for d in node.args.defaults + node.args.kw_defaults:
                    if d is not None: self.visit(d)
            inner = _Events()
            for st in node.body: inner.visit(st)
            own = set()
            if hasattr(node, "args"):
                own = {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
            bound = set(own)
            for kind, name in inner.events:
                if kind == "load" and name not in bound: self.load(name)
                elif kind == "store": bound.add(name)
            self.store(node.name)
            return
        if isinstance(node, ast.Try):
            self._cond(node.body)
            for h in node.handlers:
                if h.type is not None: self.visit(h.type)
                self.conditional += 1
                if h.name: self.store(h.name)
                for st in h.body: self.visit(st)
                if h.name: self.delete(h.name)
                self.conditional -= 1
            self._cond(node.orelse)
            for st in node.finalbody: self.visit(st)
            return
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return
        for child in ast.iter_child_nodes(node):
            self.visit(child)


def events_of(st):
    ev = _Events(); ev.visit(st); return ev.events


def stores(st):
    """Names this statement may leave bound afterwards."""
    out = set()
    for kind, name in events_of(st):
        if kind in ("store", "maybe_store"): out.add(name)
        elif kind == "delete": out.discard(name)
    return out


def free_reads(st):
    """Names read in ``st`` that no binding on the same path in ``st`` precedes.

    A binding inside an if/for/while/try body covers the reads that follow it
    in that body; it does not cover reads after the body, which may still be
    of the outer value.
    """
    ev = _Events(); ev.visit(st)
    return set(ev.free)


def direct_stores(st):
    """Names bound unconditionally by this statement at its top level."""
    if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = st.targets if isinstance(st, ast.Assign) else [st.target]
        out = set()
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
        return out
    if isinstance(st, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in st.names}
    return set()


def escapes(st):
    if isinstance(st, (ast.Break, ast.Continue)):
        return True
    for n in ast.walk(st):
        if isinstance(n, ESCAPES):
            return True
    def walk(node, depth):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                for c in child.body:
                    if walk(c, depth + 1): return True
                for c in child.orelse:
                    if walk(c, depth): return True
                for c in ast.iter_child_nodes(child):
                    if c not in child.body and c not in child.orelse and walk(c, depth): return True
            elif isinstance(child, (ast.Break, ast.Continue)):
                if depth == 0: return True
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            elif walk(child, depth):
                return True
        return False
    return walk(st, 0)


def has_await(st):
    return any(isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for n in ast.walk(st))


def defines_nested(st):
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(st))


def statement_lists(fn):
    """Every statement list inside fn, with the statements' nesting parents."""
    out = []
    def visit(stmts):
        out.append(stmts)
        for st in stmts:
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(st, field, None)
                if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    visit(sub)
            for h in getattr(st, "handlers", []) or []:
                visit(h.body)
            for case in getattr(st, "cases", []) or []:
                visit(case.body)
    visit(fn.body)
    return out


def find_function(tree, target):
    cls_name, fn_name = (target.split(".", 1) if "." in target else (None, target))
    for node in ast.walk(tree):
        if cls_name and isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == fn_name:
                    return sub, node
        elif not cls_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return node, None
    return None, None


def analyse(src, target):
    tree = ast.parse(src)
    fn, cls = find_function(tree, target)
    if fn is None:
        return None, None, []
    is_async = isinstance(fn, ast.AsyncFunctionDef)
    args = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    if fn.args.vararg: args.add(fn.args.vararg.arg)
    if fn.args.kwarg: args.add(fn.args.kwarg.arg)
    top_stmts = list(fn.body)
    nested = {n.name for n in ast.walk(fn) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn}
    fn_locals = set(args)
    for st in fn.body:
        for kind, name in events_of(st):
            if kind in ("store", "maybe_store"):
                fn_locals.add(name)
    # global/nonlocal declared names are off limits
    declared = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Global, ast.Nonlocal)): declared |= set(n.names)
    # names read inside nested functions are live everywhere (closure cells)
    captured = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not fn:
            captured |= names(n, ast.Load)
    # loops: a name stored anywhere in a loop body is bound for a block inside it
    loop_bodies = []
    for n in ast.walk(fn):
        if isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
            bound = set().union(*(stores(s) for s in n.body)) if n.body else set()
            # what a later iteration may read of what this one bound: the
            # body's free reads taken as one sequence, plus the loop test
            ev = _Events()
            for s_ in n.body:
                ev.visit(s_)
            reads = set(ev.free)
            if isinstance(n, ast.While):
                reads |= names(n.test, ast.Load)
            loop_bodies.append((n.lineno, n.end_lineno, bound, reads))
    candidates = []
    # per-statement facts, computed once (the block search is quadratic)
    facts = {}
    def fact(st):
        f = facts.get(id(st))
        if f is None:
            f = (stores(st), free_reads(st), escapes(st) or defines_nested(st), has_await(st))
            facts[id(st)] = f
        return f
    stmt_loads_cache = {}
    for stmts in statement_lists(fn):
        n = len(stmts)
        S = [fact(st)[0] for st in stmts]
        Lo = [fact(st)[1] for st in stmts]
        for i in range(n):
            if fact(stmts[i])[2]:
                continue
            for j in range(i, min(n, i + MAX_SPAN)):
                st_j = stmts[j]
                if fact(st_j)[2]:
                    break
                if not is_async and fact(st_j)[3]:
                    break
                block = stmts[i:j + 1]
                lines = block[-1].end_lineno - block[0].lineno + 1
                if lines < MIN_LINES:
                    continue
                if lines > MAX_LINES:
                    break
                bstores = set().union(*S[i:j + 1])
                bloads = set().union(*Lo[i:j + 1])
                if (bloads | bstores) & (nested | declared):
                    continue
                start_line, end_line = block[0].lineno, block[-1].end_lineno
                before_stores = set(args)
                after_loads = set()
                for st in top_stmts:
                    if st.end_lineno < start_line:
                        before_stores |= fact(st)[0]
                    elif st.lineno > end_line:
                        after_loads |= fact(st)[1]
                    else:
                        # a statement that spans the block: its parts before/after,
                        # and what the spanning constructs themselves bind on the way in
                        for sub in ast.walk(st):
                            if isinstance(sub, ast.stmt) and sub is not st:
                                if sub.end_lineno < start_line:
                                    before_stores |= fact(sub)[0]
                                elif sub.lineno > end_line:
                                    after_loads |= fact(sub)[1]
                            spans = (
                                isinstance(sub, (ast.stmt, ast.ExceptHandler))
                                and sub.lineno <= start_line
                                and sub.end_lineno >= end_line
                                and not (sub.lineno >= start_line and sub.end_lineno <= end_line)
                            )
                            if not spans:
                                continue
                            if isinstance(sub, ast.ExceptHandler) and sub.name:
                                before_stores.add(sub.name)
                            elif isinstance(sub, (ast.For, ast.AsyncFor)):
                                before_stores |= names(sub.target, ast.Store)
                            elif isinstance(sub, (ast.With, ast.AsyncWith)):
                                for item in sub.items:
                                    if item.optional_vars is not None:
                                        before_stores |= names(item.optional_vars, ast.Store)
                            elif isinstance(sub, ast.Try):
                                pass
                # names read in the block that the block may not have bound yet
                params = set()
                bound_so_far = set()
                for k in range(i, j + 1):
                    params |= {v for v in Lo[k] if v in fn_locals and v not in bound_so_far}
                    bound_so_far |= direct_stores(stmts[k])  # only an unconditional binding covers later reads
                # a read of a name the block binds later in its own text but reads first
                # is a read of the outer binding -> param (handled above); but a name only
                # ever bound in the block and never before it cannot be a param
                loop_bound = set()
                loop_carried = set()
                for lo, hi, bound, reads in loop_bodies:
                    if lo <= start_line and end_line <= hi:
                        loop_bound |= bound
                        loop_carried |= bound & reads
                unbound_reads = {v for v in params if v not in before_stores and v not in loop_bound}
                if unbound_reads:
                    continue  # the original may raise there; do not move it
                params = {v for v in params if v in before_stores or v in loop_bound}
                returns = {v for v in bstores if v in after_loads or v in captured or v in loop_carried}
                # a block followed by a return leaves nothing for the code
                # after it but what the return reads and what a closure holds
                # (a raise may be caught in this function; a break continues
                # after the loop)
                nxt = stmts[j + 1] if j + 1 < n else None
                if isinstance(nxt, ast.Return):
                    returns = {v for v in bstores if v in captured or v in free_reads(nxt)}
                direct = set().union(*(direct_stores(st) for st in block))
                # a returned name the block binds only on some paths must be
                # bound for certain before the block (an argument, or a plain
                # assignment in an earlier top-level statement of the function),
                # so the caller can pass it in and take it back whatever ran
                definite_before = set(args)
                for st in top_stmts:
                    if st.end_lineno < start_line:
                        definite_before |= direct_stores(st)
                if any(v not in direct and v not in definite_before for v in returns):
                    continue
                params |= {v for v in returns if v not in direct}
                if len(params) > MAX_PARAMS or len(returns) > MAX_RETURNS:
                    continue
                # a block ending in an assignment nobody reads (dead in the
                # original, unflagged there because the name is read earlier)
                # would be flagged in the helper: leave that statement behind
                dead = False
                for k in range(i, j + 1):
                    st_k = block[k - i]
                    if isinstance(st_k, (ast.Assign, ast.AnnAssign)):
                        targets = direct_stores(st_k)
                        read_later_in_block = set().union(*(names(b, ast.Load) for b in block[k - i + 1:])) if k < j else set()
                        read_in_own_rhs = set()
                        if targets and not (targets & returns) and not (targets & after_loads) and not (targets & read_later_in_block) and not (targets & captured):
                            dead = True
                            break
                if dead:
                    continue
                candidates.append(dict(lines=lines, start=start_line, end=end_line, params=sorted(params),
                                       returns=sorted(returns), awaits=any(fact(st)[3] for st in block),
                                       first=block[0]))
    # a name bound by a function-local import (and by nothing else) is
    # re-imported in the helper rather than passed: the same lookup at the
    # same time, and no class or constant travels as an argument
    imports = {}
    guarded = set()   # names whose import sits under a try: not re-importable
    assigned_elsewhere = set()
    parents = {}
    for parent in ast.walk(fn):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    def under_try(node):
        while node in parents and parents[node] is not fn:
            node = parents[node]
            if isinstance(node, ast.Try):
                return True
        return False
    for n in ast.walk(fn):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound = (a.asname or a.name).split(".")[0]
                if under_try(n):
                    guarded.add(bound)
                if isinstance(n, ast.ImportFrom):
                    text = f"from {'.' * n.level}{n.module or ''} import {a.name}" + (f" as {a.asname}" if a.asname else "")
                else:
                    text = f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
                imports.setdefault(bound, set()).add((n.lineno, text))
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            assigned_elsewhere.add(n.id)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            assigned_elsewhere |= names(n.target, ast.Store)
    import_bound_only = {nm for nm in imports if nm not in assigned_elsewhere}
    candidates = [c for c in candidates if not (set(c["returns"]) & import_bound_only)]
    # a guarded import (under a try) cannot be re-imported and must not travel
    # as an argument either: leave such blocks alone
    candidates = [c for c in candidates if not (set(c["params"]) & guarded & import_bound_only)]
    for c in candidates:
        reimport = []
        keep = []
        for pname in c["params"]:
            texts = {t for _l, t in imports.get(pname, ())}
            if pname not in assigned_elsewhere and len(texts) == 1 and pname not in c["returns"]:
                reimport.append(texts.pop())
            else:
                keep.append(pname)
        c["params"] = keep
        c["reimport"] = _import_block(reimport)
    candidates.sort(key=lambda c: -c["lines"])
    chosen, taken = [], []
    for c in candidates:
        if any(not (c["end"] < a or c["start"] > b) for a, b in taken):
            continue
        chosen.append(c)
        taken.append((c["start"], c["end"]))
    return fn, cls, sorted(chosen, key=lambda c: c["start"])


def _import_block(texts):
    """Single-name import lines combined the way isort writes them."""
    plain = sorted({t for t in texts if t.startswith("import ")})
    froms = {}
    for t in texts:
        if t.startswith("from "):
            mod, name = t[5:].split(" import ", 1)
            froms.setdefault(mod, set()).add(name)
    lines = list(plain)
    for mod in sorted(froms):
        lines.append(f"from {mod} import " + ", ".join(sorted(froms[mod])))
    return lines


def slug_for(lines, c, fn_name, k):
    # leading comment above the block
    j = c["start"] - 2
    words = []
    while j >= 0 and lines[j].strip().startswith("#"):
        words.insert(0, lines[j].strip("#— ─-\n "))
        j -= 1
    text = " ".join(words)
    m = re.findall(r"[A-Za-z][A-Za-z0-9]+", text)
    stop = {"the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "is", "it", "this", "that", "with", "as", "by", "at", "from", "not", "be", "was", "are", "we", "so", "if", "its", "which", "when", "one", "into", "than", "then", "also", "only", "no", "but", "every", "each", "any", "all", "live", "measured", "here"}
    m = [w.lower() for w in m if w.lower() not in stop][:3]
    if not m:
        first = c["first"]
        d = direct_stores(first)
        if d:
            m = sorted(d)[:1]
    slug = "_".join(m) if m else f"part_{k}"
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    return f"_{fn_name.strip('_')}_{slug}"


def apply(path: Path, target: str, dry=False):
    src = path.read_text()
    fn, cls, chosen = analyse(src, target)
    if fn is None or not chosen:
        return 0, []
    lines = src.splitlines(keepends=True)
    is_method = bool(fn.args.args) and fn.args.args[0].arg in ("self", "cls")
    self_name = fn.args.args[0].arg if is_method else None
    fn_indent = " " * fn.col_offset
    body_indent = fn_indent + "    "
    helpers = []
    edits = []  # (start_idx, end_idx, replacement_lines)
    used = set()
    for k, c in enumerate(chosen, start=1):
        name = slug_for(lines, c, fn.name, k)
        while name in used or name in src:
            name = f"{name}_{k}"
        used.add(name)
        block_lines = lines[c["start"] - 1 : c["end"]]
        # include leading comment lines
        j = c["start"] - 2
        while j >= 0 and lines[j].strip().startswith("#") and lines[j].startswith(" " * (len(lines[c['start']-1]) - len(lines[c['start']-1].lstrip()))):
            j -= 1
        lead = lines[j + 1 : c["start"] - 1]
        block_indent = len(block_lines[0]) - len(block_lines[0].lstrip())
        shift = block_indent - len(body_indent)
        def reindent(l):
            if l.strip() == "":
                return "\n"
            if shift > 0:
                return l[shift:] if l.startswith(" " * shift) else l
            return " " * (-shift) + l
        body = [f"{body_indent}{line}\n" for line in c.get("reimport", [])] + [reindent(l) for l in lead + block_lines]
        uses_self = self_name in c["params"]
        params = [p for p in c["params"] if p != self_name]
        in_class = cls is not None
        sig_parts = ([self_name] if (is_method and uses_self) else []) + params
        sig = ", ".join(sig_parts)
        ret = c["returns"]
        ret_line = f"{body_indent}return {', '.join(ret)}\n" if ret else ""
        kw = "async def" if c["awaits"] else "def"
        is_classmethod = self_name == "cls" or any(
            getattr(d, "id", getattr(d, "attr", "")) == "classmethod" for d in fn.decorator_list
        )
        if in_class and not uses_self:
            decorator = f"{fn_indent}@staticmethod\n"
        elif in_class and is_classmethod:
            decorator = f"{fn_indent}@classmethod\n"
        else:
            decorator = ""
        helper = decorator + f"{fn_indent}{kw} {name}({sig}):\n" + "".join(body) + ret_line + "\n"
        helpers.append(helper)
        if in_class:
            owner = self_name if is_method else cls.name
            call_target = f"{owner}.{name}"
        else:
            call_target = name
        call = f"{call_target}({', '.join(params)})"
        if c["awaits"]:
            call = f"await {call}"
        lhs = (", ".join(ret) + " = ") if ret else ""
        call_line = " " * block_indent + lhs + call + "\n"
        edits.append((j + 1, c["end"], [call_line]))
    # apply edits bottom-up
    for start_idx, end_idx, repl in sorted(edits, key=lambda e: -e[0]):
        lines[start_idx:end_idx] = repl
    # insert helpers before the function (same indentation level)
    fn_start = fn.lineno - 1
    # decorators
    if fn.decorator_list:
        fn_start = fn.decorator_list[0].lineno - 1
    lines[fn_start:fn_start] = ["".join(helpers)]
    out = "".join(lines)
    ast.parse(out)  # must parse
    reimported = set()
    for c in chosen:
        reimported |= {t.split(" import ")[-1].split(" as ")[-1].strip() for t in c.get("reimport", [])}
    if reimported:
        out = _prune_unused_local_imports(out, target, reimported)
    if not dry:
        path.write_text(out)
    return sum(c["lines"] for c in chosen), chosen


def _prune_unused_local_imports(src: str, target: str, candidates: set[str]) -> str:
    """Drop function-local imports of ``candidates`` the function no longer reads."""
    tree = ast.parse(src)
    fn, _cls = find_function(tree, target)
    if fn is None:
        return src
    parents = {}
    for parent in ast.walk(fn):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    def region_of(node):
        # the innermost compound statement holding this import, or the function
        node = parents.get(node, fn)
        while node is not fn and not isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.ExceptHandler)):
            node = parents.get(node, fn)
        return node
    lines = src.splitlines(keepends=True)
    edits = []
    for n in ast.walk(fn):
        if not isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        region = region_of(n)
        read_here = {m.id for m in ast.walk(region) if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load)}
        keep = [a for a in n.names if (a.asname or a.name).split(".")[0] not in candidates or (a.asname or a.name).split(".")[0] in read_here]
        if len(keep) == len(n.names):
            continue
        indent = " " * n.col_offset
        if not keep:
            repl = []
        else:
            copy = ast.ImportFrom(module=n.module, names=keep, level=n.level) if isinstance(n, ast.ImportFrom) else ast.Import(names=keep)
            text = ast.unparse(copy)
            repl = [indent + text + "\n"]
        edits.append((n.lineno - 1, n.end_lineno, repl))
    for start, end, repl in sorted(edits, key=lambda e: -e[0]):
        lines[start:end] = repl
    out = "".join(lines)
    ast.parse(out)
    return out


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    specs = [a for a in sys.argv[1:] if a != "--dry"]
    for spec in specs:
        path, target = spec.split("::")
        src = Path(path).read_text()
        fn, cls, chosen = analyse(src, target)
        if fn is None:
            print("MISSING", spec); continue
        total = fn.end_lineno - fn.lineno + 1
        saved = sum(c["lines"] for c in chosen)
        print(f"{spec}: {total} -> ~{total - saved + len(chosen)} ({len(chosen)} blocks, {saved} lines)")
        for c in chosen:
            print(f"   lines {c['start']}-{c['end']} ({c['lines']}) params={c['params']} returns={c['returns']}{' async' if c['awaits'] else ''}")
        if not dry and chosen:
            apply(Path(path), target)
