"""AssetLoader (CMP-38) — read-only startup loader for the builtin registries.

Parses ``agent/assets/{words,relations,roles}.tsv`` into the v14 data model
(``agent.core.model``) and resolves each Word's ``impl_key`` against the
core dispatch registry (``agent.core.registry``). Read-only and deterministic
(no fine-tune at runtime; realizes NFR-5).

Boundary (Clean Architecture): this adapter loads *data* and *wires* dispatch.
It owns no semantics — operator meaning lives in ``gr-arc-3-operators.md`` /
the goal interpreter, detector meaning in DetectFeatures. ``recognized_by`` role
predicates are parsed into ``Relation`` trees here (ASSET-A: the ``has`` operator
unifies role recognition with the Goal Relation language) but evaluated later by
AnalogizeRoles in the role-assignment env.

The dispatch table only contains impl_keys whose callable is registered at load
time; call :meth:`AssetLoader.assert_complete` once all impl modules are imported
to FK-validate that every impl_key has a callable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from agent.core import registry
from agent.core.model import (
    GoalKind,
    GoalPattern,
    Lexicon,
    Operand,
    Relation,
    Role,
    Solver,
    Word,
)

_DEFAULT_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Category -> the impl_key prefix its Words must carry (or "" = no dispatch).
# Validated at load so a mis-categorised row fails loud rather than silently
# dropping out of the dispatch table.
_CATEGORY_PREFIX = {
    "axis": "",          # category-Words (color/shape/behavior axes) — no callable
    "color": "feat_",
    "shape": "feat_",
    "behavior": "feat_",
    "scale": "feat_",
    "orientation": "feat_",  # the orientation axis Word carries feat_orient
    "logical": "op_",
    "quantifier": "op_",
    "relation": "rel_",
    "transform": "xf_",
}


# --------------------------------------------------------------------------- #
# recognized_by predicate parser:  IDENT | IDENT '(' arg (',' arg)* ')'
# --------------------------------------------------------------------------- #

_IDENT = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"


def _tokenize(text: str) -> List[str]:
    toks: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch in "(),":
            toks.append(ch)
            i += 1
        elif ch in _IDENT:
            j = i
            while j < n and text[j] in _IDENT:
                j += 1
            toks.append(text[i:j])
            i = j
        else:
            raise ValueError(f"unexpected char {ch!r} in predicate {text!r}")
    return toks


def parse_predicate(text: str) -> Operand:
    """Parse a recognized_by expression into a Relation tree (or bare string).

    ``has(controllable)`` -> Relation('has', ['controllable']);
    ``or(has(marked), inside(self, box))`` -> nested Relations; a bare ``self``
    -> the string 'self'. Operators become ``Relation.operator_word_id``; bare
    identifiers become string leaves (Role label / Word id), resolved later.
    """
    toks = _tokenize(text)
    pos = 0

    def expr() -> Operand:
        nonlocal pos
        if pos >= len(toks):
            raise ValueError(f"unexpected end of predicate {text!r}")
        head = toks[pos]
        if head in "(),":
            raise ValueError(f"expected identifier in {text!r}, got {head!r}")
        pos += 1
        if pos < len(toks) and toks[pos] == "(":
            pos += 1  # consume '('
            args: List[Operand] = []
            if pos >= len(toks):
                raise ValueError(f"missing ')' in predicate {text!r}")
            if toks[pos] != ")":
                args.append(expr())
                while pos < len(toks) and toks[pos] == ",":
                    pos += 1
                    args.append(expr())
            if pos >= len(toks) or toks[pos] != ")":
                raise ValueError(f"missing ')' in predicate {text!r}")
            pos += 1  # consume ')'
            return Relation(operator_word_id=head, operands=args, origin="builtin")
        return head  # bare leaf

    tree = expr()
    if pos != len(toks):
        raise ValueError(f"trailing tokens in predicate {text!r}: {toks[pos:]}")
    return tree


# --------------------------------------------------------------------------- #
# Loaded bundle
# --------------------------------------------------------------------------- #

@dataclass
class LoadedAssets:
    """The frozen result of a load: vocabulary, roles, and wired dispatch."""

    lexicon: Lexicon
    roles: Dict[str, Role]
    dispatch: Dict[str, Callable] = field(default_factory=dict)
    goal_patterns: Tuple[GoalPattern, ...] = ()
    goal_kinds: Tuple[GoalKind, ...] = ()
    solvers: Tuple[Solver, ...] = ()

    def role(self, label: str) -> Role:
        return self.roles[label]


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

def _read_tsv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip("\n")]
    header, body = rows[0], rows[1:]
    return header, body


def _row_dict(header: List[str], row: List[str], path: str) -> Dict[str, str]:
    if len(row) != len(header):
        raise ValueError(
            f"{path}: row has {len(row)} fields, expected {len(header)}: {row}"
        )
    return dict(zip(header, row))


def _parse_params(raw: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for tok in raw.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            params[k.strip()] = v.strip()
        else:
            params[tok] = ""  # flag-style param
    return params


class AssetLoader:
    """Loads the builtin TSV registries into a :class:`LoadedAssets` bundle."""

    def __init__(self, assets_dir: str = _DEFAULT_ASSETS_DIR) -> None:
        self.assets_dir = assets_dir

    # -- public API -------------------------------------------------------- #

    def load(self) -> LoadedAssets:
        lexicon = Lexicon(words=self._load_words(), relations=self._load_relations())
        self._validate_operator_fks(lexicon)
        roles = self._load_roles(lexicon)
        goal_patterns = self._load_goal_patterns(lexicon)
        goal_kinds = self._load_goal_kinds()
        solvers = self._load_solvers()
        self._validate_solver_fks(goal_patterns, goal_kinds, solvers)
        dispatch = {
            w.impl_key: registry.resolve(w.impl_key)
            for w in lexicon.words
            if w.impl_key and registry.is_registered(w.impl_key)
        }
        return LoadedAssets(
            lexicon=lexicon,
            roles=roles,
            dispatch=dispatch,
            goal_patterns=goal_patterns,
            goal_kinds=goal_kinds,
            solvers=solvers,
        )

    def assert_complete(self, loaded: LoadedAssets) -> None:
        """Raise if any Word's impl_key has no registered callable.

        Call once all impl modules (attributes / goal / transforms) are imported.
        """
        missing = sorted(
            {
                w.impl_key
                for w in loaded.lexicon.words
                if w.impl_key and not registry.is_registered(w.impl_key)
            }
        )
        if missing:
            raise RuntimeError(
                "AssetLoader: impl_key(s) with no registered callable: "
                + ", ".join(missing)
            )

    # -- parsers ----------------------------------------------------------- #

    def _path(self, name: str) -> str:
        return os.path.join(self.assets_dir, name)

    def _load_words(self) -> List[Word]:
        path = self._path("words.tsv")
        header, body = _read_tsv(path)
        words: List[Word] = []
        seen: set = set()
        for row in body:
            d = _row_dict(header, row, path)
            wid = d["id"]
            if wid in seen:
                raise ValueError(f"{path}: duplicate Word id {wid!r}")
            seen.add(wid)
            category = d["category"]
            impl_key = d["impl_key"]
            expected = _CATEGORY_PREFIX.get(category)
            if expected is None:
                raise ValueError(f"{path}: unknown category {category!r} (id={wid})")
            if expected and impl_key and not impl_key.startswith(expected):
                raise ValueError(
                    f"{path}: Word {wid!r} (category {category}) impl_key "
                    f"{impl_key!r} must start with {expected!r}"
                )
            # ASSET-A invariant: operator Words carry no naming-ladder slot
            # (their dispatch key lives in impl_key, not slot).
            if (impl_key.startswith("op_") or impl_key.startswith("rel_")) and d.get("slot", ""):
                raise ValueError(
                    f"{path}: operator Word {wid!r} must have an empty slot "
                    f"(got {d['slot']!r}); dispatch key belongs in impl_key"
                )
            words.append(
                Word(
                    id=wid,
                    category=category,
                    part_of_speech=d["part_of_speech"],
                    description=d.get("description", ""),
                    slot=d.get("slot", ""),
                    impl_key=impl_key,
                    params=_parse_params(d.get("params", "")),
                    # Every shipped Word is builtin; the TSV has no origin column.
                    # learned Words enter at runtime via Lexicon.add_word, not here.
                    origin="builtin",
                )
            )
        return words

    def _load_relations(self) -> List[Relation]:
        path = self._path("relations.tsv")
        header, body = _read_tsv(path)
        relations: List[Relation] = []
        for row in body:
            d = _row_dict(header, row, path)
            operands = [o.strip() for o in d["operands"].split(";") if o.strip()]
            relations.append(
                Relation(
                    operator_word_id=d["operator"],
                    operands=list(operands),
                    origin="builtin",
                )
            )
        return relations

    def _load_roles(self, lexicon: Lexicon) -> Dict[str, Role]:
        path = self._path("roles.tsv")
        header, body = _read_tsv(path)
        roles: Dict[str, Role] = {}
        for row in body:
            d = _row_dict(header, row, path)
            label = d["label"]
            if label in roles:
                raise ValueError(f"{path}: duplicate Role label {label!r}")
            tree = parse_predicate(d["recognized_by"])
            if not isinstance(tree, Relation):
                raise ValueError(
                    f"{path}: Role {label!r} recognized_by must be an operator "
                    f"application, got bare {tree!r}"
                )
            self._validate_predicate_operators(tree, lexicon, where=f"role {label}")
            roles[label] = Role(
                label=label,
                description=d.get("description", ""),
                recognized_by=tree,
                category=d.get("category", ""),
            )
        return roles

    def _load_goal_patterns(self, lexicon: Lexicon) -> Tuple[GoalPattern, ...]:
        """Load goal_patterns.tsv into GoalPattern «value» rows.

        ``active*`` rows: parse ``predicate`` into a Relation, FK-validate every
        operator against the Lexicon (op_*/rel_* impl_key), and set
        ``predicate_tree``. ``deferred*`` rows: ``predicate`` is a ``TODO:`` note
        (NOT parsed); ``predicate_tree`` stays None. ``solver_kinds`` splits on
        ``;`` like the role/relation operand lists.
        """
        path = self._path("goal_patterns.tsv")
        header, body = _read_tsv(path)
        patterns: List[GoalPattern] = []
        seen: set = set()
        for row in body:
            d = _row_dict(header, row, path)
            pid = d["id"]
            if pid in seen:
                raise ValueError(f"{path}: duplicate goal_pattern id {pid!r}")
            seen.add(pid)
            status = d["status"]
            solver_kinds = tuple(
                k.strip() for k in d["solver_kinds"].split(";") if k.strip()
            )
            tree: Optional[Relation] = None
            if status.startswith("active"):
                parsed = parse_predicate(d["predicate"])
                if not isinstance(parsed, Relation):
                    raise ValueError(
                        f"{path}: active goal_pattern {pid!r} predicate must be an "
                        f"operator application, got bare {parsed!r}"
                    )
                self._validate_predicate_operators(
                    parsed, lexicon, where=f"goal_pattern {pid}"
                )
                tree = parsed
            elif not status.startswith("deferred"):
                raise ValueError(
                    f"{path}: goal_pattern {pid!r} status {status!r} must start "
                    f"with 'active' or 'deferred'"
                )
            patterns.append(
                GoalPattern(
                    id=pid,
                    goal_kind=d["goal_kind"],
                    predicate=d["predicate"],
                    solver_kinds=solver_kinds,
                    form=d["form"],
                    distance_src=d["distance_src"],
                    origin=d["origin"],
                    status=status,
                    remark=d.get("remark", ""),
                    predicate_tree=tree,
                )
            )
        return tuple(patterns)

    def _load_goal_kinds(self) -> Tuple[GoalKind, ...]:
        """Load goal_kinds.tsv into GoalKind «value» rows (the win-condition SSOT).

        The FK target for ``GoalPattern.goal_kind`` and ``Solver.goal_kinds``; ids
        must be unique (enforced here) and FK-closed (enforced in
        :meth:`_validate_solver_fks`).
        """
        categories = frozenset(
            {"spatial", "structure", "attribute", "quantity", "order", "resource", "survival"}
        )
        path = self._path("goal_kinds.tsv")
        header, body = _read_tsv(path)
        kinds: List[GoalKind] = []
        seen: set = set()
        for row in body:
            d = _row_dict(header, row, path)
            kid = d["id"]
            if kid in seen:
                raise ValueError(f"{path}: duplicate goal_kind id {kid!r}")
            seen.add(kid)
            if d["category"] not in categories:
                raise ValueError(
                    f"{path}: goal_kind {kid!r} category {d['category']!r} "
                    f"not in {sorted(categories)}"
                )
            kinds.append(
                GoalKind(
                    id=kid,
                    category=d["category"],
                    description=d.get("description", ""),
                    remark=d.get("remark", ""),
                )
            )
        return tuple(kinds)

    def _load_solvers(self) -> Tuple[Solver, ...]:
        """Load solvers.tsv into Solver «value» rows (the SolverLibrary.prior seed).

        ``goal_kinds`` and ``parts`` split on ``;`` like the role/relation operand
        lists; both may be empty (polymorphic / runtime-bound solvers such as
        ``composite`` / ``nrpa_adaptive_playout``). ids must be unique (enforced
        here) and FK-closed (enforced in :meth:`_validate_solver_fks`).
        """
        backends = frozenset({"SearchHeuristic", "ConstrainedGenerator", "Simulator"})
        path = self._path("solvers.tsv")
        header, body = _read_tsv(path)
        solvers: List[Solver] = []
        seen: set = set()
        for row in body:
            d = _row_dict(header, row, path)
            sid = d["id"]
            if sid in seen:
                raise ValueError(f"{path}: duplicate solver id {sid!r}")
            seen.add(sid)
            if d["backend"] not in backends:
                raise ValueError(
                    f"{path}: solver {sid!r} backend {d['backend']!r} "
                    f"not in {sorted(backends)}"
                )
            goal_kinds = tuple(
                k.strip() for k in d["goal_kinds"].split(";") if k.strip()
            )
            parts = tuple(p.strip() for p in d["parts"].split(";") if p.strip())
            solvers.append(
                Solver(
                    category=d["category"],
                    id=sid,
                    goal_kinds=goal_kinds,
                    world_signature=d["world_signature"],
                    verification_horizon=d["verification_horizon"],
                    backend=d["backend"],
                    parts=parts,
                    algorithm=d["algorithm"],
                    description=d.get("description", ""),
                    remark=d.get("remark", ""),
                )
            )
        return tuple(solvers)

    # -- validation -------------------------------------------------------- #

    @staticmethod
    def _is_operator(word: Word) -> bool:
        return word.impl_key.startswith("op_") or word.impl_key.startswith("rel_")

    def _validate_operator_fks(self, lexicon: Lexicon) -> None:
        """Every taxonomy Relation operator must resolve to an operator Word."""
        for r in lexicon.relations:
            self._validate_predicate_operators(r, lexicon, where="taxonomy")

    def _validate_solver_fks(
        self,
        goal_patterns: Tuple[GoalPattern, ...],
        goal_kinds: Tuple[GoalKind, ...],
        solvers: Tuple[Solver, ...],
    ) -> None:
        """The dispatch-triangle FK closures (raise on any dangling reference).

        Four gates (empty ``;``-lists pass trivially):
          (i)   goal_patterns.goal_kind  subset of  goal_kinds.id
          (ii)  solvers.goal_kinds       subset of  goal_kinds.id
          (iii) goal_patterns.solver_kinds subset of solvers.id
          (iv)  solvers.parts            subset of  solvers.id  (self-FK)
        """
        kind_ids = {k.id for k in goal_kinds}
        solver_ids = {s.id for s in solvers}

        for p in goal_patterns:
            if p.goal_kind not in kind_ids:
                raise ValueError(
                    f"goal_patterns: {p.id!r} goal_kind {p.goal_kind!r} "
                    f"is not a goal_kinds.id"
                )
            for k in p.solver_kinds:
                if k not in solver_ids:
                    raise ValueError(
                        f"goal_patterns: {p.id!r} solver_kind {k!r} "
                        f"is not a solvers.id"
                    )
        for s in solvers:
            for gk in s.goal_kinds:
                if gk not in kind_ids:
                    raise ValueError(
                        f"solvers: {s.id!r} goal_kind {gk!r} is not a goal_kinds.id"
                    )
            for part in s.parts:
                if part not in solver_ids:
                    raise ValueError(
                        f"solvers: {s.id!r} part {part!r} is not a solvers.id"
                    )

    def _validate_predicate_operators(
        self, node: Union[Relation, str], lexicon: Lexicon, where: str
    ) -> None:
        if not isinstance(node, Relation):
            # Bare leaf (Role label / Word id / the 'self' env binding). The
            # loader intentionally does NOT check leaf arity/type: e.g.
            # inside(self, box) passes a Word (box) and an env binding (self)
            # where words.tsv declares inside operands = Role,Role. Resolving and
            # type-checking leaves per operator is the interpreter's / AnalogizeRoles'
            # job in the evaluation env (gr-arc-3-operators.md §2 preamble).
            return
        op = node.operator_word_id
        if not lexicon.has_word(op):
            raise ValueError(f"{where}: operator {op!r} is not a Word in the Lexicon")
        if not self._is_operator(lexicon.word(op)):
            raise ValueError(
                f"{where}: Word {op!r} is not an operator (impl_key must be op_*/rel_*)"
            )
        for child in node.operands:
            self._validate_predicate_operators(child, lexicon, where)
