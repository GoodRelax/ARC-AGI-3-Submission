"""[agent/core] SolveGame -- the GENERAL self-solve controller (CMP-22; TS-23 / TS-24).

The integration capstone: a per-turn ``choose_action`` / ``is_done`` that wires the WHOLE
frozen stack end-to-end, over the GENERAL clusters (abstract AbstractSituation / ModelWorld /
ModelGoal / solver), so the architecture is proven on real captures rather than the L1
tuple-keyed logic:

    GameIO.read_frame -> perceive
        -> role assignment (find_controllable + interaction observation, lifted from L1)
        -> StateAbstraction.project -> AbstractSituation
        -> ModelWorld.refine / learn (from the last transition) + predict (look-ahead)
        -> ModelGoal.hypothesize (no win yet) / refine_from_win (after a win) [+ Roadmap]
        -> solver.plan_roadmap / plan -> GamePlan ; commit GamePlan.first()
        -> validate_action -> an action in available_actions (ACTION6 coords if needed)
        -> DetectFutility.record (the played move's observed MoveEffect)
        -> on predict != observe: diagnose_divergence -> route to the phase -> act
        -> GameIO.emit_turn (a TurnRecord carrying every TURN_RECORD_KEYS field)
        -> return the committed action (a member of available_actions).

This is a SEPARATE controller from the proven L1 ``agent.core.play.Agent`` (which stays the
byte-for-byte concrete grid slice the framework + tools import). ``SolveGame`` reuses the L1
Agent's proven ideas -- ``find_controllable`` for the controllable role, and observing an
interaction (a footprint touch that advances some object's pose) to name the interactor /
target -- but lifts them to the GENERAL types and the abstract AbstractSituation.

Sequence canon (cite, never duplicate):
  - _assets/gr-arc-3-sequence-choose-action.md (sheet 1, v005) -- the per-turn stage order
    (1) EXPLORE/perceive -> (2) MODEL WORLD (refine + DetectFutility) -> (3) MODEL GOAL ->
    (4) PLAN (RHAE-0 look-ahead) -> (5) ACT (commit ONE move) -> (6) DIAGNOSE (next turn).
  - _assets/gr-arc-3-sequence-diagnose.md (sheet 3, v005) -- on the first divergence,
    diagnose_divergence localizes the delta and routes to (1)/(2)/(3)/(4); the controller
    updates that stage and re-plans so the next move CHANGES (the MPC repair loop).
  - tools/inspector/trace-schema.md -- the TurnRecord (2) emit contract (CMP-37 GameIO).

Hard rules honoured: RHAE-0 in look-ahead (only ModelWorld.predict; 0 scored moves -- the
solver guarantees this); determinism (DP-10: no RNG; no builtin ``hash()`` for identity --
AbstractSituation.canonical drives equality); no game literals (NFR-6: roles/goals are
observation-derived; the committed action is always a member of available_actions);
English/ASCII; full type annotations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agent.controllable import find_controllable
from agent.core import attributes as A
from agent.core import goal as G
from agent.core import solver as Solve
from agent.core.goal import (
    GoalPredicate,
    ModelGoal,
    Roadmap,
    relation,
)
from agent.core.perceive import Obj
from agent.core.play import (
    DetectFutility,
    Phase,
    PhaseRouter,
    RoutingVerdict,
    diagnose_divergence,
)
from agent.core.situation import AbstractSituation, StateAbstraction
from agent.core.solver import GamePlan, PlanMoves
from agent.core.world_model import (
    InteractionRule,
    ModelWorld,
    MoveEffect,
    WorldModel as NavWorldModel,
)
from agent.game_io import (
    available_action_ids,
    emit_turn,
    read_frame,
    validate_action,
)

# ----------------------------------------------------------------------- role labels
# The participating role LABELS (StateAbstraction's PARTICIPATING_ROLES). They are role
# AXES produced by observation, NOT game literals (NFR-6): "controllable" = the rigid mover
# find_controllable returns; "interactor" = an object whose touch advanced some pose;
# "target" = the goal mark the interaction revealed. Everything else is unclassified.
ROLE_CONTROLLABLE: str = "controllable"
ROLE_INTERACTOR: str = "interactor"
ROLE_TARGET: str = "target"
ROLE_CARRIED: str = "carried-state"

# The scalar gauge name for a monotone-decreasing observed Dimension (Move Budget, TERM-32).
# A NAME, never a colour/coordinate -- the projector decides it from observation.
MOVE_BUDGET: str = "move_budget"

# The CONCRETE movement repertoire the navigate Solver calibrates (the four simple-move action
# ids the framework offers for ls20). These are opaque action TOKENS, not game literals -- they
# are the same ids ``play.py::Agent`` rounds-robin through; CALIBRATE learns each one's effect
# from observation and EXPLORE/SOLVE search over only the learned ``move_delta`` (NFR-6).
NAV_ACTIONS: Tuple[int, ...] = (1, 2, 3, 4)


def _centroid_distance(a: Obj, b: Obj) -> float:
    """Manhattan distance between two objects' centroids (a frame-to-frame proximity key)."""
    return abs(a.centroid[0] - b.centroid[0]) + abs(a.centroid[1] - b.centroid[1])


def _obj_overlapping(cells: frozenset, objs: Sequence[Obj]) -> Optional[Obj]:
    """The object in ``objs`` whose cells overlap ``cells`` the MOST (or ``None``)."""
    best: Optional[Obj] = None
    best_ov = 0
    for o in objs:
        ov = len(o.cells & cells)
        if ov > best_ov:
            best, best_ov = o, ov
    return best


def _nearest_same_color(o: Obj, prev_objs: Sequence[Obj]) -> Optional[Obj]:
    """The previous-frame object of the same dominant colour nearest ``o`` (frame-to-frame
    identity by colour + proximity), mirroring the L1 Agent's interaction observer."""
    cands = [p for p in prev_objs if p.dom_color == o.dom_color]
    if not cands:
        return None
    return min(cands, key=lambda p: _centroid_distance(p, o))


@dataclass
class RoleAssignment:
    """The observation-derived role labels for the current frame's objects (NFR-6).

    Built from ``find_controllable`` (the controllable) + the L1 interaction observer (an
    object whose pose advanced after a footprint touch names the interactor; the revealed
    twin names the target). Holds, per role, the OBJECT carrying it -- the StateAbstraction
    consumes only the role LABELS, never a surface literal.
    """

    roles: Dict[Obj, str] = field(default_factory=dict)
    controllable: Optional[Obj] = None
    target: Optional[Obj] = None

    def role_of(self, obj: Obj) -> Optional[str]:
        return self.roles.get(obj)


class SolveGame:
    """The general self-solve controller (CMP-22): ``choose_action`` / ``is_done``.

    One instance plays one episode. Each ``choose_action`` runs the full per-turn pipeline
    (sheet 1 stage order) and commits EXACTLY ONE action (a member of that turn's
    ``available_actions``). The look-ahead is virtual (the solver only calls
    ``ModelWorld.predict`` -- 0 scored moves, RHAE-0). When the previous move's armed
    prediction diverges from this turn's observation, ``diagnose_divergence`` routes to the
    wrong stage, the controller updates that stage and re-plans, so the next move CHANGES
    (the MPC repair loop) -- and the loop terminates in finite turns (the futile-run guard +
    the bounded planner).

    Determinism (DP-10): role assignment, AbstractSituation identity, planning, and diagnosis are all
    canonical / sorted / RNG-free, and no builtin ``hash()`` decides identity.
    """

    def __init__(
        self,
        lexicon: Optional[A.Lexicon] = None,
        horizon: int = 16,
        trace_path: Optional[Any] = None,
        futile_run_limit: int = 8,
        settings: Any = None,
    ) -> None:
        # ---- the frozen clusters, wired once ----
        self.abstraction = StateAbstraction()
        self.world = ModelWorld()
        self.model_goal = ModelGoal()
        self.planner = PlanMoves(horizon=horizon)
        self.detect_futility = DetectFutility()
        self.lexicon = lexicon if lexicon is not None else A.Lexicon()

        # ---- per-episode working memory ----
        self.prev_situation: Optional[AbstractSituation] = None        # the AbstractSituation BEFORE the move
        self.armed_prediction: Optional[AbstractSituation] = None      # predicted next (for diagnose)
        self.armed_move: Optional[int] = None                  # the move whose effect we await
        self.goal: Optional[GoalPredicate] = None              # current goal hypothesis
        self.roadmap: Optional[Roadmap] = None                 # multi-stage track (if any)
        self.won_situation: Optional[AbstractSituation] = None         # the first win AbstractSituation (bootstrap)
        self.won: bool = False

        # ---- L1-reused role-assignment state ----
        self.ctrl_state: Optional[dict] = None                 # find_controllable carry
        self.prev_objs: Optional[List[Obj]] = None
        self.offsets: Optional[frozenset] = None               # controllable footprint shape
        self.target_obj: Optional[Obj] = None                  # locked goal mark
        self.interactor_cells: frozenset = frozenset()         # last observed interactor footprint

        # ---- the CONCRETE navigate Solver (nav=A, ADR-015 / sheet 5) -----------------------
        # SolveGame's abstract pipeline (project/diagnose/plan over AbstractSituation) is the black
        # box ABOVE; this is the concrete floor BELOW it -- a SEPARATE move source porting the
        # PROVEN ``play.py::_decide`` priority (CALIBRATE -> SOLVE -> EXPLORE -> fallback). It reads
        # grid geometry + a learned concrete WorldModel (move_delta/passable/triggers) and searches
        # the «value» ConcreteSituation =(controllable_pos, carried_pose, consumed_triggers) via the
        # reused ``solver.bfs_solve``/``bfs_to``. ``play.py::Agent`` is untouched (its byte-for-byte
        # reference is imported by the tools); this is an independent path on the GENERAL controller.
        self.nav_model: NavWorldModel = NavWorldModel()        # concrete learned nav-dynamics
        self.prev_avatar_pos: Optional[Tuple[int, int]] = None  # controllable bbox-anchor last turn
        self.last_committed: Optional[int] = None              # the move actually played last turn
        self.target_pose: Optional[frozenset] = None           # locked target ORIENTATION (win key)
        self.container: frozenset = frozenset()                # delivery region cells (deliver+match)
        self.poked: List[frozenset] = []                       # (legacy per-frame dedup; superseded below)
        # probe-status give-up (sheet 5; memo 2026-06-23): an object whose touch produced NO effect IN
        # THIS CONTEXT is 'unresponsive' -> curiosity stops re-targeting it. Keyed on
        # (context = prev AbstractSituation hash, object SHAPE) -- a literal-free, position-stable key
        # (NOT the fragile cell-set), so it RE-OPENS when the context changes (the magic-circle case).
        self._unresponsive: set = set()
        self._probe_context: str = ""                          # probe-status context key (constant for L1)
        self._pending_shape: Optional[frozenset] = None        # shape curiosity is probing (judged next turn)
        self._probe_responded: bool = False                    # a real pose change fired this turn (true effect)
        self._cal: int = -1                                    # CALIBRATE round-robin cursor
        self._cycle: int = 0                                   # fallback action cursor
        # The ConcreteSituation visited/dead set: a GUARDRAIL keyed on
        # (controllable_pos, carried_pose, consumed_triggers), used ONLY when ``_futility_check``
        # is on. The PRIMARY loop-prevention is intrinsic (the BFS per-search ``seen`` + the
        # ``poked`` give-up set + CALIBRATE round-robin), so L1 self-solves with this OFF (DP-20).
        self._concrete_seen: set = set()
        self._frontier_visited: set = set()                    # footprint cells ever occupied (frontier)
        self.goal_first_turn: Optional[int] = None             # turn the goal first locked (probe)

        # ---- bookkeeping ----
        self.turn: int = 0
        self.level: int = 0
        self.trace_path = trace_path
        self.capture_file: str = ""                            # set via bind_capture
        self.futile_run_limit = int(futile_run_limit)
        # Guardrail toggle (ADR-014 / DP-20): the futility check is a CONFIGURABLE guardrail;
        # with it off the core must still solve. The value is read off ``settings`` (duck-typed
        # -- the boundary loader ``agent.settings`` builds it) and defaults to ON so an absent
        # setting is safe. The pure core never reads the file; it receives the resolved value.
        self._futility_check: bool = (
            True if settings is None else bool(getattr(settings, "futility_check", True))
        )
        self.last_verdict: Optional[RoutingVerdict] = None
        self.last_objects: List[Obj] = []
        self.last_role_assignment: RoleAssignment = RoleAssignment()

        # The phase router: each handler records that the stage was re-entered (the diagnose
        # repair). The handlers are pure flags here (the actual stage update happens inline in
        # ``choose_action`` after routing); the router proves re-entry for a spy (SC-11).
        self._reentered: Dict[str, int] = {p: 0 for p in Phase.ALL}
        self.router = PhaseRouter(
            explore=self._mark(Phase.EXPLORE),
            world=self._mark(Phase.WORLD),
            goal=self._mark(Phase.GOAL),
            plan=self._mark(Phase.PLAN),
        )

    # ------------------------------------------------------------------ wiring helpers
    def _mark(self, phase: str) -> Callable[[RoutingVerdict], str]:
        """Build a phase handler that records a re-entry into ``phase`` (the diagnose repair
        spy) and returns the phase label."""

        def _handler(_verdict: RoutingVerdict) -> str:
            self._reentered[phase] = self._reentered.get(phase, 0) + 1
            return phase

        return _handler

    def reentries(self, phase: str) -> int:
        """How many times the controller re-entered ``phase`` via diagnose routing (spy)."""
        return self._reentered.get(phase, 0)

    def bind_capture(self, capture_file: str, level: int = 0) -> "SolveGame":
        """Record the capture file these TurnRecords reference (``capture_ref.file``) and the
        starting level. Returns ``self`` so binding chains."""
        self.capture_file = str(capture_file)
        self.level = int(level)
        return self

    # ------------------------------------------------------------------ public API
    def is_done(self, frame_response: Any = None) -> bool:
        """True once a win has been observed (the episode is solved)."""
        return self.won

    def choose_action(self, frame_response: Any) -> int:
        """Run the full per-turn pipeline and return the committed action id (always a member
        of this turn's ``available_actions``). Drives every stage of sheet 1, the (6) diagnose
        repair, and the (2) TurnRecord emit.
        """
        # (1) EXPLORE / perceive -- GameIO read_frame -> objects (settled grid) + full frames.
        objects, frames = read_frame(frame_response)
        available = available_action_ids(frame_response)
        settled_grid = self._settled_grid(frames)
        self.last_objects = objects

        # probe-status context (sheet 5). For L1 there is no precondition context (nothing to gather),
        # so a CONSTANT context = a shape-only give-up (a touched-inert shape is not re-targeted). The
        # context-keyed RE-OPEN (the magic-circle case) needs a STABLE COARSE context: the naive
        # AbstractSituation hash THRASHES because the avatar merging with the carried object flips the
        # situation each turn, so the give-up never sticks. A stable coarse context is deferred
        # (docs/deferred-todo.md); shape-only is correct wherever the context is constant.
        self._probe_context = ""
        self._probe_responded = False        # set by _observe_interaction on a real pose change (effect)

        # (2a) role assignment -- find_controllable + the L1 interaction observer, lifted to
        # the general role labels (NFR-6: derived from observation, never a literal). The observer
        # ALSO grounds the concrete nav-model (trigger -> pose_succ) and locks the deliver+match
        # goal (target_pose + container) on the first observed interaction.
        assignment = self._assign_roles(settled_grid, objects)
        self.last_role_assignment = assignment
        # probe-status give-up (sheet 5): resolve the in-flight probe -- the shape curiosity committed
        # TOWARD last turn. A carried/merged object is NOT a separate object at contact, so judge by the
        # PROBE TARGET, and on a TRUE EFFECT signal (a pose change THIS turn, ``_probe_responded``), NOT
        # trigger-table growth (a re-touch of a known trigger still responds). No effect -> 'unresponsive'.
        if self._pending_shape is not None and self.target_obj is None and not self._probe_responded:
            self._unresponsive.add((self._probe_context, self._pending_shape))
        self._pending_shape = None
        avatar_pos = (
            tuple(assignment.controllable.pos) if assignment.controllable is not None else None
        )
        if avatar_pos is not None and self.offsets is not None:
            self._frontier_visited |= {
                (avatar_pos[0] + dr, avatar_pos[1] + dc) for dr, dc in self.offsets
            }

        # (nav) learn the concrete nav-dynamics from the LAST transition (move/wall/passable),
        # then run the CONCRETE navigate Solver (nav=A, sheet 5) -- the PRIMARY move source:
        # CALIBRATE -> SOLVE -> EXPLORE. ``None`` means defer to the abstract fallback below.
        self._learn_nav(settled_grid, avatar_pos)
        concrete_move = self._decide_concrete(
            settled_grid, objects, assignment.controllable, avatar_pos
        )
        concrete_move = self._concrete_guardrail(concrete_move, settled_grid, avatar_pos)

        # (3-pre) project the OBSERVED AbstractSituation (StateAbstraction; salient-only, carry-forward).
        scalars = self._observe_scalars(frame_response, objects, assignment)
        relations = self._observe_relations(assignment, objects)
        observed = self.abstraction.project(
            objects,
            prev=self.prev_situation,
            roles=assignment.roles,
            profiles=self._profiles(objects, assignment),
            relations=relations,
            scalars=scalars,
        )

        # (2b) DetectFutility -- classify the ACTUALLY-played move's MoveEffect (every turn).
        move_effect: Optional[str] = None
        if (self._futility_check
                and self.prev_situation is not None and self.armed_move is not None):
            move_effect = self.detect_futility.record(
                self.prev_situation, observed, self._goal_distance
            )

        # (2c) MODEL WORLD -- refine rules from the last observed transition (per-move feedback).
        world_changed = False
        learned: List[str] = []
        if self.prev_situation is not None and self.armed_move is not None:
            world_changed, learned = self._refine_world(
                self.prev_situation, self.armed_move, observed
            )

        # (6) DIAGNOSE -- on the FIRST divergence of the armed prediction vs reality, localize
        # the wrong stage and route back. Updating that stage + re-planning makes the next move
        # CHANGE (MPC repair). Runs only when a prediction was armed AND it diverged.
        verdict: Optional[RoutingVerdict] = None
        if self.armed_prediction is not None and self.armed_move is not None:
            if observed.canonical() != self.armed_prediction.canonical():
                verdict = self._diagnose_and_repair(observed)
        self.last_verdict = verdict

        # (3) MODEL GOAL -- hypothesize from a prior (no win yet) or refine from the first win.
        self._update_goal(observed, assignment)

        # (4) PLAN -- RHAE-0 look-ahead over the built-in sim, ranked by the goal distance.
        game_plan = self._plan(observed, available)

        # (5) ACT -- commit ONE move (MPC). The CONCRETE navigate Solver (nav=A) is the PRIMARY
        # source: when it committed a move (CALIBRATE/SOLVE/EXPLORE), play THAT; only when it
        # deferred (``None``) does the abstract plan / ``_fallback_move`` choose (the abstract
        # pipeline above still ran, so the trace, diagnosis, and goal bootstrap stay intact).
        if concrete_move is not None and int(concrete_move) in [int(a) for a in available]:
            committed_id, coords = int(concrete_move), None
        else:
            committed_id, coords = self._commit(game_plan, observed, available, verdict)
        result = validate_action(committed_id, available, coords=coords)

        # arm the prediction of the committed move for next turn's (6) diagnose.
        self.armed_prediction = self.world.predict(observed, committed_id)
        self.armed_move = committed_id
        self.prev_situation = observed
        self.prev_objs = objects
        # carry the concrete nav-Solver state forward (so next turn's _learn_nav reads the right
        # last-transition: the controllable bbox-anchor and the move actually played).
        self.prev_avatar_pos = avatar_pos
        self.last_committed = int(committed_id)
        if self.target_obj is not None and self.goal_first_turn is None:
            self.goal_first_turn = self.turn

        # win bookkeeping (the env signal, read from the frame).
        self._update_win(frame_response, observed)

        # (2) emit -- append one TurnRecord carrying every TURN_RECORD_KEYS field.
        snapshot = self._turn_record(
            observed=observed,
            objects=objects,
            assignment=assignment,
            game_plan=game_plan,
            committed_id=committed_id,
            coords=coords,
            move_effect=move_effect,
            world_changed=world_changed,
            learned=learned,
        )
        if self.trace_path is not None:
            emit_turn(snapshot, self.trace_path)

        self.turn += 1
        return result.id

    # ------------------------------------------------------------------ (2a) role assignment
    def _assign_roles(self, grid: Optional[np.ndarray], objects: Sequence[Obj]) -> RoleAssignment:
        """Assign observation-derived role labels (NFR-6), reusing the L1 Agent's proven logic:
          * the controllable = the rigid mover ``find_controllable`` returns (matched to the
            overlapping perceive object);
          * the interactor = an object whose footprint the controllable touched and whose
            (or another object's) POSE advanced as a result (the L1 ``_observe_carried`` idea);
          * the target = the twin the interaction revealed (locked once found).
        Everything else stays unclassified (non-participating).
        """
        assignment = RoleAssignment()
        if grid is None:
            return assignment

        cells, self.ctrl_state = find_controllable(grid, self.ctrl_state)
        controllable = _obj_overlapping(cells, objects) if cells is not None else None
        if controllable is not None:
            assignment.roles[controllable] = ROLE_CONTROLLABLE
            assignment.controllable = controllable
            if self.offsets is None:
                self.offsets = frozenset(
                    (r - controllable.pos[0], c - controllable.pos[1])
                    for r, c in controllable.cells
                )

        # Interaction observation: a pose change in some object after the controllable acted
        # names the interactor (the thing under the controllable) + reveals the target twin.
        if controllable is not None and self.prev_objs is not None:
            self._observe_interaction(grid, objects, controllable, assignment)

        # Carry a locked target forward (continuity), and re-bind it to this frame's object.
        if self.target_obj is not None:
            cur = self._relocate(self.target_obj, objects)
            if cur is not None:
                assignment.roles[cur] = ROLE_TARGET
                assignment.target = cur

        return assignment

    def _observe_interaction(
        self,
        grid: Optional[np.ndarray],
        objects: Sequence[Obj],
        controllable: Obj,
        assignment: RoleAssignment,
    ) -> None:
        """The L1 interaction observer, lifted: if some object's pose advanced vs the previous
        frame, the object now UNDER the controllable's footprint is the interactor, and the
        changed object's twin (same colour + shape, different pose) is the goal target.

        ALSO grounds the CONCRETE navigate Solver (port of ``play.py::_observe_carried`` +
        ``_lock_goal``): the pose transition is learned into ``nav_model`` (trigger -> pose_succ)
        so the BFS can advance the carried pose THROUGH the interactor, and -- on the first lock --
        ``target_pose`` (the win orientation) and ``container`` (the delivery region via
        ``goal.container_of``) are recorded. NFR-6: target/twin/container are OBSERVED, never
        hard-coded; the trigger is the cell-set the footprint overlapped."""
        prev_objs = self.prev_objs or []
        for o in objects:
            po = _nearest_same_color(o, prev_objs)
            if po is not None and A.pose(po) != A.pose(o):
                self._probe_responded = True              # a pose change = a real interaction (probe-status)
                interactor = self._overlapped(controllable, prev_objs)
                if interactor is not None:
                    assignment.roles[interactor] = ROLE_INTERACTOR
                    self.interactor_cells = interactor.cells
                # Ground the concrete nav-model: the fresh footprint-overlap advanced this pose.
                trig = interactor.cells if interactor is not None else frozenset()
                self.nav_model.learn_trigger(trig, A.pose(po), A.pose(o))
                # the changed object is the carried state; its twin is the target.
                assignment.roles.setdefault(o, ROLE_CARRIED)
                if self.target_obj is None:
                    twin = self._twin_of(objects, o)
                    if twin is not None:
                        self.target_obj = twin
                        assignment.roles[twin] = ROLE_TARGET
                        assignment.target = twin
                        self._lock_nav_goal(grid, twin)
                return

    def _lock_nav_goal(self, grid: Optional[np.ndarray], target: Obj) -> None:
        """Lock the concrete deliver+match goal for the navigate Solver (port of
        ``play.py::_lock_goal``): record the target ORIENTATION (the win-pose, part of the
        ConcreteSituation so a wrong-pose delivery never counts as a win) and the delivery
        ``container`` -- the smallest single-colour box enclosing the target (``goal.container_of``,
        the generic 'a mark inside a frame'; the border colour is whatever rings the target)."""
        self.target_pose = A.pose(target)
        if grid is None:
            return
        box = G.container_of(grid, target)
        if box:
            r0, c0, r1, c1 = box
            self.container = frozenset(
                (r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            )

    def _overlapped(self, controllable: Obj, prev_objs: Sequence[Obj]) -> Optional[Obj]:
        """The previous-frame object (other than the controllable) the controllable's footprint
        overlapped -- the interaction trigger (L1 ``_overlapped``)."""
        foot = set(controllable.cells)
        for o in prev_objs:
            if o.dom_color != controllable.dom_color and (foot & o.cells):
                return o
        return None

    # ------------------------------------------------------------------ concrete nav learning
    def _learn_nav(
        self,
        grid: Optional[np.ndarray],
        avatar_pos: Optional[Tuple[int, int]],
    ) -> None:
        """Learn the concrete nav-dynamics from the LAST transition (port of ``play.py::_learn``,
        move/wall/passable half -- the trigger half is locked in :meth:`_observe_interaction`).

        ``learn_move`` records each action's translation; a no-move means a wall is ahead
        (``learn_wall``); a real move means the cells the footprint now occupies are passable
        (``learn_passable``). Game-literal-free: colours are DISCOVERED as walls/passable from
        observation; the action id is an opaque token. RHAE-0-safe (pure bookkeeping)."""
        if self.last_committed is None or self.prev_avatar_pos is None or avatar_pos is None:
            return
        disp = (avatar_pos[0] - self.prev_avatar_pos[0], avatar_pos[1] - self.prev_avatar_pos[1])
        # Learn each action's translation ONCE (lock it). The per-action step is physically
        # constant, but the OBSERVED displacement is unreliable on the real engine near a wall or
        # on a shape/overlap change (a clipped or over-shot move) -- overwriting a known-good delta
        # corrupts the BFS and strands navigation (sim != engine; the empirical L1 failure). Lock
        # from the first clean (non-zero) observation; CALIBRATE supplies it near spawn. MPC re-plans
        # from the REAL observed position each turn, so a one-off model error self-corrects.
        if self.last_committed not in self.nav_model.move_delta and disp != (0, 0):
            self.nav_model.learn_move(self.last_committed, disp)
        if grid is None:
            return
        if disp == (0, 0):
            self._learn_wall_ahead_nav(grid)
        elif self.offsets is not None:
            r0, c0 = avatar_pos
            h, w = grid.shape
            self.nav_model.learn_passable(
                int(grid[r0 + dr, c0 + dc]) for dr, dc in self.offsets
                if 0 <= r0 + dr < h and 0 <= c0 + dc < w        # footprint may straddle the board edge
            )

    def _learn_wall_ahead_nav(self, grid: np.ndarray) -> None:
        """On a no-op move, the colour at the controllable's LEADING edge (the cells the move
        would push it into) is a wall colour (port of ``play.py::_learn_wall_ahead``)."""
        d = self.nav_model.move_delta.get(self.last_committed)
        if d is None or self.offsets is None or self.prev_avatar_pos is None:
            return
        r0, c0 = self.prev_avatar_pos
        for dr, dc in self.offsets:
            if (dr + d[0], dc + d[1]) in self.offsets:
                continue                       # still inside the body -> not a leading edge
            rr, cc = r0 + dr + d[0], c0 + dc + d[1]
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                col = int(grid[rr, cc])
                if col not in self.nav_model.passable:
                    self.nav_model.learn_wall(col)

    @staticmethod
    def _twin_of(objects: Sequence[Obj], obj: Obj) -> Optional[Obj]:
        """An object with the same colour AND shape as ``obj`` in another pose (its twin)."""
        for o in objects:
            if o.pos != obj.pos and o.dom_color == obj.dom_color and A.shape(o) == A.shape(obj):
                return o
        return None

    @staticmethod
    def _relocate(obj: Obj, objects: Sequence[Obj]) -> Optional[Obj]:
        """Re-find ``obj`` in the current frame by colour + shape (frame-to-frame identity)."""
        for o in objects:
            if o.dom_color == obj.dom_color and A.shape(o) == A.shape(obj):
                return o
        return None

    # ============================================================ concrete navigate Solver
    # The per-turn CONCRETE dispatch (port of ``play.py::_decide``, priority order):
    #   (C) CALIBRATE  -- ``not nav_model.moves_known()`` -> round-robin an un-learned action;
    #   (S) SOLVE      -- ``target_obj`` locked + carried pose readable -> ``bfs_solve`` deliver+match;
    #   (E) EXPLORE    -- ``_curiosity``: nearest un-poked compact object -> ``bfs_to`` to provoke it;
    #   (F) (caller)   -- returns ``None`` so ``choose_action`` falls back to the abstract path.
    # This is the navigate Solver (nav=A, ADR-015) -- the ONE concretely-grounded Solver reading
    # grid geometry + the learned ``nav_model``. The reused ``solver.bfs_solve``/``bfs_to`` (the
    # SAME functions ``play.py`` calls) search the «value» ConcreteSituation.
    def _decide_concrete(
        self,
        grid: Optional[np.ndarray],
        objects: Sequence[Obj],
        controllable: Optional[Obj],
        avatar_pos: Optional[Tuple[int, int]],
    ) -> Optional[int]:
        """Return the concrete committed move (a NAV_ACTIONS id), or ``None`` to defer to the
        abstract fallback. Ports ``play.py::_decide`` verbatim in priority; RHAE-0 (the BFS is
        pure look-ahead over ``nav_model``); commits ONE move/turn (MPC)."""
        if grid is None or controllable is None or avatar_pos is None or self.offsets is None:
            return None
        # (C) CALIBRATE -- learn every action's translation, round-robin so the avatar stays near
        # spawn (trying one action repeatedly can strand it where that action is walled).
        if not self.nav_model.moves_known():
            for _ in range(len(NAV_ACTIONS)):
                self._cal = (self._cal + 1) % len(NAV_ACTIONS)
                if NAV_ACTIONS[self._cal] not in self.nav_model.move_delta:
                    return NAV_ACTIONS[self._cal]
        # (S) SOLVE -- goal locked -> BFS deliver+match (the carried pose must be readable so the
        # orientation axis of the win is grounded).
        if self.target_obj is not None and self.target_pose is not None:
            carried_pose = self._carried_pose(objects)
            if carried_pose is not None:
                plan = Solve.bfs_solve(
                    self.nav_model, grid, self.offsets, avatar_pos,
                    carried_pose, self.target_pose, self.container,
                )
                if plan:
                    return plan[0]
        # (E) EXPLORE -- navigate to the nearest compact, un-poked object to provoke an interaction.
        a = self._curiosity(grid, objects, controllable, avatar_pos)
        if a is not None:
            return a
        # (E2) FRONTIER -- no object reachable/worth probing now: explore the nearest UNVISITED cell
        # (ExploreWorld) to change position + knowledge so access/context re-open, instead of a blind
        # legal move (the empirical "stuck at the wall" fix; sheet 5 EXPLORE selection model).
        f = self._frontier_move(grid, avatar_pos)
        if f is not None:
            return f
        # (F) defer: nothing learned / nowhere new to go -> abstract fallback.
        return None

    def _carried_pose(self, objects: Sequence[Obj]) -> Optional[frozenset]:
        """The current pose (orientation) of the carried twin -- the object sharing the target's
        colour + shape but at another position (port of ``play.py::_carried_pose``). ``None`` when
        no goal is locked or no twin is visible this frame."""
        if self.target_obj is None:
            return None
        cands = [
            o for o in objects
            if o.pos != self.target_obj.pos
            and o.dom_color == self.target_obj.dom_color
            and A.shape(o) == A.shape(self.target_obj)
        ]
        return A.pose(cands[0]) if cands else None

    def _curiosity(
        self,
        grid: np.ndarray,
        objects: Sequence[Obj],
        controllable: Obj,
        avatar_pos: Tuple[int, int],
    ) -> Optional[int]:
        """The goalless curiosity drive (the EXPLORE arm sheet 1 lacks): rank COMPACT objects
        (``size <= 2*avatar_size``) by Manhattan distance, skip the ``_unresponsive`` give-up set
        (probe-status, sheet 5), and ``bfs_to`` the nearest reachable one to PROVOKE an interaction.

        Loop-prevention: the per-search ``seen`` set inside ``bfs_to`` (intrinsic, always on) + the
        probe-status give-up (an object whose touch produced no effect IN THIS CONTEXT is never
        re-targeted -- keyed on a position-stable shape, so it re-opens when the context changes)."""
        avatar_size = len(self.offsets) if self.offsets is not None else controllable.size
        # skip objects 'unresponsive' in THIS context (a touch that produced no effect, marked in
        # choose_action) -- a position-stable shape key, NOT the fragile cell-set (the carried-state
        # oscillation fix). An unreachable object is just skipped this turn (re-evaluated next turn).
        cands = [
            o for o in objects
            if o is not controllable and o.size <= 2 * avatar_size
            and (self._probe_context, A.shape(o)) not in self._unresponsive
        ]
        cands.sort(
            key=lambda o: abs(o.centroid[0] - controllable.centroid[0])
            + abs(o.centroid[1] - controllable.centroid[1])
        )
        for o in cands:
            path = Solve.bfs_to(self.nav_model, grid, self.offsets, avatar_pos, set(o.cells))
            if path:                                       # a move toward it -> probe it (judged next turn)
                self._pending_shape = A.shape(o)
                return path[0]
            # path == [] (already overlapping) / None (no path): try the next candidate.
        return None

    def _frontier_move(
        self,
        grid: Optional[np.ndarray],
        avatar_pos: Optional[Tuple[int, int]],
    ) -> Optional[int]:
        """Frontier exploration (ExploreWorld; sheet 5 EXPLORE selection model): when no object is
        reachable/worth probing, navigate toward the nearest UNVISITED footprint-position over the
        learned nav-dynamics, to change position + knowledge so access / context re-open. Returns the
        first move toward the nearest cell whose footprint covers a never-occupied cell, or ``None``
        when the reachable board is fully explored (or no move learned). RHAE-0 (pure BFS look-ahead);
        the per-search ``seen`` set is the intrinsic loop-prevention (always on)."""
        if (grid is None or avatar_pos is None or self.offsets is None
                or not self.nav_model.move_delta):
            return None
        start = tuple(avatar_pos)
        seen = {start}
        q: deque = deque([(start, [])])
        expand = 0
        while q and expand < 20000:
            pos, path = q.popleft()
            expand += 1
            if len(path) >= 120:
                continue
            for a, (dr, dc) in sorted(self.nav_model.move_delta.items()):
                npos = (pos[0] + dr, pos[1] + dc)
                if npos in seen:
                    continue
                if not self.nav_model.footprint_passable(grid, self.offsets, npos):
                    continue
                seen.add(npos)
                foot = {(npos[0] + ddr, npos[1] + ddc) for ddr, ddc in self.offsets}
                if foot - self._frontier_visited:          # covers a never-occupied cell -> frontier
                    return (path + [a])[0]
                q.append((npos, path + [a]))
        return None

    def _concrete_situation(
        self, avatar_pos: Optional[Tuple[int, int]], objects: Sequence[Obj]
    ) -> Optional[tuple]:
        """The «value» ConcreteSituation for the CURRENT frame: ``(controllable_pos, carried_pose,
        consumed_triggers)`` -- the search node AND the loop-prevention key (sheet 5; ADR-015).
        ``consumed_triggers`` is empty at a real frame boundary (triggers are consumed only
        WITHIN a BFS look-ahead). ``None`` when the controllable is not yet located."""
        if avatar_pos is None:
            return None
        carried = self._carried_pose(objects)
        return (tuple(avatar_pos), carried, frozenset())

    def _concrete_guardrail(
        self,
        concrete_move: Optional[int],
        grid: Optional[np.ndarray],
        avatar_pos: Optional[Tuple[int, int]],
    ) -> Optional[int]:
        """The ConcreteSituation visited/dead GUARDRAIL (ADR-014 / DP-20), GATED by
        ``_futility_check``. It records each frame's ConcreteSituation in ``_concrete_seen`` and,
        ONLY while goalless (no ``target_obj`` locked), suppresses a concrete move that would
        re-enter an already-visited concrete state -- catching a geometric EXPLORE loop of any
        period. It is a *safety valve*, NOT the mechanism: the PRIMARY loop-prevention is intrinsic
        (per-search ``seen`` + the ``poked`` give-up set + CALIBRATE round-robin), so L1 self-solves
        with this OFF (the guardrail-independence invariant, DP-20). It NEVER fires once SOLVE is
        active (a target is locked -> monotone progress to the win-set), so it cannot block the win.

        Returns the move to commit (possibly ``None`` to defer to the abstract fallback)."""
        current = self._concrete_situation(avatar_pos, self.last_objects)
        if current is not None:
            self._concrete_seen.add(current)
        if not self._futility_check or concrete_move is None or grid is None:
            return concrete_move
        if self.target_obj is not None or avatar_pos is None or self.offsets is None:
            return concrete_move                          # SOLVE active or no geometry -> never block
        delta = self.nav_model.move_delta.get(int(concrete_move))
        if delta is None:
            return concrete_move                          # an un-learned CALIBRATE probe -> allow
        npos = (avatar_pos[0] + delta[0], avatar_pos[1] + delta[1])
        # frontier exemption (DP-20: a guardrail must NEVER make solving worse than OFF): a move that
        # reaches a NEVER-visited footprint cell is productive exploration, not a futile loop -> allow it
        # (suppressing it strands the agent, the empirical coverage-8-ON vs 36-OFF regression).
        foot = {(npos[0] + dr, npos[1] + dc) for dr, dc in self.offsets}
        if foot - self._frontier_visited:
            return concrete_move
        nxt = (npos, self._carried_pose(self.last_objects), frozenset())
        if nxt in self._concrete_seen:
            return None                                   # would re-enter a visited concrete state
        return concrete_move

    # ------------------------------------------------------------------ (3-pre) projection
    def _profiles(
        self, objects: Sequence[Obj], assignment: RoleAssignment
    ) -> Mapping[Obj, A.Profile]:
        """A minimal, NFR-6-clean Profile per ROLED object: the recognised role axis only.

        We record just the role on a single ``role`` Dimension (the StateAbstraction's own
        default), so the AbstractSituation stays game-literal-free without a full vocabulary pass --
        the perceive Obj does not carry a Profile, and we never key salience on a colour."""
        profiles: Dict[Obj, A.Profile] = {}
        for obj, role in assignment.roles.items():
            profiles[obj] = A.Profile(entries={"role": (role, 1.0)})
        return profiles

    def _observe_relations(
        self, assignment: RoleAssignment, objects: Sequence[Obj]
    ) -> frozenset:
        """The salient role-keyed Relations that HOLD this frame (NFR-6: role labels only).

        Currently: ``overlaps(controllable, X)`` when the controllable's footprint overlaps a
        roled object, and ``inside(controllable, target)`` when it sits within the target's
        bbox. These are structural relation words over role labels -- never a coordinate."""
        rels: set = set()
        ctrl = assignment.controllable
        if ctrl is None:
            return frozenset()
        foot = set(ctrl.cells)
        for obj, role in assignment.roles.items():
            if obj is ctrl:
                continue
            if foot & obj.cells:
                rels.add(relation("overlaps", ROLE_CONTROLLABLE, role))
        tgt = assignment.target
        if tgt is not None and self._within_bbox(ctrl, tgt):
            rels.add(relation("inside", ROLE_CONTROLLABLE, ROLE_TARGET))
        return frozenset(rels)

    @staticmethod
    def _within_bbox(inner: Obj, outer: Obj) -> bool:
        """True iff ``inner``'s bbox sits within ``outer``'s bbox (a structural containment)."""
        ir0, ic0, ir1, ic1 = inner.bbox
        or0, oc0, or1, oc1 = outer.bbox
        return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1

    def _observe_scalars(
        self, frame_response: Any, objects: Sequence[Obj], assignment: RoleAssignment
    ) -> Mapping[str, object]:
        """Observed Markov scalar gauges. A move-budget gauge is read from the frame when the
        framework exposes one (an observation, never a baked literal); absent that, no scalar
        is invented. The projector decides the NAME (TERM-32)."""
        scalars: Dict[str, object] = {}
        budget = getattr(frame_response, "move_budget", None)
        if budget is not None:
            scalars[MOVE_BUDGET] = int(budget)
        return scalars

    # ------------------------------------------------------------------ (2c) refine world
    def _refine_world(
        self, prev: AbstractSituation, move: int, observed: AbstractSituation
    ) -> Tuple[bool, List[str]]:
        """Learn from the last observed transition (per-move feedback). Feeds the single
        ``(prev, move, observed)`` triple to ``ModelWorld.refine``; on the very first
        observation of a transition it records a direct rule so ``predict`` reproduces it.

        Returns ``(changed, learned_descriptions)`` for the TurnRecord world_model digest."""
        before_rules = len(self.world.rules)
        before_table = len(self.world._table)
        self.world.refine([(prev, move, observed)])
        # Make predict reproduce the observed transition deterministically: register a direct
        # InteractionRule mapping this prev (under this move) to the observed next, if the
        # model does not already predict it. This is the general analogue of L1 learn_move:
        # a learned local transition, keyed on the abstract AbstractSituation (NFR-6).
        if self.world.predict(prev, move).canonical() != observed.canonical():
            self.world.add_rule(_replay_rule(prev, move, observed))
        changed = (
            len(self.world.rules) != before_rules or len(self.world._table) != before_table
        )
        learned: List[str] = []
        if changed:
            learned.append("learned transition for move %d" % move)
        return changed, learned

    # ------------------------------------------------------------------ (6) diagnose + repair
    def _diagnose_and_repair(self, observed: AbstractSituation) -> RoutingVerdict:
        """Localize the first divergence (armed prediction vs observation) and route back to
        the wrong stage, updating it so the next plan CHANGES (the MPC repair, sheet 3).

        The repair per route:
          * EXPLORE -- an unknown mechanism: record the observed transition so the model now
            covers it (the next prediction will match), and DROP the armed prediction;
          * WORLD   -- a known rule mispredicted: override the learned transition with the
            observed next (the model is corrected);
          * GOAL    -- a goal-judgment error: clear the goal hypothesis so it is re-abduced;
          * PLAN    -- model + goal right, path wrong: nothing to learn -- the bounded planner
            re-plans from the new observation and the visited set avoids the same dead move.
        The router records the re-entry (SC-11 spy)."""
        verdict = diagnose_divergence(
            predicted=self.armed_prediction,
            observed=observed,
            world=self.world,
            goal=self.goal,
            prior=self.prev_situation,
            move=self.armed_move,
        )
        # Drive the agent back into the routed phase (the router records the re-entry).
        self.router.route(verdict)

        if verdict.phase in (Phase.EXPLORE, Phase.WORLD):
            # Correct/extend the model so it now reproduces the observed transition.
            if self.prev_situation is not None and self.armed_move is not None:
                self.world.add_rule(
                    _replay_rule(self.prev_situation, self.armed_move, observed)
                )
        elif verdict.phase == Phase.GOAL:
            # The goal test disagreed with reality -> re-abduce the goal next.
            self.goal = None
        # PLAN: no model/goal change; the planner re-plans below from ``observed``.
        return verdict

    # ------------------------------------------------------------------ (3) model goal
    def _update_goal(self, observed: AbstractSituation, assignment: RoleAssignment) -> None:
        """Maintain the goal hypothesis: refine from the first win (bootstrap), else hypothesize
        a provisional goal from the highest-confidence applicable prior (SC-08 / SC-09)."""
        if self.won_situation is not None and self.prev_situation is not None and self.goal is None:
            # First win observed: induce the instance-invariant goal from the win-diff.
            self.goal = self.model_goal.refine_from_win(self.prev_situation, self.won_situation)
            return
        if self.goal is None:
            hypothesis = self.model_goal.hypothesize(observed)
            if hypothesis is not None:
                self.goal = hypothesis

    # ------------------------------------------------------------------ (4) plan
    def _plan(self, observed: AbstractSituation, available: Sequence[int]) -> GamePlan:
        """RHAE-0 look-ahead over the built-in sim toward the current goal / roadmap step.

        Only ``ModelWorld.predict`` is exercised (the solver guarantees 0 scored moves); the
        moves are the env's ``available_actions`` lifted to opaque tokens. When a roadmap is
        set, plan toward its current (first-unmet) step; else toward the single goal; when
        there is no goal yet, the plan is empty (the controller falls back to a legal move)."""
        moves = self._move_tokens(available)
        if self.roadmap is not None:
            return self.planner.plan_roadmap(self.world, observed, self.roadmap, moves)
        if self.goal is None:
            return GamePlan(moves=())
        return self.planner.plan(self.world, observed, self.goal, moves)

    @staticmethod
    def _move_tokens(available: Sequence[int]) -> List[int]:
        """The planner's candidate moves: the env's available_actions as opaque int tokens,
        EXCLUDING the click (ACTION6=6) and reset (0) -- the look-ahead reasons over the simple
        movement repertoire; the click needs coordinates the abstract planner does not model."""
        return [int(a) for a in available if int(a) not in (0, 6)]

    # ------------------------------------------------------------------ (5) commit
    def _commit(
        self,
        game_plan: GamePlan,
        observed: AbstractSituation,
        available: Sequence[int],
        verdict: Optional[RoutingVerdict],
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        """Choose EXACTLY ONE legal action id to commit this turn (MPC), with coords for a
        click. Prefers the plan's first move; falls back to a deterministic legal move when the
        plan is empty / its head is not legal this turn, and AVOIDS repeating a diverging move
        (the MPC repair: after a divergence, the committed move must change)."""
        candidate = game_plan.first()
        avoid = self._avoid_move(verdict)

        if candidate is not None and int(candidate) in self._legal_simple(available):
            if avoid is None or int(candidate) != avoid:
                return int(candidate), None
        # Fallback: a deterministic legal move (smallest legal simple id), skipping the avoided
        # one when possible so a divergence forces a different move next turn.
        fallback = self._fallback_move(available, avoid)
        coords = (0, 0) if fallback == 6 else None
        return fallback, coords

    def _avoid_move(self, verdict: Optional[RoutingVerdict]) -> Optional[int]:
        """The move to avoid repeating this turn: the just-diverged move (so the next commit
        CHANGES after a divergence -- the MPC repair / no-thrash guard). Also avoids repeating
        a move that has produced a long futile run."""
        if verdict is not None and self.armed_move is not None:
            return int(self.armed_move)
        if (self._futility_check and self.goal is not None
                and self.detect_futility.futile_run() >= self.futile_run_limit
                and self.armed_move is not None):
            return int(self.armed_move)        # goalless EXPLORE reads every move 'futile' -> do NOT veto
        return None

    @staticmethod
    def _legal_simple(available: Sequence[int]) -> List[int]:
        """The legal SIMPLE movement ids (exclude reset=0 and click=6 -- a click needs coords;
        the plan tokens are simple moves)."""
        return [int(a) for a in available if int(a) not in (0, 6)]

    def _fallback_move(self, available: Sequence[int], avoid: Optional[int]) -> int:
        """A deterministic legal action when no plan move applies: the smallest legal simple id
        (skipping ``avoid`` if another legal one exists); else the smallest legal id at all
        (which may be the click/reset -- still a member of available_actions).

        ``available_actions`` is always a non-empty subset of the action space (API-01: the
        framework always offers at least RESET). An EMPTY ``available`` is a contract breach
        (no legal action exists), so we raise a clear ``ValueError`` rather than let an opaque
        ``IndexError`` leak from an empty-sequence index -- the controller commits exactly one
        action and must always have a legal one to commit."""
        ids = sorted(int(a) for a in available)
        if not ids:
            raise ValueError(
                "available_actions is empty: no legal action to commit (API-01 requires a "
                "non-empty available_actions)"
            )
        simple = [m for m in ids if m not in (0, 6)]
        if simple:
            for m in simple:
                if avoid is None or m != avoid:
                    return m
            return simple[0]
        # No simple move legal -> the smallest legal id (could be reset/click); guaranteed legal.
        return ids[0]

    # ------------------------------------------------------------------ win bookkeeping
    def _update_win(self, frame_response: Any, observed: AbstractSituation) -> None:
        """Read the env win signal off the frame (a ``state``/``is_win`` flag or a rising
        ``levels_completed``) and record the first win AbstractSituation for the goal bootstrap."""
        is_win = self._read_win_flag(frame_response)
        if is_win and not self.won:
            self.won = True
            self.won_situation = observed

    @staticmethod
    def _read_win_flag(frame_response: Any) -> bool:
        """Best-effort read of the env win signal from a framework response (or a stand-in).
        Accepts an explicit ``is_win`` bool, an Outcome-like ``state == 'win'``, or a
        ``GameState.WIN``-valued ``state``. Absent any signal -> not won."""
        flag = getattr(frame_response, "is_win", None)
        if isinstance(flag, bool):
            return flag
        state = getattr(frame_response, "state", None)
        if state is not None:
            name = getattr(state, "name", None)
            if name == "WIN":
                return True
            if isinstance(state, str) and state.lower() == "win":
                return True
        return False

    # ------------------------------------------------------------------ goal distance
    def _goal_distance(self, situation: AbstractSituation) -> object:
        """The injected SearchHeuristic ``AbstractSituation -> distance`` (API-05) DetectFutility and
        the planner rank by. Delegates to the current goal; a constant when no goal is set yet
        (so futility classification still has a total order)."""
        if self.goal is None:
            return 0
        return self.goal.distance(situation)

    # ------------------------------------------------------------------ (2) TurnRecord emit
    def _turn_record(
        self,
        observed: AbstractSituation,
        objects: Sequence[Obj],
        assignment: RoleAssignment,
        game_plan: GamePlan,
        committed_id: int,
        coords: Optional[Tuple[int, int]],
        move_effect: Optional[str],
        world_changed: bool,
        learned: List[str],
    ) -> dict:
        """Assemble one TurnRecord (trace-schema v1.0) carrying EVERY ``TURN_RECORD_KEYS`` field.

        Which cluster supplies each field:
          * ``game_objects`` <- perceive (Obj.cells / parts / a role-only Profile);
          * ``situation``    <- StateAbstraction (AbstractSituation.canonical -> hash + signature);
          * ``world_model``  <- ModelWorld (rule count + the per-move learned digest);
          * ``goal_predicate`` <- ModelGoal / GoalPredicate (predicate + describe + confidence);
          * ``game_plan``    <- solver (the GamePlan moves + horizon);
          * ``game_move``    <- the committed move id (+ a derived label);
          * ``move_effect``  <- DetectFutility (the played move's MoveEffect; ``null`` if none);
          * ``verbalize``    <- the describe() renders (world / goal / per-object names).
        """
        return {
            "turn": self.turn,
            "level": self.level,
            "capture_ref": {"file": self.capture_file, "row": self.turn},
            "game_objects": [self._object_record(o, assignment) for o in objects],
            "situation": {
                "hash": _stable_hash(observed.canonical()),
                "signature": _situation_signature(observed),
            },
            "world_model": {
                "rule_count": len(self.world.rules),
                "changed": bool(world_changed),
                "summary": _world_summary(self.world),
                "learned": list(learned),
            },
            "goal_predicate": self._goal_record(),
            "game_plan": self._plan_record(game_plan),
            "game_move": {"id": int(committed_id), "name": _move_name(committed_id)},
            "move_effect": self._effect_record(move_effect),
            "verbalize": self._verbalize_record(objects, assignment),
        }

    def _object_record(self, obj: Obj, assignment: RoleAssignment) -> dict:
        """One GameObject record (trace-schema): id / cells / parts / profile / role / name."""
        role = assignment.role_of(obj)
        profile = self._object_profile(obj, role)
        return {
            "id": _object_id(obj),
            "cells": [[int(r), int(c)] for r, c in sorted(obj.cells)],
            "parts": [self._object_record(p, assignment) for p in obj.parts],
            "profile": profile,
            "role": role,
            "name": A.render(A.Profile(entries=_profile_entries(obj, role)), self.lexicon)
            or None,
        }

    @staticmethod
    def _object_profile(obj: Obj, role: Optional[str]) -> dict:
        """The object's Dimension -> {value, confidence} map for the trace (observation-derived).
        Records the four foundational ARC axes (colour / shape-hash / size / orientation) as
        OBSERVED values -- they are descriptive trace fields, not matching keys (NFR-6)."""
        prof: dict = {
            "color": {"value": int(A.color(obj)), "confidence": 1.0},
            "size": {"value": int(A.size(obj)), "confidence": 1.0},
        }
        orient = A.orientation_index(obj)
        if orient is not None:
            prof["orientation"] = {"value": int(orient), "confidence": 0.8}
        return prof

    def _goal_record(self) -> Optional[dict]:
        """The goal_predicate record (``null`` before goal bootstrap -- AP-6)."""
        if self.goal is None:
            return None
        return {
            "predicate": self.goal.describe(),
            "confidence": _goal_confidence(self.goal, self.model_goal),
            "describe": self.goal.describe(),
        }

    @staticmethod
    def _plan_record(game_plan: GamePlan) -> Optional[dict]:
        """The game_plan record (``null`` for an empty plan)."""
        if game_plan.is_empty():
            return None
        return {
            "solver": "plan",
            "moves": [int(m) for m in game_plan.moves],
            "horizon": len(game_plan.moves),
        }

    @staticmethod
    def _effect_record(move_effect: Optional[str]) -> Optional[dict]:
        """The move_effect record (``null`` when no prior move's effect was observed)."""
        if move_effect is None:
            return None
        return {
            "value": move_effect,
            "futile": MoveEffect.futile(move_effect),
            "reason": None,
        }

    def _verbalize_record(self, objects: Sequence[Obj], assignment: RoleAssignment) -> dict:
        """The verbalize record: one-line NL renders of world / goal / each object (CMP-24)."""
        return {
            "world": _world_summary(self.world),
            "goal": self.goal.describe() if self.goal is not None else "goal not yet abduced",
            "objects": [
                _verbalize_object(o, assignment.role_of(o), self.lexicon) for o in objects
            ],
        }

    @staticmethod
    def _settled_grid(frames: Sequence[Any]) -> Optional[np.ndarray]:
        """The settled (last) grid as an int ndarray -- the state the turn settled into. Returns
        ``None`` for an empty frame list (graceful)."""
        if not frames:
            return None
        return np.asarray(frames[-1], dtype=int)


# ============================================================================ helpers
def _replay_rule(prev: AbstractSituation, move: int, observed: AbstractSituation) -> InteractionRule:
    """An :class:`InteractionRule` that reproduces the OBSERVED transition: it fires for
    ``move`` exactly when the AbstractSituation matches ``prev`` (by canonical identity) and yields
    ``observed`` (a learned local transition; the general analogue of L1 ``learn_move``).

    Keyed on ``AbstractSituation.canonical`` (DP-10: no builtin ``hash`` identity), game-literal-free
    (NFR-6: the move is an opaque token, the states are abstract AbstractSituations)."""
    prev_key = prev.canonical()

    def _applies(situation: AbstractSituation) -> bool:
        return situation.canonical() == prev_key

    def _effect(_situation: AbstractSituation) -> AbstractSituation:
        return observed

    return InteractionRule(
        move=int(move),
        applicability=_applies,
        effect=_effect,
        name="replay:%d" % int(move),
    )


def _profile_entries(obj: Obj, role: Optional[str]) -> dict:
    """Lexicon Dimension entries for naming an object via :func:`attributes.render`. Records
    the colour value on the ``colour`` dimension (the render resolves it to a word if the
    lexicon knows one; else it is omitted) -- a derived name, never a saved one."""
    entries: dict = {"colour": (int(obj.dom_color), 1.0)}
    return entries


def _object_id(obj: Obj) -> str:
    """A stable, deterministic id for an object WITHIN a frame, derived from its canonical cell
    set (sorted) -- NOT the builtin ``hash`` (DP-10). Good enough as a per-turn handle for the
    trace (cross-frame identity is the ObjectTracker's job, future cluster)."""
    return "o" + _stable_hash(tuple(sorted(obj.cells)))[:8]


def _stable_hash(value: object) -> str:
    """A deterministic, process-independent hex digest of a canonical value (DP-10). Uses a
    stable repr + a fixed FNV-1a fold, NOT the salted builtin ``hash()`` -- so the same content
    yields the same id across runs / processes (the trace must be reproducible)."""
    data = repr(value).encode("utf-8")
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def _situation_signature(situation: AbstractSituation) -> str:
    """A short, human-readable signature of a AbstractSituation's salient content (roles + scalars).
    Game-literal-free: it lists role LABELS and scalar NAMES/values (the values are observed
    gauges like move_budget, not colour/coordinate literals)."""
    roles = ";".join(sorted(situation.roles.keys()))
    scalars = ";".join("%s=%s" % (k, situation.scalars[k]) for k in sorted(situation.scalars))
    parts = [p for p in (roles, scalars) if p]
    return " | ".join(parts) if parts else "(empty)"


def _world_summary(world: ModelWorld) -> str:
    """A one-line digest of the learned world model (rule count + hidden state). Derived, never
    saved; game-literal-free."""
    n = len(world.rules)
    hidden = ("+" + ",".join(world.hidden)) if world.hidden else ""
    return "%d interaction rule%s learned%s" % (n, "" if n == 1 else "s", hidden)


def _move_name(move_id: int) -> str:
    """A derived label for a committed move id (the raw token; the learned/derived label is a
    future cluster, so we fall back to a stable token name -- trace-schema permits this)."""
    return "move_%d" % int(move_id)


def _goal_confidence(goal: GoalPredicate, model_goal: ModelGoal) -> float:
    """The goal hypothesis confidence for the trace: the source pattern's evolving confidence
    when the goal came from a prior, else a neutral induced-goal confidence."""
    src = getattr(goal, "source_pattern", None)
    if src is not None:
        try:
            return float(model_goal.patterns.confidence(src))
        except KeyError:
            return 0.5
    return 0.5


def _verbalize_object(obj: Obj, role: Optional[str], lexicon: A.Lexicon) -> str:
    """A one-line NL rendering of one object (its derived name + role)."""
    name = A.render(A.Profile(entries=_profile_entries(obj, role)), lexicon)
    label = name if name else ("colour-%d object" % int(obj.dom_color))
    role_phrase = (", %s" % role) if role else ""
    return "%s at %s%s" % (label, obj.pos, role_phrase)
