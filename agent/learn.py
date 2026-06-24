"""[Use Case] Learning service: update a WorldModel from one observed transition.

Separated from the Entity (`world_model.py`) per the CA file split (F-09): this
Use-Case module orchestrates a learning update; the Entity holds the immutable data
and the pure algorithm. May import `arcengine` (it does not need to), so it lives in
the Use-Case layer.

The canonical signature is the method form `model.learn(before, action, after)`
(F-12). This service is the named, injectable entry point the policy calls so the
learning step is mockable/replaceable without the policy reaching into the Entity.
"""

from __future__ import annotations

from agent.segment import ObjectSet
from agent.state_graph import ActionKey
from agent.world_model import WorldModel

__all__ = ["learn_transition"]


def learn_transition(
    model: WorldModel, before: ObjectSet, action: ActionKey, after: ObjectSet
) -> WorldModel:
    """Return a NEW WorldModel updated from one (before, action, after) triple (FR-102).

    Pure delegation to the immutable Entity update (FR-115): no mutation of `model`,
    no I/O, O(objects × hypotheses) (NFR-102).
    """
    return model.learn(before, action, after)
