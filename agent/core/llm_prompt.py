"""Move-proposer prompt loader + per-turn render/narrow/validate (API-04).

The proposer prompt lives in an editable ``.md`` (``agent/assets/prompts/
prompt-template.md``) loaded at runtime, so the instruction text is the single
source of truth and reviewable as a normal document. This module loads + renders
it and builds the per-turn ENFORCED grammar -- it owns NO model and NO torch.

IMPORT-SAFE WITHOUT TORCH (load-bearing): this module must be importable with no
``torch`` / ``transformers`` / ``lm-format-enforcer`` present (the default-OFF
agent boots torch-free). It only does file IO + pure dict shaping; the model
lives in :mod:`agent.core.llm_qwen` (lazy torch there).

The load/capabilities/action_rules/render_messages/narrow_schema/validate logic
is ported VERBATIM from the proven stage-2 harness
``docs/llm-prompt-design/run_proposer_test.py`` (the 5/5 real-Qwen path), so the
wired agent renders byte-identically to what was validated on Kaggle.

Determinism (DP-10): pure file IO + dict building, no RNG, no builtin ``hash()``.
English / ASCII. Offline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

# Runtime home for the bundled prompt + schemas (bundled by
# build_practice_kernel.py via the agent/assets recursion).
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "prompts",
)
PROMPT_MD = os.path.join(_PROMPTS_DIR, "prompt-template.md")
OUTPUT_SCHEMA_JSON = os.path.join(_PROMPTS_DIR, "output-action-schema.json")

# --------------------------------------------------------------------------- #
# 1. Prompt loading (verbatim from run_proposer_test.load_prompt)
# --------------------------------------------------------------------------- #

_SYSTEM = re.compile(r"<!--[\s=]*SYSTEM[\s=]*-->", re.I)
_USER = re.compile(r"<!--[\s=]*USER[\s=]*-->", re.I)
_EXLIB = re.compile(r"<!--[\s=]*EXAMPLE LIBRARY[\s=]*-->", re.I)
_EXAMPLE = re.compile(r"<!--\s*example\s+capability=(\w+)\s*-->")
# Authoring/explanatory HTML comments inside the SYSTEM/USER bodies are NOT prompt
# content: they get sent to the model verbatim, and the USER comment even carries a
# literal ``{{OBSERVATION_JSON}}`` that render_messages then substitutes -- injecting
# the (large) observation a SECOND time. Strip them so the model sees the observation
# once and no meta-prose. (The EXAMPLE-LIBRARY markers live in their own section,
# parsed before this strip, so they are unaffected.)
_BODY_COMMENT = re.compile(r"<!--.*?-->", re.S)


@dataclass(frozen=True)
class PromptTemplate:
    """The parsed prompt: the static system body (with render hooks), the per-turn
    user body (with ``{{OBSERVATION_JSON}}``), and the capability-tagged examples."""

    system: str
    user: str
    examples: Dict[str, str]


def load_prompt(md_path: str = PROMPT_MD) -> PromptTemplate:
    """Parse ``prompt-template.md`` into (system, user, examples_by_capability).

    Mirrors ``run_proposer_test.load_prompt``: strip the leading YAML frontmatter,
    split the body on the SYSTEM/USER markers, then parse the trailing EXAMPLE
    LIBRARY into ``{capability: "IN: ...\\nOUT: ..."}``. Pure file IO (DP-10)."""
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    # strip leading frontmatter (--- ... ---)
    if text.lstrip().startswith("---"):
        text = text.split("---", 2)[2]
    sys_split = _SYSTEM.split(text, 1)
    body = sys_split[1] if len(sys_split) == 2 else text
    sys_part, rest = _USER.split(body, 1)
    user_part, exlib = (_EXLIB.split(rest, 1) + [""])[:2]
    # Drop authoring comments from the bodies the model actually receives (the USER
    # comment otherwise duplicates the whole observation -- see _BODY_COMMENT).
    sys_part = _BODY_COMMENT.sub("", sys_part)
    user_part = _BODY_COMMENT.sub("", user_part)
    examples: Dict[str, str] = {}
    parts = _EXAMPLE.split(exlib)
    # parts = [pre, cap1, block1, cap2, block2, ...]
    for i in range(1, len(parts) - 1, 2):
        examples[parts[i].strip()] = parts[i + 1].strip()
    return PromptTemplate(
        system=sys_part.strip(), user=user_part.strip(), examples=examples
    )


# --------------------------------------------------------------------------- #
# 2. Per-turn render (verbatim from run_proposer_test capabilities/action_rules/
#    render_messages)
# --------------------------------------------------------------------------- #

def capabilities(obs: Dict[str, Any]) -> List[str]:
    """Which EXAMPLE LIBRARY entries this turn's controls can exercise.

    a usable button -> ``move``; ``inputs.click`` -> ``click``; ``vocabulary`` ->
    ``propose``; always ``explore``. Deterministic order."""
    caps: List[str] = []
    inputs = obs.get("inputs", {})
    if inputs.get("buttons"):
        caps.append("move")
    if inputs.get("click"):
        caps.append("click")
    if obs.get("vocabulary"):
        caps.append("propose")
    caps.append("explore")  # always
    return caps


def action_rules(obs: Dict[str, Any]) -> str:
    """The relevant control rule(s) for ``{{ACTION_RULES}}`` (e.g. the click rule
    only when ``inputs.click``)."""
    if obs.get("inputs", {}).get("click"):
        return (
            '- To click, set move to {"click":{"row":R,"col":C}} using a '
            "target's position."
        )
    return ""


def render_messages(
    tpl: PromptTemplate, obs: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Render the system+user chat message pair for this turn (cap ~3 examples,
    injected ACTION_RULES, compact OBSERVATION_JSON). Mirrors
    ``run_proposer_test.render_messages``."""
    chosen = [tpl.examples[c] for c in capabilities(obs) if c in tpl.examples][:4]
    system = tpl.system.replace("{{EXAMPLES}}", "\n\n".join(chosen))
    system = system.replace("{{ACTION_RULES}}", action_rules(obs))
    obs_json = json.dumps(obs, ensure_ascii=False, separators=(",", ":"))
    user = tpl.user.replace("{{OBSERVATION_JSON}}", obs_json)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# 3. Per-turn schema narrowing (verbatim from run_proposer_test.narrow_schema)
# --------------------------------------------------------------------------- #

def narrow_schema(base: Dict[str, Any], obs: Dict[str, Any]) -> Dict[str, Any]:
    """Build the per-turn ENFORCED grammar: only ``goal_prediction + move`` (the
    proposals / confidence_nudges / note channels are advisory and validated
    post-parse, NOT constrained). Authored in the subset lm-format-enforcer
    0.10.9 supports: no null inside an enum and no list-typed fields -- nullable
    values use ``anyOf``, not ``type: [..., "null"]``.

    ``base`` (the full output-action-schema.json) is accepted for signature
    parity with the harness; the enforced grammar is rebuilt per turn from the
    observation so it stays in the lmfe subset regardless of the dev schema's
    draft-2020 features (``oneOf`` / list-typed nullables)."""
    button_names = [b["name"] for b in obs.get("inputs", {}).get("buttons", [])]
    click_ok = bool(obs.get("inputs", {}).get("click"))
    template_ids = [t["id"] for t in obs.get("goal_templates", [])]

    tmpl = (
        {"anyOf": [{"type": "string", "enum": template_ids}, {"type": "null"}]}
        if template_ids
        else {"type": "null"}
    )
    button_shape = {
        "type": "object",
        "additionalProperties": False,
        "required": ["button"],
        "properties": {"button": {"type": "string", "enum": button_names}},
    }
    click_shape = {
        "type": "object",
        "additionalProperties": False,
        "required": ["click"],
        "properties": {
            "click": {
                "type": "object",
                "additionalProperties": False,
                "required": ["row", "col"],
                "properties": {
                    "row": {"type": "integer", "minimum": 0, "maximum": 63},
                    "col": {"type": "integer", "minimum": 0, "maximum": 63},
                },
            }
        },
    }
    shapes = ([button_shape] if button_names else []) + (
        [click_shape] if click_ok else []
    )
    move = shapes[0] if len(shapes) == 1 else {"oneOf": shapes}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["goal_prediction", "move"],
        "properties": {
            "goal_prediction": {
                "type": "object",
                "additionalProperties": False,
                "required": ["template", "description", "confidence"],
                "properties": {
                    "template": tmpl,
                    "description": {"type": "string"},
                    "target": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
            },
            "move": move,
            # OPTIONAL free-form working-memory note (not required). Added to the
            # ENFORCED grammar so the constrained decoder PERMITS it -- with
            # additionalProperties False + required = [goal_prediction, move], an
            # undeclared `note` was grammatically impossible, so the proposer could
            # never write one. Validated/stored post-parse (advisory; never the move).
            "note": {"type": "string"},
        },
    }


def load_output_schema(path: str = OUTPUT_SCHEMA_JSON) -> Dict[str, Any]:
    """Load the full output-action-schema.json (dev contract) once. Pure file IO."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 4. Post-parse validation (verbatim from run_proposer_test.validate,
#    move/goal half)
# --------------------------------------------------------------------------- #

def validate(obj: Dict[str, Any], obs: Dict[str, Any]) -> List[str]:
    """Validate the parsed output's goal/move against the turn's observation:
    ``template`` in goal_templates ids + {null}; ``target`` in objects refs +
    {null}; ``confidence`` in low/medium/high; ``button`` in inputs names; a
    ``click`` move only when ``inputs.click``. Returns a list of error strings
    (empty = valid). Mirrors ``run_proposer_test.validate``."""
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["output is not a JSON object"]
    gp = obj.get("goal_prediction")
    if not isinstance(gp, dict):
        errs.append("missing goal_prediction")
    else:
        tids = [t["id"] for t in obs.get("goal_templates", [])] + [None]
        if gp.get("template") not in tids:
            errs.append(
                "template %r not in goal_templates+null" % gp.get("template")
            )
        if gp.get("confidence") not in ("low", "medium", "high"):
            errs.append(
                "confidence %r not in low/medium/high" % gp.get("confidence")
            )
        refs = [o["ref"] for o in obs.get("objects", [])] + [None]
        if gp.get("target") not in refs:
            errs.append("target %r not an objects ref / null" % gp.get("target"))
    mv = obj.get("move")
    if not isinstance(mv, dict):
        errs.append("missing move")
    elif "button" in mv:
        names = [b["name"] for b in obs.get("inputs", {}).get("buttons", [])]
        if mv["button"] not in names:
            errs.append("button %r not in inputs.buttons" % mv["button"])
    elif "click" in mv:
        if not obs.get("inputs", {}).get("click"):
            errs.append("click move but inputs.click is false")
    else:
        errs.append("move has neither button nor click")
    return errs
