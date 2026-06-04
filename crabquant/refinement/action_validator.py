"""
Semantic action validator — rejects impossible action/failure_mode combinations.

After the LLM returns an action and the classifier determines the failure mode,
this guard checks whether the action makes logical sense for the diagnosed
problem.  If not, it overrides the action with a sensible default and logs
the reason.

Rules are intentionally conservative: they only override when the proposed
action is *obviously* wrong (e.g. modifying params when the strategy produces
zero signals).  All other combinations pass through unchanged.

Called in the refinement loop AFTER classify_failure() and BEFORE the action
is recorded in history or fed to the next LLM turn.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# ── Semantic validation rules ────────────────────────────────────────────────
# Each rule is a tuple of (action, failure_mode, predicate, override, reason).
# The predicate receives total_trades and returns True when the rule applies.

_SEMANTIC_RULES = [
    # modify_params can't fix a strategy that produces no signals at all
    (
        "modify_params",
        "flat_signal",
        lambda total_trades: total_trades == 0,
        "change_entry_logic",
        "modify_params cannot fix flat_signal (0 trades) — params can't create signals where none exist; overriding to change_entry_logic",
    ),
    # modify_params can't fix a strategy with only 1-4 trades (too selective)
    (
        "modify_params",
        "too_few_trades",
        lambda total_trades: 1 <= total_trades <= 4,
        "change_entry_logic",
        "modify_params cannot fix too_few_trades (1-4 trades) — entry logic itself is too restrictive; overriding to change_entry_logic",
    ),
    # adding filters to a strategy with 0 signals makes it worse
    (
        "add_filter",
        "flat_signal",
        lambda total_trades: total_trades == 0,
        "change_entry_logic",
        "add_filter cannot fix flat_signal (0 trades) — adding filters to a strategy with no signals makes it worse; overriding to change_entry_logic",
    ),
    # need to rewrite, not tweak params, when the module fails to load
    (
        "modify_params",
        "module_load_failed",
        lambda total_trades: True,
        "novel",
        "modify_params cannot fix module_load_failed — the strategy has a structural import/syntax error; overriding to novel",
    ),
    # need to rewrite, not tweak params, when the backtest crashes
    (
        "modify_params",
        "backtest_crash",
        lambda total_trades: True,
        "novel",
        "modify_params cannot fix backtest_crash — the strategy has a runtime bug; overriding to novel",
    ),
    # adding filters to a strategy with 1-4 trades reduces them further
    (
        "add_filter",
        "too_few_trades",
        lambda total_trades: 1 <= total_trades <= 4,
        "change_entry_logic",
        "add_filter cannot fix too_few_trades (1-4 trades) — adding filters will reduce trades further; overriding to change_entry_logic",
    ),
    # exit logic doesn't matter when there are no entries at all
    (
        "change_exit_logic",
        "flat_signal",
        lambda total_trades: total_trades == 0,
        "change_entry_logic",
        "change_exit_logic cannot fix flat_signal (0 trades) — exit logic is irrelevant with 0 entries; overriding to change_entry_logic",
    ),
]


def validate_action_semantically(
    action: str,
    failure_mode: str,
    total_trades: int,
) -> Tuple[str, str]:
    """Check whether *action* makes sense for the diagnosed *failure_mode*.

    Rules are evaluated in order; the first matching rule wins.  If no rule
    matches, the original action is returned unchanged with an empty reason.

    Args:
        action: The (already string-normalised) action type, e.g.
            ``"modify_params"``.
        failure_mode: The failure mode from ``classify_failure()``, e.g.
            ``"flat_signal"``.
        total_trades: The number of trades in the backtest that produced
            *failure_mode*.

    Returns:
        A ``(validated_action, reason)`` tuple.  *reason* is non-empty only
        when the action was overridden.
    """
    for rule_action, rule_mode, predicate, override, reason in _SEMANTIC_RULES:
        if action == rule_action and failure_mode == rule_mode and predicate(total_trades):
            logger.info("Semantic validation override: %s", reason)
            return override, reason

    return action, ""
