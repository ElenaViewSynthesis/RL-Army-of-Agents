"""Torch-free GRPO/Dr.GRPO advantages and DAPO tied-group filtering.

build_batch takes aligned task_ids and rollout dicts (run_task's flat schema).
Each task must have a whole number of groups; incomplete groups are errors.
rollout_id defaults to its input index. Reward stats cover *all* submitted
rollouts; advantage stats cover the retained groups. Standard deviations use
population normalization, and turns/p95 is the nearest-rank percentile.
"""

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping


def _reward(item):
    value = item["reward/value"] if isinstance(item, Mapping) else item
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Rewards must be finite")
    return value


def group_advantages(rewards, adv_norm="std", eps=1e-6):
    if adv_norm not in ("std", "none") or not math.isfinite(eps) or eps <= 0:
        raise ValueError("adv_norm must be std|none and eps must be positive")
    values = [_reward(r) for r in rewards]
    if not values:
        return []
    mean = statistics.fmean(values)
    denom = statistics.pstdev(values) + eps if adv_norm == "std" else 1.0
    return [(r - mean) / denom for r in values]


def filter_groups(groups, enabled):
    kept, dropped = [], 0
    for group in groups:
        values = [_reward(r) for r in group]
        if not values:
            raise ValueError("Groups must not be empty")
        if enabled and min(values) == max(values):
            dropped += 1
        else:
            kept.append(group)
    return kept, dropped


def build_batch(task_ids, rollouts, cfg):
    task_ids, rollouts = list(task_ids), list(rollouts)
    if len(task_ids) != len(rollouts):
        raise ValueError("task_ids and rollouts must have the same length")
    by_task = defaultdict(list)
    ids = set()
    for i, (task_id, rollout) in enumerate(zip(task_ids, rollouts)):
        rid = rollout.get("rollout_id", str(i))
        if not isinstance(rid, (str, int)) or rid in ids:
            raise ValueError("rollout_id must be a unique string or integer")
        ids.add(rid)
        by_task[task_id].append({**rollout, "rollout_id": rid})
    groups = []
    for rows in by_task.values():
        if len(rows) % cfg.group_size:
            raise ValueError("Incomplete same-task group; collect cfg.group_size rollouts")
        groups.extend(rows[i:i + cfg.group_size] for i in range(0, len(rows), cfg.group_size))
    kept, dropped = filter_groups(groups, cfg.filter_tied_groups)
    pairs = []
    for group in kept:
        advantages = group_advantages([_reward(r) for r in group], cfg.adv_norm)
        pairs.extend((r["rollout_id"], a) for r, a in zip(group, advantages))
    advantages = [a for _, a in pairs]
    rewards = [_reward(r) for r in rollouts]
    turns = sorted(r["env/turns_used"] for r in rollouts if r.get("env/turns_used") is not None)
    n = len(rollouts)
    mean = lambda values: statistics.fmean(values) if values else 0.0
    stats = {
        "adv/frac_dropped": dropped / len(groups) if groups else 0.0,
        "adv/mean": mean(advantages),
        "adv/std": statistics.pstdev(advantages) if advantages else 0.0,
        "reward/mean": mean(rewards),
        "reward/frac_3": sum(r == 3 for r in rewards) / n if n else 0.0,
        "reward/frac_provenance_fail": sum(r.get("reward/reason") == "provenance" for r in rollouts) / n if n else 0.0,
        "reward/frac_max_turns": sum(r.get("reward/reason") == "max_turns" for r in rollouts) / n if n else 0.0,
        "reward/mean_overlong_penalty": mean([r.get("reward/overlong_penalty", 0.0) for r in rollouts]),
        "turns/mean": mean(turns),
        "turns/p95": float(turns[math.ceil(0.95 * len(turns)) - 1]) if turns else 0.0,
    }
    return pairs, stats
