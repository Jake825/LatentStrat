"""Audit-first numeric score-equation rules for match-breakdown training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from latentstrat.pretraining.match_breakdown.parse import AllianceBreakdownRow
from latentstrat.pretraining.match_breakdown.vectorize import SeasonSchema

RULE_STATUSES = frozenset({"required", "candidate", "disabled"})


@dataclass(frozen=True)
class LinearRule:
    name: str
    season: int
    status: str
    lhs: dict[str, float]
    rhs: dict[str, float]
    weight: float = 1.0
    tolerance: float = 1e-6
    min_pass_rate: float = 0.995
    notes: str = ""


@dataclass(frozen=True)
class ResolvedRules:
    enabled: tuple[LinearRule, ...]
    audit: dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def schema_sha256(schema: SeasonSchema) -> str:
    payload = json.dumps(schema.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def load_rule_file(
    path: str | Path,
    *,
    default_tolerance: float = 1e-6,
    default_min_pass_rate: float = 0.995,
) -> tuple[LinearRule, ...]:
    rule_path = Path(path)
    payload = yaml.safe_load(rule_path.read_text(encoding="utf-8")) or {}
    season = int(payload["season"])
    defaults = payload.get("defaults", {})
    rules = []
    for raw in payload.get("rules", []):
        if raw.get("kind", "linear_equality") != "linear_equality":
            raise ValueError(f"Rule {raw.get('name')!r} must use kind='linear_equality'.")
        status = str(raw.get("status", "candidate"))
        if status not in RULE_STATUSES:
            raise ValueError(f"Rule {raw.get('name')!r} has unsupported status {status!r}.")
        rules.append(
            LinearRule(
                name=str(raw["name"]),
                season=season,
                status=status,
                lhs={str(key): float(value) for key, value in raw["lhs"].items()},
                rhs={str(key): float(value) for key, value in raw["rhs"].items()},
                weight=float(raw.get("weight", defaults.get("weight", 1.0))),
                tolerance=float(raw.get("tolerance", defaults.get("tolerance", default_tolerance))),
                min_pass_rate=float(
                    raw.get(
                        "min_pass_rate",
                        defaults.get("min_pass_rate", default_min_pass_rate),
                    )
                ),
                notes=str(raw.get("notes", "")),
            )
        )
    return tuple(rules)


def load_rules_by_season(
    rule_dir: str | Path,
    seasons: set[int] | list[int] | tuple[int, ...],
    *,
    default_tolerance: float = 1e-6,
    default_min_pass_rate: float = 0.995,
) -> tuple[dict[int, tuple[LinearRule, ...]], dict[int, str]]:
    root = Path(rule_dir)
    rules_by_season = {}
    hashes = {}
    for season in sorted(seasons):
        path = root / f"rules_{season}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Missing score-consistency rule file: {path}")
        rules_by_season[season] = load_rule_file(
            path,
            default_tolerance=default_tolerance,
            default_min_pass_rate=default_min_pass_rate,
        )
        hashes[season] = _sha256_file(path)
    return rules_by_season, hashes


def numeric_field_to_index(schema: SeasonSchema) -> dict[str, int]:
    return {
        field.source_path: index
        for index, field in enumerate(schema.encoded_fields)
        if field.kind == "numeric"
    }


def _validate_enabled_rule(rule: LinearRule, schema: SeasonSchema) -> dict[str, int]:
    indices = numeric_field_to_index(schema)
    missing = sorted((set(rule.lhs) | set(rule.rhs)) - set(indices))
    if missing:
        raise ValueError(
            f"Season {rule.season} rule {rule.name!r} references non-numeric or missing "
            f"schema fields: {', '.join(missing)}."
        )
    return indices


def _rule_residual_numpy(
    values: np.ndarray, rule: LinearRule, field_to_index: dict[str, int]
) -> np.ndarray:
    lhs = np.zeros(len(values), dtype=np.float64)
    rhs = np.zeros(len(values), dtype=np.float64)
    for field, coefficient in rule.lhs.items():
        lhs += coefficient * values[:, field_to_index[field]]
    for field, coefficient in rule.rhs.items():
        rhs += coefficient * values[:, field_to_index[field]]
    return lhs - rhs


def audit_rules(
    *,
    season: int,
    schema: SeasonSchema,
    rows: tuple[AllianceBreakdownRow, ...],
    values: Tensor,
    masks: Tensor,
    indices: np.ndarray,
    rules: tuple[LinearRule, ...],
    rule_file_hash: str,
    scope: str,
) -> ResolvedRules:
    """Promote passing candidates and abort when required rules drift."""

    selected_values = values[indices].numpy()
    selected_masks = masks[indices].numpy()
    enabled = []
    audits = []
    for rule in rules:
        entry: dict[str, Any] = asdict(rule)
        if rule.status == "disabled":
            entry.update({"enabled": False, "reason": "tracked-disabled-hypothesis"})
            audits.append(entry)
            continue
        field_to_index = _validate_enabled_rule(rule, schema)
        rule_indices = [field_to_index[field] for field in (*rule.lhs, *rule.rhs)]
        observed = np.all(selected_masks[:, rule_indices] > 0, axis=1)
        observed_values = selected_values[observed]
        residual = _rule_residual_numpy(observed_values, rule, field_to_index)
        failures = np.abs(residual) > rule.tolerance
        pass_rate = float(np.mean(~failures)) if len(residual) else 0.0
        failure_positions = np.flatnonzero(observed)[failures][:5]
        entry.update(
            {
                "enabled": pass_rate >= rule.min_pass_rate,
                "rows_audited": int(len(residual)),
                "failing_rows": int(np.sum(failures)),
                "pass_rate": pass_rate,
                "mean_abs_residual": float(np.mean(np.abs(residual))) if len(residual) else None,
                "max_abs_residual": float(np.max(np.abs(residual))) if len(residual) else None,
                "failure_examples": [
                    rows[int(indices[position])].row_id for position in failure_positions
                ],
            }
        )
        if entry["enabled"]:
            enabled.append(rule)
        else:
            entry["reason"] = "raw-audit-below-min-pass-rate"
            if rule.status == "required":
                raise ValueError(
                    f"Required season {season} rule {rule.name!r} passed "
                    f"{pass_rate:.6f}, below {rule.min_pass_rate:.6f}."
                )
        audits.append(entry)
    return ResolvedRules(
        enabled=tuple(enabled),
        audit={
            "season": season,
            "scope": scope,
            "schema_sha256": schema_sha256(schema),
            "rule_file_sha256": rule_file_hash,
            "enabled_rules": [rule.name for rule in enabled],
            "rules": audits,
        },
    )


def _linear_expr(pred: Tensor, terms: dict[str, float], field_to_index: dict[str, int]) -> Tensor:
    result = pred[:, 0] * 0
    for field, coefficient in terms.items():
        result = result + coefficient * pred[:, field_to_index[field]]
    return result


def score_consistency_loss(
    pred: Tensor, rules: tuple[LinearRule, ...] | list[LinearRule], field_to_index: dict[str, int]
) -> Tensor:
    losses = []
    for rule in rules:
        residual = _linear_expr(pred, rule.lhs, field_to_index) - _linear_expr(
            pred, rule.rhs, field_to_index
        )
        losses.append(rule.weight * F.smooth_l1_loss(residual, torch.zeros_like(residual)))
    if not losses:
        return pred.sum() * 0
    return torch.stack(losses).mean()


def score_consistency_report(
    pred: Tensor, rules: tuple[LinearRule, ...] | list[LinearRule], field_to_index: dict[str, int]
) -> dict[str, Any]:
    by_rule = {}
    all_residuals = []
    for rule in rules:
        residual = (
            _linear_expr(pred, rule.lhs, field_to_index)
            - _linear_expr(pred, rule.rhs, field_to_index)
        ).detach()
        absolute = torch.abs(residual)
        all_residuals.append(absolute)
        by_rule[rule.name] = {
            "mean_abs_residual": float(absolute.mean().cpu()),
            "max_abs_residual": float(absolute.max().cpu()),
        }
    if not all_residuals:
        return {"mean_abs_residual": None, "max_abs_residual": None, "by_rule": {}}
    combined = torch.cat(all_residuals)
    return {
        "mean_abs_residual": float(combined.mean().cpu()),
        "max_abs_residual": float(combined.max().cpu()),
        "by_rule": by_rule,
    }


__all__ = [
    "LinearRule",
    "ResolvedRules",
    "audit_rules",
    "load_rule_file",
    "load_rules_by_season",
    "numeric_field_to_index",
    "schema_sha256",
    "score_consistency_loss",
    "score_consistency_report",
]
