#!/usr/bin/env python3
"""Rank sensitivity sweep scenarios and emit conservative tuning recommendations.

This script reads tools/sensitivity_sweep_results.json and scores each scenario
against objective-specific metric envelopes and preferred target centers. It
keeps recommendations conservative by combining behavior score with parameter
change magnitude penalty.

Outputs:
    - tools/sensitivity_recommendation.json
    - tools/sensitivity_recommendation.md
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rov_sim


INPUT_PATH = os.path.join(ROOT, "tools", "sensitivity_sweep_results.json")
OUT_JSON = os.path.join(ROOT, "tools", "sensitivity_recommendation.json")
OUT_MD = os.path.join(ROOT, "tools", "sensitivity_recommendation.md")

METRICS = [
    "surge_max_speed_mps",
    "stop_distance_m",
    "stop_time_10pct_s",
    "yaw_steady_rate_deg_s",
    "heave_max_speed_mps",
]


def default_envelopes():
    # Conservative acceptance envelopes anchored around current expected behavior.
    return {
        "surge_max_speed_mps": [0.42, 0.46],
        "stop_distance_m": [0.50, 0.60],
        "stop_time_10pct_s": [3.30, 4.00],
        "yaw_steady_rate_deg_s": [0.90, 1.10],
        "heave_max_speed_mps": [-0.02, 0.02],
    }


def objective_specs(envelopes):
    mid = {k: (v[0] + v[1]) * 0.5 for k, v in envelopes.items()}
    return {
        "balanced": {
            "description": "Minimize overall deviation with equal priorities",
            "targets": dict(mid),
            "weights": {k: 1.0 for k in METRICS},
        },
        "agility": {
            "description": "Favor higher surge speed and yaw authority",
            "targets": {
                "surge_max_speed_mps": 0.455,
                "stop_distance_m": mid["stop_distance_m"],
                "stop_time_10pct_s": mid["stop_time_10pct_s"],
                "yaw_steady_rate_deg_s": 1.08,
                "heave_max_speed_mps": 0.0,
            },
            "weights": {
                "surge_max_speed_mps": 1.6,
                "stop_distance_m": 0.8,
                "stop_time_10pct_s": 0.8,
                "yaw_steady_rate_deg_s": 1.6,
                "heave_max_speed_mps": 0.6,
            },
        },
        "precision": {
            "description": "Favor tighter stopping and steadier heading",
            "targets": {
                "surge_max_speed_mps": mid["surge_max_speed_mps"],
                "stop_distance_m": 0.52,
                "stop_time_10pct_s": 3.45,
                "yaw_steady_rate_deg_s": 0.95,
                "heave_max_speed_mps": 0.0,
            },
            "weights": {
                "surge_max_speed_mps": 0.9,
                "stop_distance_m": 1.7,
                "stop_time_10pct_s": 1.4,
                "yaw_steady_rate_deg_s": 1.3,
                "heave_max_speed_mps": 0.7,
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze sensitivity sweep and recommend conservative tuning deltas")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to sensitivity_sweep_results.json")
    parser.add_argument("--out-json", default=OUT_JSON, help="Path for machine-readable recommendation")
    parser.add_argument("--out-md", default=OUT_MD, help="Path for markdown summary")
    parser.add_argument(
        "--objective",
        default="balanced",
        choices=["balanced", "agility", "precision"],
        help="Objective profile used for target centers and metric weights",
    )
    parser.add_argument(
        "--change-penalty-weight",
        type=float,
        default=0.25,
        help="Weight applied to parameter-change magnitude penalty",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.02,
        help="Minimum score improvement vs baseline required before recommending a non-baseline scenario",
    )
    parser.add_argument(
        "--inband-penalty-scale",
        type=float,
        default=0.35,
        help="Scale for distance-to-target penalty even when inside the envelope",
    )
    return parser.parse_args()


def baseline_constants_lookup():
    return {
        "LIN_DRAG_BODY": tuple(rov_sim.LIN_DRAG_BODY),
        "ADDED_MASS_BODY": tuple(rov_sim.ADDED_MASS_BODY),
        "THRUSTER_SPEED_LOSS_COEF": float(rov_sim.THRUSTER_SPEED_LOSS_COEF),
    }


def normalized_band_penalty(value, low, high):
    if value is None:
        return 1.0
    span = max(1e-9, high - low)
    if low <= value <= high:
        return 0.0
    if value < low:
        return (low - value) / span
    return (value - high) / span


def normalized_target_distance(value, target, low, high):
    if value is None:
        return 1.0
    half_span = max(1e-9, 0.5 * (high - low))
    return abs(value - target) / half_span


def scalar_relative_change(new_v, base_v):
    denom = max(abs(base_v), 1e-9)
    return abs(new_v - base_v) / denom


def tuple_relative_change(new_vals, base_vals):
    comps = []
    for new_v, base_v in zip(new_vals, base_vals):
        denom = max(abs(base_v), 1e-9)
        comps.append(abs(new_v - base_v) / denom)
    if not comps:
        return 0.0
    return math.sqrt(sum(x * x for x in comps) / len(comps))


def patch_change_magnitude(patches, baseline_values):
    if not patches:
        return 0.0
    magnitudes = []
    for key, new_val in patches.items():
        base_val = baseline_values.get(key)
        if base_val is None:
            continue
        if isinstance(new_val, (list, tuple)) and isinstance(base_val, (list, tuple, tuple)):
            magnitudes.append(tuple_relative_change(list(new_val), list(base_val)))
        else:
            magnitudes.append(scalar_relative_change(float(new_val), float(base_val)))
    if not magnitudes:
        return 0.0
    return sum(magnitudes) / len(magnitudes)


def score_row(row, envelopes, targets, weights, baseline_values, change_penalty_weight, inband_penalty_scale):
    metric_breakdown = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for metric, bounds in envelopes.items():
        low, high = bounds
        target = targets[metric]
        weight = float(weights.get(metric, 1.0))
        value = row.get(metric)

        band_penalty = normalized_band_penalty(value, low, high)
        target_distance = normalized_target_distance(value, target, low, high)
        # Outside envelope: penalize strongly. Inside envelope: smaller preference shaping.
        metric_penalty = band_penalty + inband_penalty_scale * target_distance

        metric_breakdown[metric] = {
            "value": value,
            "target": target,
            "bounds": [low, high],
            "weight": weight,
            "band_penalty": band_penalty,
            "target_distance": target_distance,
            "metric_penalty": metric_penalty,
        }

        weighted_sum += weight * metric_penalty
        weight_total += weight

    behavior_penalty = weighted_sum / max(1e-9, weight_total)
    change_penalty = patch_change_magnitude(row.get("patches", {}), baseline_values)
    total_score = behavior_penalty + change_penalty_weight * change_penalty

    return {
        "behavior_penalty": behavior_penalty,
        "change_penalty": change_penalty,
        "total_score": total_score,
        "per_metric": metric_breakdown,
    }


def choose_recommendation(ranked, min_improvement):
    if not ranked:
        return None

    baseline = None
    for row in ranked:
        if row.get("scenario") == "baseline":
            baseline = row
            break

    best = ranked[0]
    if baseline is None:
        return {
            "recommended_scenario": best["scenario"],
            "reason": "No baseline row found; selecting minimum score scenario",
        }

    improvement = baseline["score"]["total_score"] - best["score"]["total_score"]
    if best["scenario"] != "baseline" and improvement >= min_improvement:
        return {
            "recommended_scenario": best["scenario"],
            "reason": "Best non-baseline scenario improves total score beyond minimum threshold",
            "improvement_vs_baseline": improvement,
        }

    return {
        "recommended_scenario": "baseline",
        "reason": "No scenario improved enough over baseline under conservative threshold",
        "improvement_vs_baseline": improvement,
    }


def write_markdown(path, objective, envelopes, targets, weights, ranked, recommendation):
    lines = []
    lines.append("# Sensitivity Recommendation")
    lines.append("")
    lines.append(f"## Objective: {objective}")
    lines.append("")
    lines.append("## Metric Configuration")
    for metric in METRICS:
        bounds = envelopes[metric]
        lines.append(
            f"- {metric}: bounds=[{bounds[0]}, {bounds[1]}], "
            f"target={targets[metric]}, weight={weights[metric]}"
        )
    lines.append("")

    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | Scenario | Total | Behavior | Change |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    for idx, row in enumerate(ranked, start=1):
        score = row["score"]
        lines.append(f"| {idx} | {row['scenario']} | {score['total_score']:.5f} | {score['behavior_penalty']:.5f} | {score['change_penalty']:.5f} |")
    lines.append("")

    lines.append("## Recommendation")
    lines.append(f"- recommended_scenario: {recommendation['recommended_scenario']}")
    lines.append(f"- reason: {recommendation['reason']}")
    if "improvement_vs_baseline" in recommendation:
        lines.append(f"- improvement_vs_baseline: {recommendation['improvement_vs_baseline']:.5f}")

    selected = None
    for row in ranked:
        if row["scenario"] == recommendation["recommended_scenario"]:
            selected = row
            break

    if selected is not None:
        lines.append("")
        lines.append("### Recommended Patch")
        lines.append(f"- patches: {json.dumps(selected.get('patches', {}), sort_keys=True)}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = json.load(f)

    envelopes = default_envelopes()
    objectives = objective_specs(envelopes)
    profile = objectives[args.objective]
    targets = profile["targets"]
    weights = profile["weights"]
    baseline_values = baseline_constants_lookup()

    scored = []
    for row in rows:
        score = score_row(
            row,
            envelopes,
            targets,
            weights,
            baseline_values,
            args.change_penalty_weight,
            args.inband_penalty_scale,
        )
        enriched = dict(row)
        enriched["score"] = score
        scored.append(enriched)

    ranked = sorted(scored, key=lambda r: r["score"]["total_score"])
    recommendation = choose_recommendation(ranked, args.min_improvement)

    output = {
        "input": args.input,
        "objective": args.objective,
        "objective_description": profile["description"],
        "envelopes": envelopes,
        "targets": targets,
        "weights": weights,
        "change_penalty_weight": args.change_penalty_weight,
        "inband_penalty_scale": args.inband_penalty_scale,
        "min_improvement": args.min_improvement,
        "ranking": ranked,
        "recommendation": recommendation,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    write_markdown(args.out_md, args.objective, envelopes, targets, weights, ranked, recommendation)

    print("Sensitivity recommendation written:")
    print(f"  {args.out_json}")
    print(f"  {args.out_md}")
    print(f"Recommended scenario: {recommendation['recommended_scenario']}")


if __name__ == "__main__":
    main()