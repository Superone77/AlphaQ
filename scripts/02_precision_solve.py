#!/usr/bin/env python3
"""
AlphaQ precision solving: read alpha CSV, solve MILP for bit allocation under bpp budget.
Outputs recipe CSV (name, bit_width) for 03_gptq_from_recipe.py and 04_eval_quantized.py.

Usage:
  python scripts/02_precision_solve.py --alpha_csv docs/alpha_qwen3_full.csv --output_dir docs/bit_recipes --bpp 3.5 --gamma 10.0
"""
import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pulp
except ImportError:
    pulp = None


def load_layer_stats_from_csv(csv_path: Path) -> dict:
    """Load name -> {alpha, variance} from CSV with columns name, alpha, variance (or layer, module_type)."""
    layer_stats = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("name", "").strip()
            if not name:
                continue
            try:
                a = float(row.get("alpha", "").strip() or "nan")
                v = row.get("variance", "").strip()
                v = float(v) if v and v.strip() else float("nan")
            except ValueError:
                continue
            if not math.isfinite(a):
                continue
            if not math.isfinite(v) or v <= 0:
                v = 1e-10
            layer_stats[name] = {"alpha": a, "variance": v}
    return layer_stats


def build_and_solve_milp(layer_stats_map: dict, candidate_bits: list, bpp_budget: float, gamma: float = 1.0) -> dict:
    """AlphaQ-style MILP: minimize sum_l sensitivity_l * variance_l * q(b); constraint: mean bit <= bpp_budget."""
    if pulp is None:
        raise ImportError("PuLP required. Install with: pip install pulp")
    E = len(layer_stats_map)
    if E == 0:
        return {}
    layer_names = sorted(layer_stats_map.keys())
    alphas = []
    variances = []
    for name in layer_names:
        stats = layer_stats_map[name]
        a = float(stats.get("alpha", float("nan")))
        v = float(stats.get("variance", float("nan")))
        if not math.isfinite(v):
            v = 1e-10
        alphas.append(a)
        variances.append(v)
    candidate_bits = [int(b) for b in candidate_bits]
    if any(b <= 0 for b in candidate_bits):
        raise ValueError("candidate_bits must be positive integers")
    min_bit = min(candidate_bits)
    if bpp_budget < min_bit:
        raise ValueError(f"bpp_budget {bpp_budget} < min_bit {min_bit}")
    alpha0 = statistics.median(alphas)
    eps = 1e-8
    sensitivities = [((alpha0 / max(a, eps)) ** gamma) for a in alphas]
    var_eps = 1e-12
    clamped_variances = [max(v, var_eps) for v in variances]
    q_b_scalar = {b: 2.0 ** (-2 * b) for b in candidate_bits}
    prob = pulp.LpProblem("AlphaQ_GlobalLayers", pulp.LpMinimize)
    x = {}
    for i in range(E):
        for b in candidate_bits:
            x[(i, b)] = pulp.LpVariable(f"x_{i}_{b}", lowBound=0, upBound=1, cat=pulp.LpBinary)
    obj_terms = []
    for i in range(E):
        s_l = sensitivities[i]
        v_l = clamped_variances[i]
        for b in candidate_bits:
            obj_terms.append(x[(i, b)] * (s_l * v_l * q_b_scalar[b]))
    prob += pulp.lpSum(obj_terms), "Total_Cost"
    for i in range(E):
        prob += pulp.lpSum([x[(i, b)] for b in candidate_bits]) == 1, f"one_bit_{i}"
    prob += pulp.lpSum([x[(i, b)] * b for i in range(E) for b in candidate_bits]) <= bpp_budget * E, "bit_budget"
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"MILP status={pulp.LpStatus[status]}. Try higher bpp_budget or different candidate_bits.")
    assignment = {}
    for i, layer_name in enumerate(layer_names):
        chosen_b = None
        for b in candidate_bits:
            if pulp.value(x[(i, b)]) >= 0.5:
                chosen_b = b
                break
        if chosen_b is None:
            chosen_b = min_bit
        assignment[layer_name] = chosen_b
    return assignment


def main():
    p = argparse.ArgumentParser(description="AlphaQ precision solve: MILP bit allocation")
    p.add_argument("--alpha_csv", type=str, required=True, help="CSV from 01_compute_alpha.py (name, alpha, variance)")
    p.add_argument("--output_dir", type=str, default="docs/bit_recipes", help="Directory for recipe CSV(s)")
    p.add_argument("--bpp", type=float, default=3.5, help="Average bits per parameter (e.g. 3.0, 3.5, 4.0)")
    p.add_argument("--gamma", type=float, default=10.0, help="Sensitivity exponent (alpha0/alpha)^gamma")
    p.add_argument("--candidate_bits", type=str, default="2,3,4", help="Comma-separated candidate bit widths")
    args = p.parse_args()

    alpha_path = Path(args.alpha_csv)
    if not alpha_path.is_absolute():
        alpha_path = ROOT / alpha_path
    if not alpha_path.exists():
        print(f"Error: alpha CSV not found: {alpha_path}")
        return 1

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_bits = [int(x.strip()) for x in args.candidate_bits.split(",")]
    layer_stats = load_layer_stats_from_csv(alpha_path)
    print(f"Loaded {len(layer_stats)} layers from {alpha_path}")

    assignment = build_and_solve_milp(
        layer_stats_map=layer_stats,
        candidate_bits=candidate_bits,
        bpp_budget=args.bpp,
        gamma=args.gamma,
    )
    out_csv = out_dir / f"qwen3_coder_next_gamma{args.gamma}_bpp{args.bpp:.1f}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "bit_width"])
        for name in sorted(assignment.keys()):
            w.writerow([name, assignment[name]])
    print(f"Saved recipe: {out_csv} ({len(assignment)} layers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
