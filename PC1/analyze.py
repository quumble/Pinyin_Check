"""
analyze.py — turn coded feature vectors into convergence statistics.

Primary statistic (registered): convergence = mean over features of
|p - 0.5| * 2, where p is the fraction of trials showing the feature in a cell.
0 = maximal uncertainty (every feature a coin-flip); 1 = every feature unanimous.

Alternative statistic (registered, a throwback to earlier consensus studies):
feature-set stability = the size of the majority feature set (features present in
>= threshold of trials), reported alongside the primary.

Primary hypothesis (confirmatory): the arms (void / real / romanized) differ in
convergence. Directional sub-predictions: romanized > void; real > void;
L8 > L4 > L2. Tonal vs atonal is exploratory.

Secondary hypothesis: romanized-L8 reproduces the ORIGINAL character creature,
measured by overlap on EMERGENT features (collapses_forward, place_not_creature,
warps_space, mythological_framing, feeds_on_abstract, calmed_by_order).

Outputs results/summary.json, summary.csv, report.md.

Usage:
    python analyze.py
    python analyze.py --threshold 0.6
"""

from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict

from common import CODED_DIR, RESULTS_DIR, iter_jsonl
from schema import FEATURE_KEYS, EMERGENT_KEYS


def load_coded() -> list[dict]:
    rows = list(iter_jsonl(CODED_DIR / "coded.jsonl"))
    if not rows:
        raise SystemExit("No coded data. Run code.py first.")
    return rows


def prevalence(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {k: 0.0 for k in FEATURE_KEYS}
    acc = {k: 0 for k in FEATURE_KEYS}
    for r in rows:
        for k in FEATURE_KEYS:
            acc[k] += int(r["features"].get(k, 0))
    return {k: acc[k] / n for k in FEATURE_KEYS}


def convergence(prev: dict[str, float]) -> float:
    """Primary statistic."""
    return sum(abs(p - 0.5) * 2 for p in prev.values()) / len(prev) if prev else 0.0


def majority_set(prev: dict[str, float], threshold: float) -> list[str]:
    """Alternative statistic support: the majority feature set."""
    return [k for k, p in prev.items() if p >= threshold]


def emergent_rate(prev: dict[str, float]) -> float:
    vals = [prev[k] for k in EMERGENT_KEYS]
    return sum(vals) / len(vals) if vals else 0.0


def analyze(threshold: float) -> None:
    rows = load_coded()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["model"], r["arm"], r["tone"], r["length"])].append(r)

    summary = []
    for (model, arm, tone, length), rs in sorted(by_cell.items()):
        prev = prevalence(rs)
        maj = majority_set(prev, threshold)
        summary.append({
            "model": model, "arm": arm, "tone": tone, "length": length,
            "n": len(rs),
            "convergence": round(convergence(prev), 4),         # primary
            "majority_set_size": len(maj),                      # alternative
            "emergent_rate": round(emergent_rate(prev), 4),
            "majority_features": maj,
            "prevalence": {k: round(v, 3) for k, v in prev.items()},
        })

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "arm", "tone", "length", "n", "convergence",
                    "majority_set_size", "emergent_rate"] + FEATURE_KEYS)
        for s in summary:
            w.writerow([s["model"], s["arm"], s["tone"], s["length"], s["n"],
                        s["convergence"], s["majority_set_size"], s["emergent_rate"]]
                       + [s["prevalence"][k] for k in FEATURE_KEYS])

    write_report(summary, threshold)
    print(f"Wrote results/summary.json, summary.csv, report.md ({len(summary)} cells)")
    print_headline(summary)


def write_report(summary: list[dict], threshold: float) -> None:
    L = ["# Pinyin Check — PC1 Results\n",
         f"Majority-set threshold: {threshold}. Primary statistic: convergence "
         "(mean distance-from-50%). Alternative: majority-set size.\n",
         "## Convergence by cell\n",
         "| model | arm | tone | len | n | convergence | maj-set | emergent |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['model']} | {s['arm']} | {s['tone']} | {s['length']} | "
                 f"{s['n']} | {s['convergence']:.3f} | {s['majority_set_size']} | "
                 f"{s['emergent_rate']:.3f} |")

    # Primary test: arm means at matched length/tone, averaged over model.
    L += ["\n## Primary: convergence by arm (averaged over model)\n",
          "| length | tone | void | real | romanized |",
          "|---|---|---|---|---|"]
    agg = defaultdict(list)
    for s in summary:
        agg[(s["length"], s["tone"], s["arm"])].append(s["convergence"])
    lengths = sorted({s["length"] for s in summary})
    tones = sorted({s["tone"] for s in summary})
    for length in lengths:
        for tone in tones:
            def m(arm):
                v = agg.get((length, tone, arm))
                return f"{sum(v)/len(v):.3f}" if v else "—"
            L.append(f"| {length} | {tone} | {m('void')} | {m('real')} | {m('romanized')} |")
    L.append("\n*Registered predictions: romanized > void; real > void; "
             "convergence rises with length. Direction is secondary to the "
             "primary claim that the arms differ at all.*\n")

    # Secondary test: romanized-L8 emergent overlap with the original creature.
    L += ["## Secondary: does romanized-L8 reproduce the original creature?\n",
          "Emergent-feature prevalence in romanized-L8 cells (overlap with the "
          "character-study creature):\n",
          "| model | tone | " + " | ".join(EMERGENT_KEYS) + " |",
          "|---|---|" + "|".join("---" for _ in EMERGENT_KEYS) + "|"]
    for s in summary:
        if s["arm"] == "romanized" and s["length"] == 8:
            cells = " | ".join(f"{s['prevalence'][k]:.2f}" for k in EMERGENT_KEYS)
            L.append(f"| {s['model']} | {s['tone']} | {cells} |")
    L.append("")

    (RESULTS_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")


def print_headline(summary: list[dict]) -> None:
    agg = defaultdict(list)
    for s in summary:
        agg[s["arm"]].append(s["convergence"])
    print("\nHeadline — mean convergence by arm:")
    for arm in ("void", "real", "romanized"):
        v = agg.get(arm, [])
        if v:
            print(f"  {arm:11s} {sum(v)/len(v):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze PC1 coded data.")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="prevalence threshold for the majority feature set")
    args = ap.parse_args()
    analyze(args.threshold)


if __name__ == "__main__":
    main()
