"""
code.py — blind feature-coding of the creature descriptions, with a human
gold-set validator.

Why this design (the credibility argument, stated plainly):
  * BLIND + SHUFFLED: the judge sees only the response text, never which model
    produced it nor the stimulus arm. Responses are shuffled before coding. If a
    judge codes Claude- and GPT-generated text as equally convergent under blind
    conditions, the "shared-bias" objection to LLM-coding-LLM largely dissolves.
  * CROSS-JUDGE: you choose the judge with --judge; default differs from the
    subject where possible. Run it twice with different judges to report
    inter-rater agreement.
  * HUMAN GOLD SET: outputs are in English even when the stimulus is pinyin, so a
    human can hand-code a held-out subset. --build-gold writes a CSV to label;
    --validate reports judge-vs-human agreement (accuracy + Cohen's kappa).

Per the preregistration, the automated coding is trusted ONLY if, on a 50-item
gold set, overall accuracy >= 0.90 AND per-feature kappa >= 0.60. Features below
the kappa threshold are dropped from the confirmatory analysis and reported
separately.

Usage:
    python code.py --build-gold 50          # writes data/gold/gold_labels.csv
    #   ... hand-fill the 0/1 feature columns ...
    python code.py --validate               # judge-vs-human agreement
    python code.py                          # code everything (blind)
    python code.py --judge openai           # use GPT as the judge instead
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (
    build_client, Progress, iter_jsonl,
    RAW_DIR, CODED_DIR, GOLD_DIR,
)
from schema import FEATURE_KEYS, feature_block_for_prompt

JUDGE_INSTRUCTIONS = (
    "You are a careful annotator coding a short description of an imaginary "
    "creature. Decide each feature strictly from the text: PRESENT (1) or "
    "ABSENT (0). Do not infer beyond what the text states."
)


def judge_prompt(response_text: str) -> str:
    return (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        "Description:\n"
        f"<description>\n{response_text}\n</description>\n\n"
        "Code these features as 1 (present) or 0 (absent), judging only from the "
        "description above:\n\n"
        f"{feature_block_for_prompt()}\n\n"
        "Respond with a single JSON object mapping each feature key (the quoted "
        "string) to 0 or 1. No prose, no markdown, no code fences."
    )


def parse_judge(text: str) -> dict[str, int] | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        raw = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None
    return {k: (1 if raw.get(k, 0) in (1, "1", True, "true", "True") else 0)
            for k in FEATURE_KEYS}


def load_raw() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(RAW_DIR.glob("*.jsonl")):
        rows.extend(iter_jsonl(p))
    return rows


def resp_id(r: dict) -> str:
    return f"{r['provider']}|{r['model']}|{r['cell_id']}|{r['trial_index']}"


def code_texts(judge, texts: list[str], workers: int, max_tokens: int = 400) -> list[dict | None]:
    """Code a list of texts; returns parsed dicts (or None) in the SAME order."""
    results: list[dict | None] = [None] * len(texts)
    prog = Progress(len(texts), every=25, label="  code ")
    lock = threading.Lock()

    def one(i, txt):
        try:
            comp = judge.complete(judge_prompt(txt), max_tokens=max_tokens, temperature=0.0)
            parsed = parse_judge(comp.text)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  ! judge #{i}: {type(e).__name__}\n")
            parsed = None
        results[i] = parsed
        with lock:
            prog.tick(error=parsed is None)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, i, t) for i, t in enumerate(texts)]
        for _ in as_completed(futures):
            pass
    return results


# ---------------------------------------------------------------- gold set

def build_gold(n: int, seed: int) -> None:
    rows = load_raw()
    if not rows:
        sys.exit("No raw responses found. Run run.py first.")
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLD_DIR / "gold_labels.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["resp_id", "response"] + FEATURE_KEYS)
        for r in sample:
            w.writerow([resp_id(r), r["response"]] + ["" for _ in FEATURE_KEYS])
    print(f"Wrote {len(sample)} rows to {path}")
    print("Fill each feature column with 0 or 1, then: python code.py --validate")


def read_gold() -> list[dict]:
    path = GOLD_DIR / "gold_labels.csv"
    if not path.exists():
        sys.exit("No gold file. Run: python code.py --build-gold 50")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if all(row.get(k, "").strip() in ("0", "1") for k in FEATURE_KEYS):
                rows.append(row)
    if not rows:
        sys.exit("Gold file has no fully-labeled rows yet.")
    return rows


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def validate(judge_provider: str, judge_model: str | None, workers: int) -> None:
    gold = read_gold()
    judge = build_client(judge_provider, judge_model)
    coded = code_texts(judge, [g["response"] for g in gold], workers)

    print(f"\nJudge: {judge.provider}/{judge.model}   gold rows: {len(gold)}\n")
    print(f"{'feature':28s} {'acc':>6s} {'kappa':>7s}  {'gate':>5s}")
    print("-" * 52)
    overall_agree = overall_n = 0
    n_pass = 0
    for k in FEATURE_KEYS:
        h = [int(g[k]) for g, c in zip(gold, coded) if c is not None]
        j = [c[k] for c in coded if c is not None]
        if not h:
            continue
        acc = sum(x == y for x, y in zip(h, j)) / len(h)
        kappa = cohens_kappa(h, j)
        overall_agree += sum(x == y for x, y in zip(h, j))
        overall_n += len(h)
        gate = "ok" if kappa >= 0.60 else "DROP"
        n_pass += int(kappa >= 0.60)
        print(f"{k:28s} {acc:6.2f} {kappa:7.2f}  {gate:>5s}")
    print("-" * 52)
    overall_acc = overall_agree / max(overall_n, 1)
    print(f"{'OVERALL':28s} {overall_acc:6.2f}")
    print(f"\nRegistered gate: overall acc >= 0.90 AND per-feature kappa >= 0.60.")
    print(f"  overall accuracy: {overall_acc:.2f}  ->  "
          f"{'PASS' if overall_acc >= 0.90 else 'FAIL'}")
    print(f"  features passing kappa: {n_pass}/{len(FEATURE_KEYS)} "
          f"(failing ones are dropped from confirmatory analysis)")


# ---------------------------------------------------------------- full coding

def code_all(judge_provider: str, judge_model: str | None, workers: int, seed: int) -> None:
    rows = load_raw()
    if not rows:
        sys.exit("No raw responses found. Run run.py first.")

    # BLIND + SHUFFLE: code text detached from metadata, in shuffled order,
    # then re-associate by original index.
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    shuffled_texts = [rows[i]["response"] for i in order]

    judge = build_client(judge_provider, judge_model)
    print(f"Coding {len(rows)} responses blind with {judge.provider}/{judge.model} ...")
    coded_shuffled = code_texts(judge, shuffled_texts, workers)

    coded_by_index: dict[int, dict | None] = {}
    for pos, original_i in enumerate(order):
        coded_by_index[original_i] = coded_shuffled[pos]

    CODED_DIR.mkdir(parents=True, exist_ok=True)
    out = CODED_DIR / "coded.jsonl"
    ok = fail = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            feats = coded_by_index.get(i)
            if feats is None:
                fail += 1
                continue
            ok += 1
            fh.write(json.dumps({
                "resp_id": resp_id(r),
                "provider": r["provider"], "model": r["model"],
                "arm": r["arm"], "tone": r["tone"], "length": r["length"],
                "cell_id": r["cell_id"], "trial_index": r["trial_index"],
                "judge": f"{judge.provider}/{judge.model}",
                "features": feats,
            }, ensure_ascii=False) + "\n")
    print(f"Coded {ok} ok, {fail} failed -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind feature-coding + validation.")
    ap.add_argument("--build-gold", type=int, metavar="N",
                    help="sample N responses to a CSV for hand-labeling")
    ap.add_argument("--validate", action="store_true",
                    help="report judge-vs-human agreement on the gold set")
    ap.add_argument("--judge", default="anthropic", help="anthropic | openai")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    if args.build_gold:
        build_gold(args.build_gold, args.seed)
    elif args.validate:
        validate(args.judge, args.judge_model, args.workers)
    else:
        code_all(args.judge, args.judge_model, args.workers, args.seed)


if __name__ == "__main__":
    main()
