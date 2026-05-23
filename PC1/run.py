"""
run.py — run the trials.

For each cell in stimuli.json, query each model N times with the fixed prompt.
Responses are appended to data/raw/<provider>.jsonl with full metadata. A trial
is keyed by (provider, model, cell_id, trial_index); on re-run, trials already
present are skipped, so an interrupted run resumes without re-spending. This
resume behaviour is operational (crash recovery) ONLY — N is fixed in advance at
150/cell per the preregistration; do not use resume as optional stopping.

Concurrency uses a standard-library thread pool (no async, no extra deps).

Usage:
    python run.py --dry-run                  # cost estimate, no API calls
    python run.py --n 3                       # tiny wiring check
    python run.py --n 150                     # the registered run
    python run.py --n 150 --claude-model claude-sonnet-4-6   # cheaper bulk run
    python run.py --providers anthropic       # one provider only
"""

from __future__ import annotations
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (
    load_config, build_client, estimate_cost, Progress, iter_jsonl,
    RAW_DIR, DEFAULT_CLAUDE_MODEL, DEFAULT_GPT_MODEL, Config,
)


def trial_key(provider: str, model: str, cell_id: str, idx: int) -> str:
    return f"{provider}|{model}|{cell_id}|{idx}"


def load_done(path) -> set[str]:
    done: set[str] = set()
    for r in iter_jsonl(path):
        try:
            done.add(trial_key(r["provider"], r["model"], r["cell_id"], r["trial_index"]))
        except KeyError:
            pass
    return done


def run_provider(provider: str, model: str, cfg: Config, n: int,
                 workers: int, max_tokens: int, temperature: float) -> None:
    client = build_client(provider, model)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{provider}.jsonl"
    done = load_done(out_path)

    todo = [(cell, idx)
            for cell in cfg.cells
            for idx in range(n)
            if trial_key(provider, client.model, cell.id, idx) not in done]

    if not todo:
        print(f"[{provider}] all {len(cfg.cells) * n} trials already present — nothing to do.")
        return

    print(f"[{provider}/{client.model}] {len(todo)} trials to run "
          f"({len(done)} already done), {workers} workers")

    write_lock = threading.Lock()
    prog = Progress(len(todo), every=25, label=f"[{provider}] ")

    def do_one(cell, idx):
        prompt = cfg.prompt_for(cell)
        try:
            comp = client.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  ! {cell.id}#{idx}: {type(e).__name__}: {e}\n")
            prog.tick(error=True)
            return
        record = {
            "provider": provider, "model": client.model,
            "cell_id": cell.id, "arm": cell.arm, "tone": cell.tone,
            "length": cell.length, "stimulus": cell.text,
            "source_chars": cell.source_chars, "trial_index": idx,
            "prompt": prompt, "response": comp.text,
            "input_tokens": comp.input_tokens, "output_tokens": comp.output_tokens,
            "ts": time.time(),
        }
        with write_lock:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        prog.tick()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(do_one, cell, idx) for (cell, idx) in todo]
        for _ in as_completed(futures):
            pass

    print(f"[{provider}] done — {prog.n - prog.errors} ok, {prog.errors} errors")


def dry_run(cfg: Config, n: int, models: dict[str, str]) -> None:
    cells = len(cfg.cells)
    print("DRY RUN — no API calls.\n")
    print(f"  cells: {cells}    N/cell: {n}")
    total = 0
    for provider, model in models.items():
        calls = cells * n
        total += calls
        cost = estimate_cost(model, calls)
        print(f"  {provider:10s} {model:22s} {calls:6d} calls   ~${cost:6.2f} (standard tier)")
    print(f"  {'TOTAL':10s} {'':22s} {total:6d} calls")
    print("\n  Batch tier is typically ~50% of standard. Output is capped at --max-tokens.")
    print("\n  Sample prompts:")
    for c in cfg.cells[:3]:
        print(f'    [{c.id}] "{cfg.prompt_for(c)}"')


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PC1 convergence trials.")
    ap.add_argument("--n", type=int, default=150, help="trials per cell per model")
    ap.add_argument("--providers", default="anthropic,openai")
    ap.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    ap.add_argument("--gpt-model", default=DEFAULT_GPT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    chosen = [p.strip() for p in args.providers.split(",") if p.strip()]
    models = {}
    if "anthropic" in chosen:
        models["anthropic"] = args.claude_model
    if "openai" in chosen:
        models["openai"] = args.gpt_model

    if args.dry_run:
        dry_run(cfg, args.n, models)
        return

    for provider, model in models.items():
        try:
            run_provider(provider, model, cfg, args.n,
                         args.workers, args.max_tokens, args.temperature)
        except RuntimeError as e:
            sys.stderr.write(f"[{provider}] skipped: {e}\n")


if __name__ == "__main__":
    main()
