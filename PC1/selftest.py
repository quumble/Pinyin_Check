"""
selftest.py — verify the whole pipeline offline, with no API calls and no keys.

It swaps in mock Claude/GPT clients and a mock keyword-based judge, runs a small
end-to-end pass (run -> code -> analyze) into a temporary data area, checks the
outputs look sane, and cleans up. Use this to confirm your checkout works before
spending any money.

    python selftest.py
"""

from __future__ import annotations
import json
import random
import sys
import tempfile
from pathlib import Path

import common
from common import Completion
from schema import FEATURE_KEYS, EMERGENT_KEYS

# Three canned descriptions that share many features (so coding has signal).
SNIPPETS = [
    ("A long flat eel-like creature woven from marsh reeds in a narrow dark gorge "
     "above black stagnant pools. A spiraling wind howls; a percussive crash splits "
     "stone as paths collapse. It collapses forward, fierce and perverse."),
    ("A two-dimensional mat of fibrous reeds in a confined crevice, dark and "
     "vanishing at the edges, with whistling wind and a crashing boom. It erodes the "
     "path and crumbles forward. It feeds on certainty."),
    ("A geological phenomenon, not an animal: a titanic landslide of ruined masonry "
     "warping space so distances change. It howls and crashes and collapses forward. "
     "Embedded in old lore, it can be calmed by a straight path."),
]

KEYWORDS = {
    "long_flat_body": ["flat", "eel", "two-dimensional"],
    "woven_fibrous_texture": ["woven", "reed", "fibrous"],
    "tendrils_roots": ["tendril", "root"],
    "composed_of_ruins": ["masonry", "ruined", "beams", "rubble"],
    "narrow_confined": ["narrow", "confined", "crevice", "gorge"],
    "deep_stagnant_water": ["stagnant", "pools", "marsh"],
    "dark_vanishing": ["dark", "vanishing"],
    "spiraling_wind": ["spiraling", "whistling", "howl", "wind"],
    "crashing_sound": ["crash", "percussive", "boom"],
    "causes_collapse": ["collapse", "erode", "crumble"],
    "collapses_forward": ["collapses forward", "crumbles forward"],
    "fierce_perverse": ["fierce", "perverse"],
    "geological_scale": ["geological", "titanic"],
    "place_not_creature": ["phenomenon, not"],
    "warps_space": ["warping space", "distances change"],
    "mythological_framing": ["lore"],
    "feeds_on_abstract": ["feeds on", "certainty"],
    "calmed_by_order": ["calmed by", "straight path"],
}


class MockClient:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model
        self._r = random.Random(hash(provider) % 9973)

    def complete(self, prompt, max_tokens=200, temperature=1.0):
        return Completion(self._r.choice(SNIPPETS), 40, 55, self.model, self.provider)


class MockJudge:
    provider = "mock"
    model = "mock-judge"

    def complete(self, prompt, max_tokens=400, temperature=0.0):
        desc = prompt.split("<description>")[1].split("</description>")[0].lower()
        feats = {k: (1 if any(w in desc for w in KEYWORDS.get(k, [])) else 0)
                 for k in FEATURE_KEYS}
        return Completion(json.dumps(feats), 200, 120, self.model, self.provider)


def main() -> None:
    import run
    import code as codemod
    import analyze

    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw"; coded = tmp / "coded"; res = tmp / "results"
        raw.mkdir(); coded.mkdir(); res.mkdir()

        # Redirect all module paths to the temp area.
        for mod in (common, run, codemod, analyze):
            if hasattr(mod, "RAW_DIR"):
                mod.RAW_DIR = raw
            if hasattr(mod, "CODED_DIR"):
                mod.CODED_DIR = coded
            if hasattr(mod, "RESULTS_DIR"):
                mod.RESULTS_DIR = res
        common.RAW_DIR, common.CODED_DIR, common.RESULTS_DIR = raw, coded, res

        # Swap in mock clients.
        run.build_client = lambda p, m=None: MockClient(p, m or "mock-" + p)
        codemod.build_client = lambda p, m=None: MockJudge()

        cfg = common.load_config()
        if len(cfg.cells) != 18:
            failures.append(f"expected 18 cells, got {len(cfg.cells)}")

        # 1. run
        for provider in ("anthropic", "openai"):
            run.run_provider(provider, "mock-" + provider, cfg, n=5,
                             workers=8, max_tokens=200, temperature=1.0)
        raw_lines = sum(1 for _ in common.iter_jsonl(raw / "anthropic.jsonl")) + \
            sum(1 for _ in common.iter_jsonl(raw / "openai.jsonl"))
        expected = 18 * 5 * 2
        if raw_lines != expected:
            failures.append(f"expected {expected} raw rows, got {raw_lines}")

        # 2. resume: re-running should add nothing
        run.run_provider("anthropic", "mock-anthropic", cfg, n=5,
                         workers=8, max_tokens=200, temperature=1.0)
        raw_lines2 = sum(1 for _ in common.iter_jsonl(raw / "anthropic.jsonl"))
        if raw_lines2 != 18 * 5:
            failures.append("resume re-ran completed trials")

        # 3. code (blind)
        codemod.code_all("mock", "mock-judge", workers=8, seed=21)
        coded_rows = list(common.iter_jsonl(coded / "coded.jsonl"))
        if len(coded_rows) != expected:
            failures.append(f"expected {expected} coded rows, got {len(coded_rows)}")
        if coded_rows and set(coded_rows[0]["features"]) != set(FEATURE_KEYS):
            failures.append("coded feature keys do not match schema")

        # 4. analyze
        analyze.analyze(threshold=0.6)
        summary = json.loads((res / "summary.json").read_text(encoding="utf-8"))
        if len(summary) != 36:
            failures.append(f"expected 36 summary cells, got {len(summary)}")
        if not (res / "report.md").exists():
            failures.append("report.md not written")

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nSELFTEST PASSED — run/code/analyze wired correctly, resume works, "
          "schema matches, 36 cells summarized.")


if __name__ == "__main__":
    main()
