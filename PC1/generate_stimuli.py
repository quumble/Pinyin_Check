"""
generate_stimuli.py — (re)generate PC1/stimuli.json, seed-locked & reproducible.

The committed stimuli.json is the source of truth for a run. This script exists
so the construction is fully auditable: run it and you get the same file back,
byte for byte. It is the ONLY script that needs `pypinyin`.

Design (see PREREGISTRATION.md):
  Arms
    void       phonotactically valid syllables that DO NOT exist in Mandarin
    real       real Mandarin syllables in a nonsense combination
    romanized  the original 8-character phrase, truncated and romanized
  Sub-arms   atonal / tonal
  Lengths    2, 4, 8

Usage:
    python generate_stimuli.py
"""

from __future__ import annotations
import json
import random
from pathlib import Path

from pypinyin import lazy_pinyin, Style
from pypinyin.constants import PINYIN_DICT

SEED = 21
LENGTHS = [2, 4, 8]
ORIG_CHARS = "湫隘飂戾豗隤杳莚"
PROMPT = "please describe {stimulus} as an imaginary creature"
OUT = Path(__file__).resolve().parent / "stimuli.json"

INITIALS = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x',
            'zh', 'ch', 'sh', 'r', 'z', 'c', 's']
FINALS = ['a', 'o', 'e', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'ong',
          'ia', 'ie', 'iao', 'ian', 'in', 'iang', 'ing', 'iong',
          'ua', 'uo', 'uai', 'uan', 'uang', 've', 'van']
TONE_MAP = {'a': 'āáǎà', 'e': 'ēéěè', 'i': 'īíǐì',
            'o': 'ōóǒò', 'u': 'ūúǔù', 'v': 'ǖǘǚǜ'}


def real_syllable_inventory() -> set[str]:
    """Every attested atonal Mandarin syllable, per pypinyin's dictionary."""
    out: set[str] = set()
    for codepoint in PINYIN_DICT:
        for syl in lazy_pinyin(chr(codepoint), style=Style.NORMAL):
            if syl.isalpha():
                out.add(syl)
    return out


def add_tone(syllable: str, rng: random.Random) -> str:
    """Place a random tone mark on the syllable's main vowel (simplified)."""
    for vowel in ['a', 'e', 'o', 'i', 'u', 'v']:
        if vowel in syllable:
            return syllable.replace(vowel, rng.choice(TONE_MAP[vowel]), 1)
    return syllable


def group(syllables: list[str]) -> str:
    """Join syllables; for length>=6 insert a comma at the midpoint (4+4 style)."""
    if len(syllables) >= 6:
        half = len(syllables) // 2
        return " ".join(syllables[:half]) + ", " + " ".join(syllables[half:])
    return " ".join(syllables)


def main() -> None:
    rng = random.Random(SEED)
    real = real_syllable_inventory()
    combos = [i + f for i in INITIALS for f in FINALS]
    real_combos = sorted({s for s in combos if s in real})
    void_combos = sorted({s for s in combos if s not in real and len(s) <= 5})

    orig_atonal = lazy_pinyin(ORIG_CHARS, style=Style.NORMAL)
    orig_tonal = lazy_pinyin(ORIG_CHARS, style=Style.TONE)

    cells = []

    def add(arm, tone, length, text, syls, src=None):
        cells.append({"id": f"{arm}_{tone}_L{length}", "arm": arm, "tone": tone,
                      "length": length, "text": text, "syllables": syls,
                      "source_chars": src})

    for arm, pool in [("void", void_combos), ("real", real_combos)]:
        for length in LENGTHS:
            syls = rng.sample(pool, length)
            add(arm, "atonal", length, group(syls), syls)
            toned = [add_tone(s, rng) for s in syls]
            add(arm, "tonal", length, group(toned), toned)

    for length in LENGTHS:
        a, t = orig_atonal[:length], orig_tonal[:length]
        add("romanized", "atonal", length, group(a), a, ORIG_CHARS[:length])
        add("romanized", "tonal", length, group(t), t, ORIG_CHARS[:length])

    config = {"seed": SEED, "lengths": LENGTHS,
              "prompt_template": PROMPT, "cells": cells}
    OUT.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.name} — {len(cells)} cells")
    for c in cells:
        src = f"   [{c['source_chars']}]" if c["source_chars"] else ""
        print(f"  {c['id']:24s}  {c['text']}{src}")


if __name__ == "__main__":
    main()
