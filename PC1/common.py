"""
common.py — shared utilities for the PC1 pipeline.

Deliberately dependency-light. The only third-party imports are the official
`anthropic` and `openai` SDKs (and `pypinyin`, only in generate_stimuli.py).
Retry/backoff and progress reporting are hand-rolled below so there are no
hidden behaviours: everything that happens during a run is visible in this file.

Nothing here reads or logs your API keys beyond handing them to the SDK, which
reads them from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY).
"""

from __future__ import annotations
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths — everything is relative to this PC1 folder, so the scripts work no
# matter what directory you launch them from.
# ----------------------------------------------------------------------------
PC1_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PC1_DIR / "stimuli.json"
RAW_DIR = PC1_DIR / "data" / "raw"
CODED_DIR = PC1_DIR / "data" / "coded"
GOLD_DIR = PC1_DIR / "data" / "gold"
RESULTS_DIR = PC1_DIR / "results"


# ----------------------------------------------------------------------------
# Stimulus config
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    id: str
    arm: str            # void | real | romanized
    tone: str           # atonal | tonal
    length: int         # 2 | 4 | 8
    text: str           # the string fed to the model
    syllables: list[str]
    source_chars: str | None = None


@dataclass(frozen=True)
class Config:
    seed: int
    lengths: list[int]
    prompt_template: str
    cells: list[Cell]

    def prompt_for(self, cell: Cell) -> str:
        return self.prompt_template.format(stimulus=cell.text)


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [
        Cell(id=c["id"], arm=c["arm"], tone=c["tone"], length=c["length"],
             text=c["text"], syllables=c["syllables"],
             source_chars=c.get("source_chars"))
        for c in raw["cells"]
    ]
    return Config(seed=raw["seed"], lengths=raw["lengths"],
                  prompt_template=raw["prompt_template"], cells=cells)


# ----------------------------------------------------------------------------
# Hand-rolled retry with exponential backoff + jitter.
# Visible and simple: retry on any exception up to `attempts` times, sleeping
# min(cap, base * 2**k) plus a little randomness, then give up and re-raise.
# ----------------------------------------------------------------------------
def with_retry(fn, *, attempts: int = 6, base: float = 1.0, cap: float = 60.0):
    last = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — intentional: retry everything transient
            last = e
            if k == attempts - 1:
                break
            delay = min(cap, base * (2 ** k)) + random.uniform(0, base)
            time.sleep(delay)
    raise last  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Provider clients — one tiny class each, same .complete() signature.
# Synchronous on purpose: simpler to read and debug than async, and a 5,400-call
# run finishes comfortably overnight. Concurrency is handled by a thread pool in
# run.py if you want it; the client itself stays dead simple.
# ----------------------------------------------------------------------------
@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


# Default models. Override on the command line.
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"
DEFAULT_GPT_MODEL = "gpt-5.5"

# Approximate USD per 1M tokens (standard tier). Used ONLY for the --dry-run
# estimate; never affects billing. Edit if prices change.
PRICE_PER_MTOK = {
    "claude-opus-4-7":   {"in": 15.0, "out": 75.0},
    "claude-sonnet-4-6": {"in": 3.0,  "out": 15.0},
    "gpt-5.5":           {"in": 5.0,  "out": 30.0},
}


class ClaudeClient:
    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
        from anthropic import Anthropic
        self.model = model
        self._client = Anthropic()

    def complete(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 1.0) -> Completion:
        def call():
            return self._client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        resp = with_retry(call)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return Completion(text.strip(), resp.usage.input_tokens,
                          resp.usage.output_tokens, self.model, self.provider)


class GPTClient:
    provider = "openai"

    def __init__(self, model: str = DEFAULT_GPT_MODEL):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        from openai import OpenAI
        self.model = model
        self._client = OpenAI()

    def complete(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 1.0) -> Completion:
        def call():
            # Prefer the Responses API; fall back to chat.completions.
            try:
                r = self._client.responses.create(
                    model=self.model, input=prompt,
                    max_output_tokens=max_tokens, temperature=temperature,
                )
                return ("responses", r)
            except (AttributeError, TypeError):
                r = self._client.chat.completions.create(
                    model=self.model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return ("chat", r)
        kind, resp = with_retry(call)
        if kind == "responses":
            text = resp.output_text
            in_tok = getattr(resp.usage, "input_tokens", 0)
            out_tok = getattr(resp.usage, "output_tokens", 0)
        else:
            text = resp.choices[0].message.content or ""
            in_tok = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
        return Completion(text.strip(), in_tok, out_tok, self.model, self.provider)


def build_client(provider: str, model: str | None = None):
    if provider == "anthropic":
        return ClaudeClient(model or DEFAULT_CLAUDE_MODEL)
    if provider == "openai":
        return GPTClient(model or DEFAULT_GPT_MODEL)
    raise ValueError(f"Unknown provider: {provider!r}")


def estimate_cost(model: str, n_calls: int, avg_in: int = 200, avg_out: int = 150) -> float:
    p = PRICE_PER_MTOK.get(model)
    if not p:
        return float("nan")
    return n_calls * (avg_in / 1e6 * p["in"] + avg_out / 1e6 * p["out"])


# ----------------------------------------------------------------------------
# Tiny progress reporter (no tqdm). Prints a running count to stderr.
# ----------------------------------------------------------------------------
class Progress:
    def __init__(self, total: int, every: int = 25, label: str = ""):
        self.total = total
        self.every = every
        self.label = label
        self.n = 0
        self.errors = 0
        self.t0 = time.time()

    def tick(self, error: bool = False) -> None:
        self.n += 1
        if error:
            self.errors += 1
        if self.n % self.every == 0 or self.n == self.total:
            dt = time.time() - self.t0
            rate = self.n / dt if dt > 0 else 0
            sys.stderr.write(
                f"  {self.label}{self.n}/{self.total} "
                f"({self.errors} err, {rate:.1f}/s)\n")
            sys.stderr.flush()


def iter_jsonl(path: Path):
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)
