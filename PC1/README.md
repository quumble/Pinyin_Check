# PC1 — Does convergence survive romanization?

This is the first experiment (PC1) in the Pinyin_Check series. It is a follow-up
to *Semantic Convergence in Chinese Character Space* (Bo & Claude, March 2026),
which showed that independent LLM instances given a Chinese **character** string
converge on the same imaginary creature.

Chinese characters carry meaning, so that convergence might be **semantic**.
English nonsense-word studies (quumble, scrunklopt) showed a different,
**phonaesthetic** convergence on genuinely empty input. PC1 asks the question the
prior paper left open: **what happens when you strip Chinese to pinyin?** Pinyin
is supposed to be just sound — but its syllables still map onto real Mandarin
morphemes, so a strong model might quietly re-attach meaning. This experiment is
built to catch that.

The hypotheses, sample size, and analysis are fixed in advance in
[`PREREGISTRATION.md`](PREREGISTRATION.md). Read that first if you want to know
what we predicted before seeing any data.

## The design in one table

|              | `void`                              | `real`                          | `romanized`                       |
|--------------|-------------------------------------|---------------------------------|-----------------------------------|
| What it is   | syllables that obey Mandarin sound rules but **don't exist** | real Mandarin syllables in a **nonsense** order | the original 8-character phrase, **romanized** |
| Tests        | pure phonaesthetics (no morpheme)   | re-semanticization              | how much rode on the *characters* |

Each arm runs **atonal** and **tonal**, at lengths **2, 4, 8** syllables → 18
cells. One fixed string per cell, run 150× per model on Claude and GPT.

## Why these scripts look the way they do

This is a **flat folder of plain Python scripts**, on purpose. There is no
package to install, no hidden entry points, no build step. Every script says at
the top what it does, imports its siblings directly, and can be read top to
bottom. The only third-party libraries are the official `anthropic` and `openai`
SDKs, plus `pypinyin` (used by one script to build the stimuli). Retry/backoff
and progress reporting are hand-written in `common.py` so nothing important
happens inside a dependency you can't see.

```
PC1/
  PREREGISTRATION.md     hypotheses & analysis plan, fixed in advance
  README.md              this file
  requirements.txt       three dependencies, pinned loosely
  stimuli.json           the 18 frozen stimulus strings (seed 21)
  common.py              config loader + Claude/GPT clients + retry + progress
  schema.py              the 18-feature coding taxonomy
  generate_stimuli.py    regenerates stimuli.json (reproducible)
  run.py                 runs the trials -> data/raw/
  code.py                blind feature-coding + human gold-set validation -> data/coded/
  analyze.py             convergence statistics -> results/
  selftest.py            offline mock pipeline check (no API, no keys)
  data/raw/  data/coded/  data/gold/   (created as you go; .gitkeep tracked)
  results/
```

## Run it (Windows PowerShell)

You do not need a virtual environment. Install the three dependencies into
whatever Python you use and run the scripts directly.

```powershell
cd PC1
pip install -r requirements.txt

# keys for this session (they are read from the environment, never logged)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."

# 0. sanity-check the whole pipeline with fake data — no API calls, no keys needed
python selftest.py

# 1. see what a real run would cost, before spending anything
python run.py --dry-run

# 2. a tiny real run to confirm wiring and watch the spend
python run.py --n 3

# 3. the registered run (N=150/cell). Resumable if interrupted.
python run.py --n 150
#    cheaper Claude side, if you want it:
#    python run.py --n 150 --claude-model claude-sonnet-4-6

# 4. validate the coder against your own labels BEFORE trusting it
python code.py --build-gold 50         # writes data/gold/gold_labels.csv
#    ... open the CSV, fill each feature column with 0 or 1 ...
python code.py --validate              # must clear the registered gate

# 5. code everything (blind + shuffled)
python code.py

# 6. analyze
python analyze.py
```

## Cost

At ~200 input + ~150 output tokens per trial, the full 5,400-call run is roughly:

| Claude side          | GPT side | total (standard) | total (batch ~50%) |
|----------------------|----------|------------------|--------------------|
| Opus  (~$38)         | ~$15     | **~$53**         | **~$27**           |
| Sonnet (~$8)         | ~$15     | **~$23**         | **~$12**           |

`python run.py --dry-run` prints a live estimate against the prices in
`common.py`. The single biggest lever is the Claude model: Sonnet is ~5× cheaper
on the Anthropic side and may converge just as tightly for a task this simple.

## The honest caveats

- **Tone marks** in the generated stimuli are placed mechanically. The `void`
  arm doesn't care (emptiness is the point), but if the tonal `real`/`romanized`
  results get published, have a Mandarin-fluent reader check those specific
  strings first.
- **LLM coding LLM** is a real circularity. It is handled — blind coding, a
  cross-judge option, and a human-validated gold set with a pre-registered trust
  gate — but the gold-set step is not optional. Skipping it invalidates the
  confirmatory claims.
- **v1 = one fixed string per cell.** Whether the *arm type* converges in general
  (a fresh random string per trial) is left to a later experiment.

## License

Apache-2.0 (see repository root).
