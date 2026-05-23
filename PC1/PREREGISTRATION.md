# PC1 Preregistration

**Study:** Does semantic/phonaesthetic convergence in large language models
survive romanization of Chinese?
**Authors:** Bo & Claude
**Date registered:** prior to data collection
**Repository:** https://github.com/quumble/Pinyin_Check (folder `PC1/`)

This document is committed *before* any trials are run. Its purpose is to fix the
hypotheses, sample size, analysis, and the rule for trusting the automated coder
in advance, so that the confirmatory results cannot be shaped after seeing the
data. Anything not designated confirmatory below is exploratory.

---

## 1. Background

A prior study (*Semantic Convergence in Chinese Character Space*, Bo & Claude,
March 2026) found that independent LLM instances given a semantically-loaded
Chinese **character** string (湫隘飂戾，豗隤杳莚) converge on the same imaginary
creature. Chinese characters carry meaning, so that convergence could be
**semantic**. English "nonsense word" studies (quumble, scrunklopt) showed a
distinct **phonaesthetic** convergence on semantically empty input. The open
question, flagged explicitly in the prior paper (§7.2), is whether convergence
survives when Chinese is reduced to **pinyin**, which (in principle) decouples
sound from meaning. Pinyin is not truly empty — its syllables map onto real
Mandarin morphemes — so a strong model may re-attach meaning. PC1 is designed to
detect exactly that.

## 2. Design

A fixed-string, between-cell design. Each cell is one frozen stimulus string
presented to a model N times in independent, memoryless conversations.

- **Arms (3):**
  - `void` — phonotactically valid syllables that do **not** exist in Mandarin
    (the closest possible "Chinese quumble"; nothing to map to a morpheme).
  - `real` — real Mandarin syllables in a nonsense combination (maximally
    Chinese-sounding; invites re-semanticization).
  - `romanized` — the original 8-character phrase, romanized (same underlying
    "word" as the prior study, stripped to sound).
- **Sub-arms (2):** `atonal`, `tonal` (with tone marks).
- **Lengths (3):** 2, 4, 8 syllables.
- **Cells:** 3 × 2 × 3 = **18**. One fixed string per cell, frozen in
  `stimuli.json` (seed 21) and committed to the repository.
- **Models (2):** one Anthropic Claude model and one OpenAI GPT model
  (flagship by default; the bulk Claude run may use a cheaper Claude tier — model
  identity is logged with every trial). DeepSeek is deliberately excluded.
- **Prompt (fixed):** `please describe {stimulus} as an imaginary creature`.
- **Sampling:** temperature 1.0, output capped at 200 tokens.

## 3. Sample size and stopping rule

**N = 150 trials per cell per model**, fixed in advance. Total = 18 × 150 × 2 =
**5,400 trials**. There is **no optional stopping** and no interim analysis. The
runner's resume capability is for crash recovery only; it must not be used to
inspect partial data and decide whether to continue.

## 4. Hypotheses

### 4.1 Primary (confirmatory)

> **H1. The three arms (void, real, romanized) differ in convergence.**

This is the headline. It is intentionally directionless: the arms are constructed
to differ, and the existence of a difference is the robust, falsifiable claim.
Operationalized as a difference in the primary convergence statistic (§6) across
arms, pooled over models, tested with a non-parametric test across cells
(Kruskal–Wallis across the three arms; α = 0.05).

### 4.2 Directional sub-predictions (confirmatory, but subordinate to H1)

These are registered predictions about *direction*. Being wrong on direction does
not threaten H1; it is itself an informative result.

- **H1a.** `romanized` converges more tightly than `void`.
- **H1b.** `real` converges more tightly than `void`.
- **H1c.** Convergence increases with length: L8 > L4 > L2 (monotonic, within arm).

### 4.3 Secondary

> **H2. The romanized original (`romanized`, length 8) reproduces the *specific*
> creature from the character study**, not merely *a* tightly-converged creature.

Operationalized as mean prevalence of the six **emergent** features
(`collapses_forward`, `place_not_creature`, `warps_space`,
`mythological_framing`, `feeds_on_abstract`, `calmed_by_order`) in the
romanized-L8 cells. High emergent overlap = the creature survived romanization;
tight convergence with low emergent overlap = it converged on something *else*
(itself an interesting result). H2 is secondary precisely because it is the most
failure-prone claim and should not carry the headline.

### 4.4 Exploratory (not confirmatory)

Tonal vs. atonal effects; per-feature prevalence patterns; model-specific
differences (Claude vs. GPT); cross-model agreement on majority feature sets; any
serendipitous pattern. These will be clearly labeled "exploratory" in any writeup.

## 5. Coding and the judge-validation gate

Creature descriptions (always in English, even when the stimulus is pinyin) are
coded against a fixed 18-feature schema (`schema.py`) by an LLM judge. To guard
against the LLM-coding-LLM circularity:

- **Blind + shuffled:** the judge sees only response text, never the source model
  or arm; responses are shuffled before coding.
- **Cross-judge / inter-rater:** coding may be run with more than one judge model
  and agreement reported.
- **Human gold set:** a 50-item random subset is hand-coded by the researcher.

**Registered trust gate (committed in advance):** the automated coding is trusted
only if, on the 50-item gold set, **overall judge-vs-human accuracy ≥ 0.90 AND
per-feature Cohen's κ ≥ 0.60.** Any feature with κ < 0.60 is **dropped from the
confirmatory analysis** and reported separately as exploratory. If overall
accuracy < 0.90, the automated coding is not used for confirmatory claims and the
coding procedure is revised and re-validated before any confirmatory analysis.

## 6. Primary and alternative statistics (registered)

- **Primary — convergence:** for each cell, `mean over features of |p − 0.5| × 2`,
  where `p` is the fraction of trials showing the feature. Range 0 (every feature
  a coin-flip) to 1 (every feature unanimous).
- **Alternative — majority-set size:** the number of features present in ≥ 0.60 of
  trials in a cell (a throwback to earlier consensus-feature studies). Reported
  alongside the primary; not a substitute for it.

## 7. What would falsify / surprise us

- H1 null (arms do **not** differ): convergence is insensitive to whether the
  input has mappable morphemes — a strong, surprising result.
- H1a/H1b reversed (`void` converges *more* tightly than `real`/`romanized`):
  emptiness produces sharper attractors than partial meaning — would reframe PC2.
- H2 null with H1 confirmed: convergence survives romanization but lands on a
  *different* creature — the quumble "translates" to a new attractor.

## 8. Deviations

Any deviation from this plan will be recorded in the repository with its reason
and the date it was made.
