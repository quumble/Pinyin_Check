"""
schema.py — the feature taxonomy used to code creature descriptions.

Each feature is a binary present/absent judgment, derived from the consensus
tables of the original character study (Sections 4.1 and 5.1 of
"Semantic Convergence in Chinese Character Space"). The coder (an LLM judge)
emits one 0/1 value per feature per response.

EMERGENT features are the ones that arose across instances WITHOUT tracing to a
single character (e.g. "collapses forward", "feeds on something abstract").
These matter most: if a void or romanized stimulus reproduces an emergent
feature, that is strong evidence of a genuine attractor rather than surface
character-reading. The secondary hypothesis (does romanized-L8 reproduce the
ORIGINAL creature) is measured chiefly on emergent-feature overlap.
"""

from __future__ import annotations

# (key, human label, description shown to the judge, group, is_emergent)
FEATURES = [
    ("long_flat_body", "Long flat compressed body",
     "The creature has an elongated, flattened, eel-like, ribbon-like, or two-dimensional body.",
     "morphology", False),
    ("woven_fibrous_texture", "Woven / fibrous / reed texture",
     "Its body or surface is woven, matted, fibrous, reed-like, grassy, or thatched.",
     "morphology", False),
    ("tendrils_roots", "Tendrils / root-like limbs",
     "It has tendrils, roots, creeping grass-like appendages, or many thin trailing limbs.",
     "morphology", False),
    ("composed_of_ruins", "Composed of ruins / debris",
     "Its body incorporates masonry, beams, petrified wood, rubble, tiles, or built debris.",
     "morphology", False),

    ("narrow_confined", "Narrow / confined habitat",
     "It dwells in or embodies narrow, cramped, confined, gorge-like, or claustrophobic space.",
     "habitat", False),
    ("deep_stagnant_water", "Deep stagnant water",
     "It is associated with deep pools, black or stagnant water, marsh, or bog.",
     "habitat", False),
    ("dark_vanishing", "Dark / vanishing / low-visibility",
     "It is dark, dim, half-visible, dissolving at the edges, or shrouds its surroundings in gloom.",
     "habitat", False),

    ("spiraling_wind", "Spiraling / howling wind",
     "It is accompanied by spiraling, whistling, howling, keening, or vortex-like wind.",
     "sound_motion", False),
    ("crashing_sound", "Crashing / percussive sound",
     "It produces a crashing, percussive, or booming sound, like splitting rock or falling boulders.",
     "sound_motion", False),
    ("causes_collapse", "Causes collapse / erosion",
     "It causes stone to crumble, paths to collapse, or terrain to erode.",
     "sound_motion", False),
    ("collapses_forward", "Collapses forward (locomotion)",
     "It moves by collapsing or crumbling forward, landsliding, cascading, or re-forming as it goes.",
     "sound_motion", True),

    ("fierce_perverse", "Fierce / perverse temperament",
     "It is fierce, perverse, spiteful, contrary, or destructive (with or without malice).",
     "temperament", False),

    ("geological_scale", "Geological / titanic scale",
     "It is mountain-sized, hundreds of feet tall, horizon-spanning, or geological in scale.",
     "scale_ontology", False),
    ("place_not_creature", "A place/process, not an animal",
     "It is framed as a phenomenon, place, force, or process rather than a biological animal.",
     "scale_ontology", True),
    ("warps_space", "Warps space / geography",
     "It distorts space: paths narrow, distances change, or geography is revised.",
     "scale_ontology", True),
    ("mythological_framing", "Mythological / lore framing",
     "It is embedded in lore: forbidden texts, rituals, names, dynastic or legendary framing.",
     "scale_ontology", True),

    ("feeds_on_abstract", "Feeds on something abstract",
     "It feeds on or consumes an abstraction: certainty, meaning, significance, memory, history, direction, or time.",
     "emergent", True),
    ("calmed_by_order", "Can be calmed by order / clarity",
     "It can be repelled, calmed, or defeated by acts of order: straight paths, music, repair, or clear structure.",
     "emergent", True),
]

FEATURE_KEYS = [f[0] for f in FEATURES]
EMERGENT_KEYS = [f[0] for f in FEATURES if f[4]]


def feature_block_for_prompt() -> str:
    """Numbered feature list inserted into the judge prompt."""
    return "\n".join(f'{i}. "{key}" — {desc}'
                     for i, (key, _label, desc, _grp, _em) in enumerate(FEATURES, 1))
