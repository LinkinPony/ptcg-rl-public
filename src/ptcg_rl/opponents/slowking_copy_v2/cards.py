"""Card catalog and immutable source inventory for Slowking Copy v2."""

from __future__ import annotations

from typing import Final

from ptcg_rl.opponents.slowking_copy import cards as v1

# Reuse the immutable v1 deck and engine-verified card identities.
BASIC_PSYCHIC: Final = v1.BASIC_PSYCHIC
BOOMERANG_ENERGY: Final = v1.BOOMERANG_ENERGY
TELEPATH_PSYCHIC_ENERGY: Final = v1.TELEPATH_PSYCHIC_ENERGY

CONKELDURR: Final = v1.CONKELDURR
FEZANDIPITI_EX: Final = v1.FEZANDIPITI_EX
KYUREM: Final = v1.KYUREM
SLOWPOKE: Final = v1.SLOWPOKE
SLOWKING: Final = v1.SLOWKING
SMOOCHUM: Final = v1.SMOOCHUM
LATIAS_EX: Final = v1.LATIAS_EX
ANNIHILAPE: Final = v1.ANNIHILAPE
MEGA_KANGASKHAN_EX: Final = v1.MEGA_KANGASKHAN_EX
MEOWTH_EX: Final = v1.MEOWTH_EX

SECRET_BOX: Final = v1.SECRET_BOX
NIGHT_STRETCHER: Final = v1.NIGHT_STRETCHER
ULTRA_BALL: Final = v1.ULTRA_BALL
WONDROUS_PATCH: Final = v1.WONDROUS_PATCH
POKE_PAD: Final = v1.POKE_PAD
COUNTER_GAIN: Final = v1.COUNTER_GAIN
CIPHERMANIAC: Final = v1.CIPHERMANIAC
LILLIES_DETERMINATION: Final = v1.LILLIES_DETERMINATION
ACADEMY_AT_NIGHT: Final = v1.ACADEMY_AT_NIGHT

CONKELDURR_TANTRUM: Final = v1.CONKELDURR_TANTRUM
GUTSY_SWING: Final = v1.GUTSY_SWING
TRIFROST: Final = v1.TRIFROST
SEEK_INSPIRATION: Final = v1.SEEK_INSPIRATION
SUPER_PSY_BOLT: Final = v1.SUPER_PSY_BOLT
DELIGHTFUL_KISS: Final = v1.DELIGHTFUL_KISS
ANNIHILAPE_TANTRUM: Final = v1.ANNIHILAPE_TANTRUM
DESTINED_FIGHT: Final = v1.DESTINED_FIGHT

SCRIPT_NAME: Final = "slowking_copy_v2"
OPPONENT_NAME: Final = "slowking_copy_v2"

DECK_COUNTS: Final = v1.DECK_COUNTS
DECK: Final = v1.DECK
DECK_DIGEST: Final = v1.DECK_DIGEST
PAYLOAD_POKEMON: Final = v1.PAYLOAD_POKEMON
ENERGY_CARDS: Final = v1.ENERGY_CARDS
NON_RULE_POKEMON: Final = v1.NON_RULE_POKEMON
RULE_BOX_POKEMON: Final = v1.RULE_BOX_POKEMON
POKEMON: Final = v1.POKEMON

# The v2 agent inherits prompt handling from v1, so the frozen inventory must
# include both packages and every shared helper imported by that implementation.
IMPLEMENTATION_FILES: Final = (
    "src/ptcg_rl/actions/selection.py",
    "src/ptcg_rl/engine/constants.py",
    "src/ptcg_rl/engine/runtime.py",
    "src/ptcg_rl/opponents/slowking_copy/__init__.py",
    "src/ptcg_rl/opponents/slowking_copy/agent.py",
    "src/ptcg_rl/opponents/slowking_copy/cards.py",
    "src/ptcg_rl/opponents/slowking_copy/tactics.py",
    "src/ptcg_rl/opponents/slowking_copy/view.py",
    "src/ptcg_rl/opponents/slowking_copy_v2/__init__.py",
    "src/ptcg_rl/opponents/slowking_copy_v2/agent.py",
    "src/ptcg_rl/opponents/slowking_copy_v2/cards.py",
    "src/ptcg_rl/opponents/slowking_copy_v2/tactics.py",
)
