"""Generalist sequence architecture.

Concrete model components intentionally live in their submodules.  Keeping this
package initializer light prevents the sequence config discriminator from
eagerly importing the snapshot backbone during config parsing.
"""

from ptcg_rl.model.sequence.config import (
    GENERALIST_SEQUENCE_ARCHITECTURE,
    GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT,
    GENERALIST_SEQUENCE_V2_ARCHITECTURE,
    GENERALIST_SEQUENCE_V3_ARCHITECTURE,
    GeneralistSequenceConfig,
)

__all__ = [
    "GENERALIST_SEQUENCE_ARCHITECTURE",
    "GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT",
    "GENERALIST_SEQUENCE_V2_ARCHITECTURE",
    "GENERALIST_SEQUENCE_V3_ARCHITECTURE",
    "GeneralistSequenceConfig",
]
