"""Engine-backed forward-model utilities."""

from ptcg_rl.engine.coin import ManualCoinAnalysis, analyze_manual_coin_branches
from ptcg_rl.engine.effect_types import (
    CardRef,
    EffectSummary,
    HpChange,
)
from ptcg_rl.engine.effects import parse_effect_logs
from ptcg_rl.engine.feature_vectors import (
    DYNAMIC_EFFECT_FEATURE_NAMES,
    DYNAMIC_EFFECT_FEATURE_SIZE,
    DynamicEffectFeatureRow,
    build_dynamic_effect_feature_table,
)
from ptcg_rl.engine.forward_model import (
    ActionResolution,
    enumerate_select_actions,
    extract_dynamic_effect_features,
    resolve_action_once,
    resolve_candidate_actions,
)
from ptcg_rl.engine.session import BattleSession, HiddenInformation, SearchSession
from ptcg_rl.engine.vector_battle import (
    FinishedGame,
    VectorBattlePool,
    VectorGame,
)

__all__ = [
    "DYNAMIC_EFFECT_FEATURE_NAMES",
    "DYNAMIC_EFFECT_FEATURE_SIZE",
    "ActionResolution",
    "BattleSession",
    "CardRef",
    "DynamicEffectFeatureRow",
    "EffectSummary",
    "HiddenInformation",
    "HpChange",
    "ManualCoinAnalysis",
    "SearchSession",
    "FinishedGame",
    "VectorBattlePool",
    "VectorGame",
    "analyze_manual_coin_branches",
    "build_dynamic_effect_feature_table",
    "enumerate_select_actions",
    "extract_dynamic_effect_features",
    "parse_effect_logs",
    "resolve_action_once",
    "resolve_candidate_actions",
]
