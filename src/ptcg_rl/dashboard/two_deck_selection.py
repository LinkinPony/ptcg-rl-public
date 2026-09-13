"""Read-only service for the two-submission deck recommendation."""

from __future__ import annotations

from ptcg_rl.dashboard.public_environment import PublicEnvironmentService
from ptcg_rl.dashboard.public_environment_models import EnvironmentWindow
from ptcg_rl.dashboard.repository import DashboardRepository
from ptcg_rl.dashboard.training_deck_strength import TrainingDeckStrengthService
from ptcg_rl.dashboard.training_deck_strength_models import TrainingEvidenceRange
from ptcg_rl.dashboard.two_deck_selection_analysis import (
    build_two_deck_selection,
)
from ptcg_rl.dashboard.two_deck_selection_models import TwoDeckSelectionPayload
from ptcg_rl.dashboard.workbench_models import CheckpointInfo


class TwoDeckSelectionService:
    """Compose immutable read-only evidence into a two-slot recommendation."""

    def __init__(
        self,
        repository: DashboardRepository,
        training: TrainingDeckStrengthService,
        public_environment: PublicEnvironmentService,
    ) -> None:
        self.repository = repository
        self.training = training
        self.public_environment = public_environment
        self._cache: dict[tuple[object, ...], TwoDeckSelectionPayload] = {}

    def recommend(
        self,
        run: str,
        *,
        checkpoint: CheckpointInfo | None,
        training_range: TrainingEvidenceRange,
        window_days: EnvironmentWindow,
    ) -> TwoDeckSelectionPayload:
        """Return a cached shared-scenario posterior without launching work."""
        for _attempt in range(3):
            revision = self.repository.training_history_revision(run)
            training = self.training.standings(run, checkpoint=checkpoint)
            public = self.public_environment.summary(
                run,
                window_days=window_days,
                checkpoint_version=None,
            )
            cache_key = (
                *revision,
                None if checkpoint is None else checkpoint.pair_manifest_sha256,
                training_range,
                window_days,
                public.snapshot_fingerprint,
                public.generated_at_utc,
                public.as_of_date,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            matrices = tuple(
                self.training.matrix(
                    run,
                    range_name=training_range,
                    checkpoint_version=(
                        None if checkpoint is None else checkpoint.version
                    ),
                    controller="all",
                    candidate_seat=seat,
                    opponent_set="all",
                )
                for seat in (0, 1)
            )
            if self.repository.training_history_revision(run) != revision:
                continue
            result = build_two_deck_selection(
                training=training,
                matrices=matrices,
                public=public,
                checkpoint=checkpoint,
                training_range=training_range,
                window_days=window_days,
                seed_material="|".join(
                    str(item)
                    for item in (
                        training.run_id,
                        None if checkpoint is None else checkpoint.pair_manifest_sha256,
                        training_range,
                        training.ranges[training_range].metadata.started_at_utc,
                        training.ranges[training_range].metadata.ended_at_utc,
                        public.snapshot_fingerprint,
                    )
                ),
            )
            if len(self._cache) >= 32:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
            return result
        raise RuntimeError("training history changed during portfolio composition")


__all__ = ["TwoDeckSelectionService", "build_two_deck_selection"]
