"""PyTorch Lightning orchestration for behavior-cloning training."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger as loguru_logger
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.model.input_schema import policy_input_schema_metadata
from ptcg_rl.model.network import build_agent_policy_value_net
from ptcg_rl.model.state_encoder import LEGACY_STATE_ENCODER_MISSING_KEYS
from ptcg_rl.training.bc_dataset import (
    BCBatch,
    KaggleStepDataConfig,
    iter_bc_batches,
    move_bc_batch_to_device,
)
from ptcg_rl.training.behavior_cloning import (
    BehaviorCloningConfig,
    LossBreakdown,
    MetricAccumulator,
    _metric_loss,
    _save_checkpoint,
    _set_seed,
    _write_json,
    behavior_cloning_loss,
)
from ptcg_rl.training.run_config import (
    resolve_training_output_dir,
    resolved_training_config_dump,
)


class _BCIterableDataset(IterableDataset[BCBatch]):
    """Iterable wrapper that exposes existing streaming BC batches to Lightning."""

    def __init__(
        self,
        config: KaggleStepDataConfig,
        *,
        split: str,
        seed: int,
        epoch_provider: Callable[[], int],
        max_batches: int | None,
    ) -> None:
        """Initialize a split-specific streaming dataset."""
        super().__init__()
        self._config = config
        self._split = split
        self._seed = seed
        self._epoch_provider = epoch_provider
        self._max_batches = max_batches

    def __iter__(self) -> Iterator[BCBatch]:
        """Yield tensor-ready BC batches for the current Lightning epoch."""
        yield from iter_bc_batches(
            self._config,
            split=cast(Any, self._split),
            epoch=self._epoch_provider(),
            seed=self._seed,
            device=None,
            max_batches=self._max_batches,
        )


class BehaviorCloningDataModule(pl.LightningDataModule):
    """Lightning datamodule over bounded-memory BC batch streams."""

    def __init__(self, config: BehaviorCloningConfig) -> None:
        """Initialize dataloaders from the Hydra-backed BC config."""
        super().__init__()
        self._config = config

    def train_dataloader(self) -> DataLoader[BCBatch]:
        """Return the training stream dataloader."""
        return self._dataloader(
            split="train",
            max_batches=self._config.max_train_batches_per_epoch,
        )

    def val_dataloader(self) -> DataLoader[BCBatch]:
        """Return the validation stream dataloader."""
        return self._dataloader(
            split="validation",
            max_batches=self._config.max_validation_batches,
        )

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        """Move custom BC dataclass batches onto the trainer device."""
        del dataloader_idx
        if isinstance(batch, BCBatch):
            return move_bc_batch_to_device(
                batch,
                device=device,
                non_blocking=self._config.data.pin_memory,
            )
        return batch

    def _dataloader(self, *, split: str, max_batches: int | None) -> DataLoader[BCBatch]:
        dataset = _BCIterableDataset(
            self._config.data,
            split=split,
            seed=self._config.seed,
            epoch_provider=self._current_epoch,
            max_batches=max_batches,
        )
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def _current_epoch(self) -> int:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 0
        return int(trainer.current_epoch)


class BehaviorCloningLightningModule(pl.LightningModule):
    """Lightning module for BC policy/value training."""

    def __init__(self, config: BehaviorCloningConfig) -> None:
        """Build the policy/value network and metric state."""
        super().__init__()
        self.config = config
        self.model = build_agent_policy_value_net(config.model)
        if config.initial_weights_checkpoint is not None:
            _load_initial_model_weights(
                self.model,
                deck_records.repo_path(config.initial_weights_checkpoint),
            )
        self.history: list[dict[str, Any]] = []
        self.best_validation_loss: float | None = None
        self._train_metrics = MetricAccumulator()
        self._validation_metrics = MetricAccumulator(collect_details=True)
        self._latest_train_metrics: dict[str, Any] = {}
        self._last_epoch_record: dict[str, Any] | None = None
        self.initial_validation_metrics: dict[str, Any] | None = None
        self._capture_initial_validation = False

    def forward(self, batch: BCBatch) -> Any:
        """Return raw model outputs for a collated BC batch."""
        return self.model(batch.states, batch.options)

    def configure_optimizers(self) -> Any:
        """Build the Lightning-managed optimizer and optional warmup schedule."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        warmup_steps = self.config.warmup_steps
        if warmup_steps <= 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_epoch_start(self) -> None:
        """Reset train metrics for the epoch."""
        self._train_metrics = MetricAccumulator()

    def training_step(self, batch: BCBatch, batch_idx: int) -> Tensor:
        """Run one BC optimization step."""
        del batch_idx
        breakdown = self._loss(batch)
        self._train_metrics.update(breakdown, batch)
        self.log(
            "train_step_loss",
            breakdown.loss.detach(),
            on_step=True,
            on_epoch=False,
            batch_size=len(batch.actions),
        )
        if self._train_metrics.batches % self.config.log_every_batches == 0:
            loguru_logger.info(
                "bc train progress {}",
                json.dumps(
                    {
                        "epoch": int(self.current_epoch),
                        "batch": self._train_metrics.batches,
                        "train": self._train_metrics.as_dict(),
                    },
                    sort_keys=True,
                ),
            )
        return breakdown.loss

    def on_train_epoch_end(self) -> None:
        """Persist last-checkpoint artifacts even when validation is empty."""
        self._latest_train_metrics = self._train_metrics.as_dict()
        self.log(
            "train_loss",
            float(self._latest_train_metrics["loss"]),
            prog_bar=True,
        )
        self._log_scalar_metrics("train", self._latest_train_metrics)
        self.log("validation_loss", float("inf"), prog_bar=False)
        if not self._validation_will_run():
            self._record_epoch(
                train_metrics=self._latest_train_metrics,
                validation_metrics=MetricAccumulator(collect_details=True).as_dict(),
                update_best=False,
            )

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics for the epoch."""
        self._validation_metrics = MetricAccumulator(collect_details=True)

    def validation_step(self, batch: BCBatch, batch_idx: int) -> None:
        """Run one validation batch and accumulate detailed metrics."""
        del batch_idx
        breakdown = self._loss(batch)
        self._validation_metrics.update(breakdown, batch)

    def on_validation_epoch_end(self) -> None:
        """Finalize validation metrics and runtime checkpoint exports."""
        if self.trainer.sanity_checking:
            return
        validation_metrics = self._validation_metrics.as_dict()
        if self._capture_initial_validation:
            self.initial_validation_metrics = validation_metrics
            self._capture_initial_validation = False
            return
        validation_loss = _metric_loss(validation_metrics)
        self.log(
            "validation_loss",
            float("inf") if validation_loss is None else validation_loss,
            prog_bar=True,
        )
        self._log_scalar_metrics("validation", validation_metrics)
        train_metrics = self._latest_train_metrics or self._train_metrics.as_dict()
        self._record_epoch(
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            update_best=True,
        )

    def capture_initial_validation(self) -> None:
        """Route the next validation pass to immutable warm-start metrics."""
        if self._capture_initial_validation:
            raise RuntimeError("initial validation capture is already pending")
        self._capture_initial_validation = True

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Store config and metric history in the Lightning checkpoint."""
        checkpoint["model_config"] = self.config.model.model_dump(mode="json")
        checkpoint["policy_input_schema"] = policy_input_schema_metadata()
        checkpoint["training_config"] = resolved_training_config_dump(
            self.config,
            task_name="bc",
            run=self.config.run,
            output_dir=self.config.output_dir,
        )
        checkpoint["history"] = self.history
        checkpoint["best_validation_loss"] = self.best_validation_loss
        checkpoint["last_epoch_record"] = self._last_epoch_record

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore metric history used for summaries and runtime exports."""
        history = checkpoint.get("history")
        if isinstance(history, list):
            self.history = [
                dict(item) for item in history if isinstance(item, Mapping)
            ]
        best_validation_loss = checkpoint.get("best_validation_loss")
        if isinstance(best_validation_loss, int | float):
            self.best_validation_loss = float(best_validation_loss)
        last_epoch_record = checkpoint.get("last_epoch_record")
        if isinstance(last_epoch_record, Mapping):
            self._last_epoch_record = dict(last_epoch_record)

    def _loss(self, batch: BCBatch) -> LossBreakdown:
        return behavior_cloning_loss(
            self.model,
            batch,
            value_loss_weight=self.config.value_loss_weight,
            prize_diff_loss_weight=self.config.prize_diff_loss_weight,
            opponent_card_loss_weight=self.config.opponent_card_loss_weight,
            opponent_hand_loss_weight=self.config.opponent_hand_loss_weight,
            chosen_effect_loss_weight=self.config.chosen_effect_loss_weight,
        )

    def _record_epoch(
        self,
        *,
        train_metrics: Mapping[str, Any],
        validation_metrics: Mapping[str, Any],
        update_best: bool,
    ) -> None:
        epoch_record = {
            "epoch": int(self.current_epoch),
            "train": dict(train_metrics),
            "validation": dict(validation_metrics),
        }
        self._last_epoch_record = epoch_record
        self._upsert_history(epoch_record)
        self._write_runtime_artifacts(epoch_record, update_best=update_best)

    def _upsert_history(self, epoch_record: Mapping[str, Any]) -> None:
        epoch = int(epoch_record["epoch"])
        for index, existing in enumerate(self.history):
            if int(existing.get("epoch", -1)) == epoch:
                self.history[index] = dict(epoch_record)
                return
        self.history.append(dict(epoch_record))
        self.history.sort(key=lambda record: int(record.get("epoch", -1)))

    def _write_runtime_artifacts(
        self,
        epoch_record: Mapping[str, Any],
        *,
        update_best: bool,
    ) -> None:
        if not self._is_global_zero():
            return
        output_dir = deck_records.repo_path(
            resolve_training_output_dir(
                task_name="bc",
                run=self.config.run,
                output_dir=self.config.output_dir,
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(epoch_record["epoch"])
        if self.config.checkpoints.save_runtime_last:
            _save_checkpoint(
                output_dir / "checkpoint_last.pt",
                model=self.model,
                config=self.config,
                epoch=epoch,
                metrics=epoch_record,
            )
        validation_metrics = cast(Mapping[str, Any], epoch_record["validation"])
        validation_loss = _metric_loss(validation_metrics)
        is_best = update_best and validation_loss is not None and (
            self.best_validation_loss is None
            or validation_loss < self.best_validation_loss
        )
        if is_best:
            self.best_validation_loss = validation_loss
            if self.config.checkpoints.save_runtime_best:
                _save_checkpoint(
                    output_dir / "checkpoint_best.pt",
                    model=self.model,
                    config=self.config,
                    epoch=epoch,
                    metrics=epoch_record,
                )
        _write_json(output_dir / "metrics.json", {"history": self.history})

    def _is_global_zero(self) -> bool:
        trainer = getattr(self, "trainer", None)
        return trainer is None or bool(trainer.is_global_zero)

    def _log_scalar_metrics(self, prefix: str, metrics: Mapping[str, Any]) -> None:
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            scalar = float(value)
            if not math.isfinite(scalar):
                continue
            self.log(f"{prefix}/{name}", scalar, prog_bar=False)

    def _validation_will_run(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        num_val_batches = getattr(trainer, "num_val_batches", None)
        if isinstance(num_val_batches, list):
            return any(_batch_count_enabled(batches) for batches in num_val_batches)
        if isinstance(num_val_batches, tuple):
            return any(_batch_count_enabled(batches) for batches in num_val_batches)
        if isinstance(num_val_batches, int):
            return num_val_batches != 0
        if isinstance(num_val_batches, float):
            return num_val_batches != 0.0
        return False


def run_lightning_behavior_cloning(config: BehaviorCloningConfig) -> dict[str, Any]:
    """Train the pointer policy/value model with PyTorch Lightning."""
    _set_seed(config.seed)
    output_dir = deck_records.repo_path(
        resolve_training_output_dir(
            task_name="bc",
            run=config.run,
            output_dir=config.output_dir,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loguru_log_path = _resolve_loguru_log_path(config, output_dir)
    try:
        _configure_loguru(config, output_dir)
        loguru_logger.info(
            "starting BC training output_dir={} epochs={} batch_size={}",
            deck_records.display_path(output_dir),
            config.epochs,
            config.data.batch_size,
        )
        _prune_disabled_checkpoints(config, output_dir)

        checkpoint_callbacks = _build_model_checkpoints(config, output_dir)
        lightning_logger = _build_lightning_logger(config, output_dir)
        module = BehaviorCloningLightningModule(config)
        datamodule = BehaviorCloningDataModule(config)
        trainer = _build_trainer(
            config,
            output_dir=output_dir,
            checkpoint_callbacks=checkpoint_callbacks,
            lightning_logger=lightning_logger,
        )
        resume_checkpoint = _resolve_resume_checkpoint(config, output_dir)
        if config.evaluate_initial_weights:
            if resume_checkpoint is not None:
                raise ValueError(
                    "initial-weight evaluation cannot run while resuming an existing "
                    "Lightning checkpoint"
                )
            module.capture_initial_validation()
            trainer.validate(module, datamodule=datamodule, verbose=False)
            if module.initial_validation_metrics is None:
                raise RuntimeError("initial-weight validation did not produce metrics")
            _write_json(
                output_dir / "initial_validation.json",
                module.initial_validation_metrics,
            )
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_checkpoint)

        summary = _training_summary(
            config,
            output_dir=output_dir,
            module=module,
            checkpoint_callbacks=checkpoint_callbacks,
            resume_checkpoint=resume_checkpoint,
            loguru_log_path=loguru_log_path,
            tensorboard_log_dir=_tensorboard_log_dir(lightning_logger),
        )
        _write_json(output_dir / "summary.json", summary)
        loguru_logger.info("finished BC training {}", json.dumps(summary, sort_keys=True))
        return summary
    except Exception:
        loguru_logger.exception("BC training failed")
        raise
    finally:
        _restore_loguru(config)


def _build_model_checkpoints(
    config: BehaviorCloningConfig,
    output_dir: Path,
) -> list[ModelCheckpoint]:
    checkpoints_dir = output_dir / "checkpoints"
    callbacks: list[ModelCheckpoint] = []
    if config.checkpoints.save_lightning_last:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoints_dir,
                filename="last-step",
                save_top_k=0,
                save_last=True,
                every_n_train_steps=(
                    config.checkpoints.lightning_last_every_n_train_steps
                ),
                auto_insert_metric_name=False,
            )
        )
    if config.checkpoints.save_lightning_best:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoints_dir,
                filename="best",
                monitor="validation_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                auto_insert_metric_name=False,
            )
        )
    return callbacks


def _build_lightning_logger(
    config: BehaviorCloningConfig,
    output_dir: Path,
) -> TensorBoardLogger | bool:
    if not config.tensorboard.enabled:
        return False
    save_dir = (
        output_dir
        if config.tensorboard.save_dir is None
        else deck_records.repo_path(config.tensorboard.save_dir)
    )
    return TensorBoardLogger(
        save_dir=str(save_dir),
        name=config.tensorboard.name,
        version=config.tensorboard.version,
        default_hp_metric=config.tensorboard.default_hp_metric,
    )


def _configure_loguru(
    config: BehaviorCloningConfig,
    output_dir: Path,
) -> tuple[int, ...]:
    if not config.loguru.enabled:
        return ()

    loguru_logger.remove()
    sink_ids: list[int] = []
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}"
    if config.loguru.console_enabled:
        sink_ids.append(
            loguru_logger.add(
                sys.stderr,
                level=config.loguru.level,
                format=log_format,
                enqueue=config.loguru.enqueue,
            )
        )
    if config.loguru.file_enabled:
        log_path = _resolve_loguru_log_path(config, output_dir)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_kwargs: dict[str, Any] = {
                "level": config.loguru.level,
                "format": log_format,
                "enqueue": config.loguru.enqueue,
                "serialize": config.loguru.serialize,
            }
            if config.loguru.rotation is not None:
                file_kwargs["rotation"] = config.loguru.rotation
            if config.loguru.retention is not None:
                file_kwargs["retention"] = config.loguru.retention
            sink_ids.append(loguru_logger.add(log_path, **file_kwargs))
    return tuple(sink_ids)


def _restore_loguru(config: BehaviorCloningConfig) -> None:
    if not config.loguru.enabled:
        return
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="INFO")


def _resolve_loguru_log_path(
    config: BehaviorCloningConfig,
    output_dir: Path,
) -> Path | None:
    if not config.loguru.enabled or not config.loguru.file_enabled:
        return None
    path = config.loguru.file_path
    if path is None:
        return output_dir / "train.log"
    return path if path.is_absolute() else output_dir / path


def _prune_disabled_checkpoints(
    config: BehaviorCloningConfig,
    output_dir: Path,
) -> None:
    if not config.checkpoints.save_runtime_last:
        _unlink_if_exists(output_dir / "checkpoint_last.pt")
    if not config.checkpoints.save_runtime_best:
        _unlink_if_exists(output_dir / "checkpoint_best.pt")

    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return
    if not config.checkpoints.save_lightning_last:
        _unlink_if_exists(checkpoints_dir / "last.ckpt")
    if not config.checkpoints.save_lightning_best:
        for checkpoint_path in checkpoints_dir.glob("best*.ckpt"):
            _unlink_if_exists(checkpoint_path)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _batch_count_enabled(value: Any) -> bool:
    if isinstance(value, int | float):
        return float(value) != 0.0
    return bool(value)


def _build_trainer(
    config: BehaviorCloningConfig,
    *,
    output_dir: Path,
    checkpoint_callbacks: Sequence[ModelCheckpoint],
    lightning_logger: TensorBoardLogger | bool,
) -> pl.Trainer:
    trainer_kwargs = _trainer_device_kwargs(config)
    return pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=config.epochs,
        callbacks=list(checkpoint_callbacks),
        logger=lightning_logger,
        enable_progress_bar=config.enable_progress_bar,
        gradient_clip_val=config.gradient_clip_norm,
        log_every_n_steps=max(1, config.log_every_batches),
        num_sanity_val_steps=0,
        precision=cast(Any, config.precision),
        **trainer_kwargs,
    )


def _trainer_device_kwargs(config: BehaviorCloningConfig) -> dict[str, Any]:
    if config.accelerator != "auto" or config.devices != "auto":
        return {"accelerator": config.accelerator, "devices": config.devices}

    device = config.device.lower()
    if device == "cpu":
        return {"accelerator": "cpu", "devices": "auto"}
    if device == "gpu":
        return {"accelerator": "gpu", "devices": "auto"}
    if device == "cuda":
        return {"accelerator": "gpu", "devices": 1}
    if device.startswith("cuda:"):
        return {"accelerator": "gpu", "devices": [int(device.split(":", 1)[1])]}
    return {"accelerator": "auto", "devices": "auto"}


def _resolve_resume_checkpoint(
    config: BehaviorCloningConfig,
    output_dir: Path,
) -> Path | None:
    if config.resume_from_checkpoint is not None:
        checkpoint_path = deck_records.repo_path(config.resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path
    if not config.auto_resume:
        return None
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    return checkpoint_path if checkpoint_path.exists() else None


def _load_initial_model_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Warm-start model weights without restoring optimizer or trainer state."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"initial weights checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _model_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing - LEGACY_STATE_ENCODER_MISSING_KEYS or unexpected:
        raise RuntimeError("initial weights checkpoint is incompatible with model")


def _model_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Extract a runtime or Lightning model state dictionary."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("initial weights checkpoint must be a mapping")
    raw_state: Any = checkpoint
    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            raw_state = candidate
            break
    if not isinstance(raw_state, Mapping):
        raise TypeError("initial weights checkpoint does not contain model weights")
    state_dict = {str(key): value for key, value in raw_state.items()}
    if state_dict and all(key.startswith("model.") for key in state_dict):
        state_dict = {
            key.removeprefix("model."): value for key, value in state_dict.items()
        }
    return state_dict


def _checkpoint_sha256(path: Path) -> str:
    """Return a streaming SHA256 fingerprint for one checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _training_summary(
    config: BehaviorCloningConfig,
    *,
    output_dir: Path,
    module: BehaviorCloningLightningModule,
    checkpoint_callbacks: Sequence[ModelCheckpoint],
    resume_checkpoint: Path | None,
    loguru_log_path: Path | None,
    tensorboard_log_dir: Path | None,
) -> dict[str, Any]:
    checkpoint_last = output_dir / "checkpoint_last.pt"
    checkpoint_best = output_dir / "checkpoint_best.pt"
    lightning_last = output_dir / "checkpoints" / "last.ckpt"
    lightning_best = _best_lightning_checkpoint_path(checkpoint_callbacks)
    initial_weights = (
        deck_records.repo_path(config.initial_weights_checkpoint)
        if config.initial_weights_checkpoint is not None
        else None
    )
    history = module.history
    return {
        "output_dir": deck_records.display_path(output_dir),
        "run": config.run.model_dump(mode="json"),
        "epochs": config.epochs,
        "checkpoint_policy": config.checkpoints.model_dump(mode="json"),
        "loguru_log_path": _display_existing_path(loguru_log_path),
        "tensorboard_log_dir": _display_existing_path(tensorboard_log_dir),
        "best_validation_loss": module.best_validation_loss,
        "initial_validation": module.initial_validation_metrics,
        "initial_validation_path": _display_existing_path(
            output_dir / "initial_validation.json"
            if config.evaluate_initial_weights
            else None
        ),
        "last": history[-1] if history else None,
        "checkpoint_last": _display_existing_path(checkpoint_last),
        "checkpoint_best": _display_existing_path(checkpoint_best),
        "lightning_checkpoint_last": _display_existing_path(lightning_last),
        "lightning_checkpoint_best": _display_existing_path(lightning_best),
        "initial_weights_checkpoint": (
            deck_records.display_path(initial_weights)
            if initial_weights is not None
            else None
        ),
        "initial_weights_sha256": (
            _checkpoint_sha256(initial_weights)
            if initial_weights is not None
            else None
        ),
        "resume_checkpoint": (
            deck_records.display_path(resume_checkpoint)
            if resume_checkpoint is not None
            else None
        ),
    }


def _tensorboard_log_dir(lightning_logger: TensorBoardLogger | bool) -> Path | None:
    if isinstance(lightning_logger, TensorBoardLogger):
        return Path(lightning_logger.log_dir)
    return None


def _best_lightning_checkpoint_path(
    checkpoint_callbacks: Sequence[ModelCheckpoint],
) -> Path | None:
    for checkpoint_callback in checkpoint_callbacks:
        if checkpoint_callback.best_model_path:
            return Path(checkpoint_callback.best_model_path)
    return None


def _display_existing_path(path: Path | None) -> str | None:
    return deck_records.display_path(path) if path is not None and path.exists() else None
