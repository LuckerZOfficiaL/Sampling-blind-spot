"""Lightweight Weights & Biases wrapper with graceful no-op fallback.

Provides a thin abstraction over wandb that:
- Initializes a run only when use_wandb=True
- Falls back to no-ops if wandb is unavailable or disabled
- Never crashes the experiment due to logging failures
"""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _try_import_wandb():
    """Attempt to import wandb, return None if unavailable."""
    try:
        import wandb  # pylint: disable=import-outside-toplevel
        return wandb
    except ImportError:
        return None


class WandbRun:  # pylint: disable=broad-exception-caught
    """Safe wrapper around a wandb run.

    All methods are no-ops when wandb is disabled or unavailable.
    Errors in logging are caught and printed as warnings, never raised.
    """

    def __init__(self, enabled: bool = False, run=None):
        self._enabled = enabled
        self._run = run
        self._wandb = _try_import_wandb() if enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._run is not None

    @property
    def run(self):
        """Access the underlying wandb.Run, or None."""
        return self._run

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics. No-op if disabled."""
        if not self.enabled:
            return
        try:
            kwargs = {"step": step} if step is not None else {}
            self._run.log(data, **kwargs)
        except Exception as exc:
            print(f"[wandb] Warning: log failed: {exc}")

    def config_update(self, data: Dict[str, Any]) -> None:
        """Update run config. No-op if disabled."""
        if not self.enabled:
            return
        try:
            self._run.config.update(data)
        except Exception as exc:
            print(f"[wandb] Warning: config update failed: {exc}")

    def summary_update(self, data: Dict[str, Any]) -> None:
        """Update run summary. No-op if disabled."""
        if not self.enabled:
            return
        try:
            for key, value in data.items():
                self._run.summary[key] = value
        except Exception as exc:
            print(f"[wandb] Warning: summary update failed: {exc}")

    def log_artifact(
        self,
        path: str,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a file or directory as a wandb Artifact. No-op if disabled."""
        if not self.enabled:
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
            file_path = Path(path)
            if file_path.is_dir():
                artifact.add_dir(str(file_path))
            else:
                artifact.add_file(str(file_path))
            self._run.log_artifact(artifact)
        except Exception as exc:
            print(f"[wandb] Warning: artifact upload failed: {exc}")

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
    ) -> None:
        """Log a wandb.Table. No-op if disabled."""
        if not self.enabled:
            return
        try:
            table = self._wandb.Table(columns=columns, data=data)
            self._run.log({key: table})
        except Exception as exc:
            print(f"[wandb] Warning: table log failed: {exc}")

    def finish(self) -> None:
        """Finish the run. No-op if disabled."""
        if not self.enabled:
            return
        try:
            self._run.finish()
        except Exception as exc:
            print(f"[wandb] Warning: finish failed: {exc}")


@contextmanager
def timer(wb_run: WandbRun, key: str):
    """Context manager that measures wall-clock time and logs it.

    Args:
        wb_run: WandbRun instance (logging is skipped if disabled).
        key: Metric key to log, e.g. "time/model_load_s".

    Yields:
        None. The elapsed time is logged on exit.
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    wb_run.log({key: elapsed})


def init_wandb(
    enabled: bool,
    project: str = "llm-activations",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Sequence[str]] = None,
) -> WandbRun:
    """Initialize a W&B run if enabled, otherwise return a no-op wrapper.

    Args:
        enabled: Whether to actually start a wandb run.
        project: W&B project name.
        name: Run name.
        config: Run config dict to log as wandb.config.
        tags: Optional tags for the run.

    Returns:
        A WandbRun wrapper (no-op if disabled or wandb unavailable).
    """
    if not enabled:
        return WandbRun(enabled=False)

    wandb = _try_import_wandb()
    if wandb is None:
        print("[wandb] Warning: wandb not installed, logging disabled")
        return WandbRun(enabled=False)

    try:
        run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            tags=list(tags) if tags else None,
        )
        return WandbRun(enabled=True, run=run)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[wandb] Warning: init failed ({exc}), logging disabled")
        return WandbRun(enabled=False)
