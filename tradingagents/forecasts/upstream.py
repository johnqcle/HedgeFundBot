"""Load a separate Kronos checkout. Nothing in this package vendors the model.

Kronos imports itself as the top-level package ``model``. A checkout placed
inside HedgeFundBot would be an accidental nested repository and can be swept
into the install tree, so it is refused.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tradingagents.forecasts.constants import (
    KRONOS_REPO,
    REVIEWED_UPSTREAM_COMMIT,
)


class KronosUpstreamError(RuntimeError):
    """A configured Kronos checkout or weight download cannot be used."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path | None:
    """The HedgeFundBot checkout when this file still lives in it."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "tradingagents").is_dir():
        return candidate
    return None


def _contained(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def is_nested_checkout(path: Path) -> bool:
    """True when ``path`` sits inside this package or this repository."""
    resolved = path.resolve()
    if _contained(resolved, package_root()):
        return True
    root = repo_root()
    return root is not None and _contained(resolved, root)


def resolve_kronos_home(configured: str | None) -> Path:
    """The upstream checkout, or an error that names why it cannot be used."""
    raw = (configured or os.environ.get("TRADINGAGENTS_KRONOS_HOME") or "").strip()
    if raw:
        home = Path(raw).expanduser()
    else:
        root = repo_root()
        home = (root.parent / "Kronos") if root is not None else None
        if home is None:
            raise KronosUpstreamError(
                "kronos_home_unset",
                "Set TRADINGAGENTS_KRONOS_HOME to a Kronos clone that is not inside HedgeFundBot. "
                f"Upstream is {KRONOS_REPO}.",
            )
    home = home.resolve()
    if is_nested_checkout(home):
        raise KronosUpstreamError(
            "nested_kronos_checkout",
            f"{home} is inside HedgeFundBot. Clone Kronos beside the repository "
            "(for the intended Windows layout, C:\\Users\\johnq\\Projects\\Kronos) "
            "and point TRADINGAGENTS_KRONOS_HOME there. Do not checkout Kronos under "
            "tradingagents/, vendor/, or the repository root.",
        )
    if not (home / "model" / "kronos.py").is_file():
        raise KronosUpstreamError(
            "kronos_home_missing",
            f"No Kronos model package at {home}. Clone {KRONOS_REPO} outside this "
            "repository and set TRADINGAGENTS_KRONOS_HOME.",
        )
    return home


def upstream_commit(home: Path) -> str | None:
    """HEAD of the checkout, when it is a git working tree."""
    if not (home / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(home), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def load_predictor(home: Path, *, model_id: str, model_revision: str, tokenizer_id: str,
                   tokenizer_revision: str, device: str, max_context: int):
    """Import upstream ``model`` from ``home`` and build a ``KronosPredictor``.

    Weights download from Hugging Face on first use into the Hugging Face cache,
    not into this repository. The import is delayed so a run with Kronos disabled
    does not need torch.
    """
    import sys

    home_str = str(home)
    existing = sys.modules.get("model")
    existing_file = str(getattr(existing, "__file__", "") or "")
    if existing is not None and not existing_file.startswith(home_str):
        raise KronosUpstreamError(
            "model_import_conflict",
            "Another top-level package named 'model' is already imported "
            f"({existing_file or 'unknown'}). Kronos loads itself under that name, "
            "so this process cannot load it.",
        )
    if home_str not in sys.path:
        sys.path.insert(0, home_str)
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    except ImportError as exc:
        raise KronosUpstreamError(
            "dependency_missing",
            "Kronos could not be imported. Install the optional extra in the "
            f"HedgeFundBot environment (`pip install -e \".[kronos]\"`) and do not "
            f"apply Kronos requirements.txt over it. Import error: {exc}",
        ) from exc

    try:
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
        model = Kronos.from_pretrained(model_id, revision=model_revision)
    except Exception as exc:
        raise KronosUpstreamError(
            "weights_unavailable",
            "The pinned Kronos weights could not be loaded from Hugging Face "
            f"({model_id}@{model_revision}, {tokenizer_id}@{tokenizer_revision}). "
            f"{type(exc).__name__}: {exc}",
        ) from exc

    tokenizer.eval()
    model.eval()
    try:
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    except Exception as exc:
        raise KronosUpstreamError(
            "predictor_init_failed",
            f"KronosPredictor could not move to device {device!r}. {type(exc).__name__}: {exc}",
        ) from exc
    return predictor


def commit_warning(actual: str | None) -> str | None:
    if actual and actual != REVIEWED_UPSTREAM_COMMIT:
        return (
            f"checkout HEAD {actual} differs from the reviewed interface commit "
            f"{REVIEWED_UPSTREAM_COMMIT}; predict() was read at the reviewed commit"
        )
    return None
