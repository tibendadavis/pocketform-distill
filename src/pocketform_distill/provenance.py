"""Record what produced a stage's output: the config and the git commit. Every stage writes one run.json."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel


def git_commit() -> dict[str, str | bool]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": False}


def write_run_info(out_dir: Path, stage: str, config: BaseModel, **extra: object) -> None:
    # No timestamp on purpose: the same config and commit must give byte-identical output.
    info = {"stage": stage, "git": git_commit(), "config": config.model_dump(mode="json"), **extra}
    (out_dir / "run.json").write_text(json.dumps(info, indent=2) + "\n")
