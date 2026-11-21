"""The builder's taskset: one task per domain brief."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

from autogenesis.briefs import briefs_for
from autogenesis.prompts import BRIEF_TEMPLATE, BUILDER_SYSTEM_PROMPT


class AutogenesisConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "all"] = "train"
    """`eval` holds out whole domain families (grids, code tracing)."""
    repeats: int = 1
    """Times each brief appears per pass, for more episodes per brief in training."""


class BriefData(vf.TaskData):
    brief_id: str
    family: str


class BuilderTask(vf.Task[BriefData, vf.State, vf.TaskConfig]):
    """The builder's task. Scoring needs the sandbox and the solver panel, so the
    env records the rewards (`AutogenesisEnv.run`), not task reward hooks."""


class AutogenesisTaskset(vf.Taskset[BuilderTask, AutogenesisConfig]):
    def load(self) -> Iterator[BuilderTask]:
        for repeat in range(self.config.repeats):
            for brief in briefs_for(self.config.split):
                yield BuilderTask(
                    BriefData(
                        id=f"{brief.id}#{repeat}" if self.config.repeats > 1 else brief.id,
                        name=brief.title,
                        prompt=BRIEF_TEMPLATE.format(
                            title=brief.title, description=brief.description
                        ),
                        system_prompt=BUILDER_SYSTEM_PROMPT,
                        brief_id=brief.id,
                        family=brief.family,
                    ),
                    self.config.task,
                )
