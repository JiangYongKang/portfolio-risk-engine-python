"""Versioning of parameter sets / scenarios and compatibility rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RulesVersion:
    version: str
    created_at: str
    parent: str | None
    compatible_parents: frozenset[str]
    notes: str = ""


class VersionRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, RulesVersion] = {}

    def register(self, rules: RulesVersion) -> None:
        raise NotImplementedError

    def get(self, version: str) -> RulesVersion:
        raise NotImplementedError

    def is_compatible(self, old: str, new: str) -> bool:
        raise NotImplementedError
