"""Versioning of parameter sets / scenarios and compatibility rules.

A :class:`RulesVersion` represents one generation of the risk rules
(return convention, quantile rule, covariance estimator, scenario PnL
convention, ...).  Every risk report and scenario result is stamped with a
version so historical conclusions can be reproduced later.

Compatibility policy:

* Each new version *explicitly* declares the old versions whose conclusions
  remain reproducible under the new rules (``compatible_parents``).
* Recomputing a conclusion originally produced at ``old`` under rules
  ``new`` is permitted iff ``new == old`` or ``old`` is declared compatible.
* Otherwise recomputation is refused with
  :class:`IncompatibleVersionError` unless the caller explicitly opts in
  via ``allow_breaking=True`` (results are then stamped with the new
  version and a breaking-change marker).
"""

from __future__ import annotations

from dataclasses import dataclass

from risk_engine.errors import (
    IncompatibleVersionError,
    UnknownVersionError,
)


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
        if rules.version in self._rules:
            raise ValueError(f"version '{rules.version}' already registered")
        if rules.parent is not None and rules.parent not in self._rules:
            raise UnknownVersionError(
                f"parent version '{rules.parent}' of new version "
                f"'{rules.version}' does not exist"
            )
        for old in rules.compatible_parents:
            if old not in self._rules:
                raise UnknownVersionError(
                    f"compatible parent '{old}' declared by "
                    f"'{rules.version}' does not exist"
                )
        self._rules[rules.version] = rules

    def get(self, version: str) -> RulesVersion:
        try:
            return self._rules[version]
        except KeyError:
            raise UnknownVersionError(f"version '{version}' is not registered")

    def versions(self) -> tuple[str, ...]:
        return tuple(self._rules)

    def is_compatible(self, old: str, new: str) -> bool:
        """True iff conclusions produced at ``old`` reproduce under ``new``."""
        rules_new = self.get(new)
        if new == old:
            # still require that old exists
            self.get(old)
            return True
        return old in rules_new.compatible_parents

    def check_compatible(
        self,
        old: str,
        new: str,
        *,
        allow_breaking: bool = False,
    ) -> None:
        """Raise IncompatibleVersionError unless reproduction is allowed."""
        if self.is_compatible(old, new):
            return
        if allow_breaking:
            return
        raise IncompatibleVersionError(
            f"conclusions stamped '{old}' are not declared reproducible "
            f"under '{new}'; pass allow_breaking=True to recompute and "
            "re-stamp under the new rules"
        )


def default_registry() -> VersionRegistry:
    """Registry seeded with the v1 baseline used by every default request."""
    reg = VersionRegistry()
    reg.register(
        RulesVersion(
            version="v1",
            created_at="2026-01-01",
            parent=None,
            compatible_parents=frozenset(),
            notes="baseline: sample covariance, lower-rank historical VaR",
        )
    )
    return reg
