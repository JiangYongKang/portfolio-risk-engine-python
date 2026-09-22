"""Versioned parameter sets and reproducible scenario evaluation.

Compatibility model
-------------------
Each :class:`VersionedParameterSet` carries:

* ``version``        – the parameter-set version itself;
* ``rule_version``   – the calculation-rule generation it was authored under;
* ``compatible_with`` – older rule generations still able to reproduce it.

Re-evaluating a parameter set under a *new* rule version reproduces the
historical conclusion when the new rule is listed in ``compatible_with``
(the snapshot of the old inputs is replayed under the pinned rule).  If the
new rule is not listed, the result is marked ``reproducible=False`` with
reason ``version_incompatible`` and the caller must explicitly accept the
new rule before a value is produced.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from app.errors import VersionIncompatibleError, VersionNotFoundError
from app.models import (
    Portfolio,
    ScenarioEvaluation,
    VersionedParameterSet,
)
from app.risk.scenarios import ScenarioEngine


@dataclass(frozen=True)
class ScenarioSnapshot:
    """Frozen replay input for a parameter-set version."""

    scenario_name: str
    portfolio: Portfolio
    factor_map: dict[str, dict[str, float]] = field(default_factory=dict)


class ParameterRegistry:
    def __init__(self, engine: ScenarioEngine, current_rule_version: int = 1) -> None:
        self._engine = engine
        self.current_rule_version = current_rule_version
        self._sets: dict[tuple[str, int], VersionedParameterSet] = {}
        self._snapshots: dict[tuple[str, int], ScenarioSnapshot] = {}

    def register(
        self,
        parameter_set: VersionedParameterSet,
        snapshot: ScenarioSnapshot | None = None,
    ) -> None:
        key = (parameter_set.name, parameter_set.version)
        if key in self._sets:
            raise VersionIncompatibleError(
                f"parameter set {parameter_set.name!r} version "
                f"{parameter_set.version} already exists; versions are immutable"
            )
        self._sets[key] = parameter_set.model_copy(deep=True)
        if snapshot is not None:
            self._snapshots[key] = copy.deepcopy(snapshot)

    def get(self, name: str, version: int) -> VersionedParameterSet:
        key = (name, version)
        if key not in self._sets:
            raise VersionNotFoundError(
                f"parameter set {name!r} version {version} not found"
            )
        return self._sets[key]

    def is_compatible(self, name: str, version: int, rule_version: int | None = None) -> bool:
        ps = self.get(name, version)
        target = self.current_rule_version if rule_version is None else rule_version
        if target == ps.rule_version:
            return True
        return target in ps.compatible_with

    def evaluate_versioned(
        self,
        name: str,
        version: int,
        *,
        accept_incompatible: bool = False,
    ) -> ScenarioEvaluation:
        ps = self.get(name, version)
        snap = self._snapshots.get((name, version))
        if snap is None:
            raise VersionNotFoundError(
                f"no replay snapshot stored for {name!r} version {version}"
            )

        if not self.is_compatible(name, version):
            if not accept_incompatible:
                raise VersionIncompatibleError(
                    f"parameter set {name!r} v{version} was authored under rule "
                    f"v{ps.rule_version}; current rule is v{self.current_rule_version} "
                    "and compatibility was not declared"
                )
            result = self._engine.evaluate(
                snap.scenario_name, snap.portfolio, snap.factor_map
            )
            return ScenarioEvaluation(
                scenario=snap.scenario_name,
                version=version,
                rule_version=self.current_rule_version,
                reproducible=False,
                result=result,
                note=(
                    f"recomputed under new rule v{self.current_rule_version}; "
                    f"authored under v{ps.rule_version}, incompatibility accepted"
                ),
            )

        result = self._engine.evaluate(
            snap.scenario_name, snap.portfolio, snap.factor_map
        )
        return ScenarioEvaluation(
            scenario=snap.scenario_name,
            version=version,
            rule_version=ps.rule_version,
            reproducible=True,
            result=result,
            note=(
                f"replayed historical inputs under pinned rule v{ps.rule_version}"
                if self.current_rule_version != ps.rule_version
                else "evaluated under original rule"
            ),
        )
