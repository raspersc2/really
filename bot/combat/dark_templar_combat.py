from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    AttackTarget,
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
)
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitTreeQueryType
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to, cy_distance_to_squared
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES, DETECTION_STRUCTURES

if TYPE_CHECKING:
    from ares import AresBot


# revealed, this many enemy per templar is where taking out a detector stops being
# worth it, killing one hands cloak back to everyone around it
MAX_ENEMY_PER_TEMPLAR_NEAR_DETECTOR: int = 3
# fewer than this away from the target is just something to swing at in passing
MIN_GROUP_TO_CHASE: int = 3
# further to leave the target than to arrive, so the edge can't be flipped back over
AT_TARGET_DISTANCE: float = 15.0
LEFT_TARGET_DISTANCE: float = 20.0
# how long a decision outlives whatever caused it
COMMIT_SECONDS: float = 3.0
# running gets longer, turning round early just walks back into what we ran from
FLEE_COMMIT_SECONDS: float = 8.0


class TemplarState(IntEnum):
    """Lowest priority first, a Dark Templar can always move up straight away."""

    ADVANCE = auto()
    ARMY = auto()
    WORKERS = auto()
    DETECTORS = auto()
    FLEE = auto()


@dataclass
class Commitment:
    state: TemplarState
    until: float
    retreat_to: Point2 | None = None


@dataclass
class DarkTemplarCombat(BaseCombat):
    """Execute behavior for Dark Templar harassing enemy bases.

    Parameters
    ----------
    ai : AresBot
        Bot object that will be running the game
    config : Dict[Any, Any]
        Dictionary with the data from the configuration file
    mediator : ManagerMediator
        Used for getting information from managers in Ares.
    """

    ai: "AresBot"
    config: dict
    mediator: ManagerMediator
    _commitments: dict[int, Commitment] = field(default_factory=dict, init=False)
    _assigned_targets: dict[int, Point2] = field(default_factory=dict, init=False)
    _at_target: set[int] = field(default_factory=set, init=False)

    def execute(self, units: Union[list[Unit], Units], **kwargs) -> None:
        """Hunt detection, then workers, then the army, spreading out over `targets`.

        Parameters
        ----------
        units : list[Unit]
            The Dark Templar to control.
        **kwargs :
            targets : list[Point2]
                Enemy bases to split across, must not be empty.
            retreat_targets : list[Point2]
                Where to run when losing, the furthest is used, must not be empty.
        """
        targets: list[Point2] = kwargs["targets"]
        retreat_targets: list[Point2] = kwargs["retreat_targets"]
        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=units,
            # further than turret / spore / cannon detection range
            distances=12,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid
        grid: np.ndarray = self.mediator.get_ground_grid

        for i, unit in enumerate(units):
            # `units` changes order and size, that mustn't move anyone to another base
            target: Point2 | None = self._assigned_targets.get(unit.tag)
            if target not in targets:
                target = targets[i % len(targets)]
                self._assigned_targets[unit.tag] = target
            close_enemy: list[Unit] = [
                u
                for u in near_enemy[unit.tag]
                if (not u.is_cloaked or u.is_cloaked and u.is_revealed)
                and (not u.is_burrowed or u.is_burrowed and u.is_visible)
                and not u.is_memory
                and not u.is_snapshot
                and not u.is_flying
                and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]
            # `is_detector` is False until finished, when it's too late to stop it
            detectors: list[Unit] = [
                u
                for u in close_enemy
                if u.is_detector or u.type_id in DETECTION_STRUCTURES
            ]
            workers: list[Unit] = [u for u in close_enemy if u.type_id in WORKER_TYPES]
            only_enemy_units: list[Unit] = [
                u for u in close_enemy if u.type_id not in ALL_STRUCTURES
            ]

            armed_enemy: list[Unit] = [
                u
                for u in only_enemy_units
                if u.can_attack_ground and u.type_id not in WORKER_TYPES
            ]

            at_target: bool = self._is_at_target(unit, target)
            groups: dict[TemplarState, list[Unit]] = {
                TemplarState.DETECTORS: detectors,
                TemplarState.WORKERS: workers,
                TemplarState.ARMY: armed_enemy,
            }
            commitment: Commitment = self._commit(
                unit,
                self._wanted_state(unit, units, target, groups),
                groups,
                retreat_targets,
            )

            maneuver: CombatManeuver = CombatManeuver()
            maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
            # `AttackTarget` every step restarts the swing, so each one follows a
            # `ShootTargetInRange` or nothing in range ever gets hit
            if commitment.state == TemplarState.FLEE:
                maneuver.add(ShootTargetInRange(unit, only_enemy_units))
                maneuver.add(
                    PathUnitToTarget(unit=unit, target=commitment.retreat_to, grid=grid)
                )
            elif commitment.state == TemplarState.DETECTORS:
                maneuver.add(ShootTargetInRange(unit, detectors))
                maneuver.add(
                    AttackTarget(
                        unit=unit, target=cy_closest_to(unit.position, detectors)
                    )
                )
            elif commitment.state == TemplarState.WORKERS:
                maneuver.add(ShootTargetInRange(unit, workers))
                maneuver.add(ShootTargetInRange(unit, armed_enemy))
                maneuver.add(
                    AttackTarget(
                        unit=unit, target=cy_closest_to(unit.position, workers)
                    )
                )
            elif commitment.state == TemplarState.ARMY:
                maneuver.add(ShootTargetInRange(unit, armed_enemy))
                maneuver.add(
                    AttackTarget(
                        unit=unit, target=cy_closest_to(unit.position, armed_enemy)
                    )
                )
            elif at_target:
                maneuver.add(ShootTargetInRange(unit, close_enemy))
                maneuver.add(AMove(unit=unit, target=target))
            else:
                # on the way there only units get hit in passing, a building would
                # hold us in place until it's dead
                maneuver.add(ShootTargetInRange(unit, only_enemy_units))
                maneuver.add(PathUnitToTarget(unit=unit, target=target, grid=grid))
            self.ai.register_behavior(maneuver)

    def _is_at_target(self, unit: Unit, target: Point2) -> bool:
        limit: float = (
            LEFT_TARGET_DISTANCE if unit.tag in self._at_target else AT_TARGET_DISTANCE
        )
        if cy_distance_to_squared(unit.position, target) < limit**2:
            self._at_target.add(unit.tag)
            return True
        self._at_target.discard(unit.tag)
        return False

    def _wanted_state(
        self,
        unit: Unit,
        units: Union[list[Unit], Units],
        target: Point2,
        groups: dict[TemplarState, list[Unit]],
    ) -> TemplarState:
        """What this step alone calls for, `_commit` decides if we act on it."""
        armed_enemy: list[Unit] = groups[TemplarState.ARMY]
        detectors: list[Unit] = groups[TemplarState.DETECTORS]
        if self._should_flee(unit, units, armed_enemy, detectors):
            return TemplarState.FLEE
        if detectors:
            return TemplarState.DETECTORS
        if self._worth_chasing(groups[TemplarState.WORKERS], target):
            return TemplarState.WORKERS
        if self._worth_chasing(armed_enemy, target):
            return TemplarState.ARMY
        return TemplarState.ADVANCE

    def _commit(
        self,
        unit: Unit,
        wanted: TemplarState,
        groups: dict[TemplarState, list[Unit]],
        retreat_targets: list[Point2],
    ) -> Commitment:
        """Move up in priority at once, only back down after holding for a while.

        Everything `wanted` depends on is a count or a distance, and those flicker
        around their limits. A decision can only be cut short if nothing is left to
        act on, there's no point staying committed to attacking what isn't there.
        """
        now: float = self.ai.time
        hold: float = (
            FLEE_COMMIT_SECONDS if wanted == TemplarState.FLEE else COMMIT_SECONDS
        )
        current: Commitment | None = self._commitments.get(unit.tag)
        if current and wanted == current.state:
            current.until = now + hold
            return current
        if (
            current
            and wanted < current.state
            and now < current.until
            and (current.state == TemplarState.FLEE or groups[current.state])
        ):
            return current

        # fixed on the way in, the furthest one changes as we run
        retreat_to: Point2 | None = None
        if wanted == TemplarState.FLEE:
            retreat_to = max(
                retreat_targets,
                key=lambda t: cy_distance_to_squared(unit.position, t),
            )
        self._commitments[unit.tag] = Commitment(wanted, now + hold, retreat_to)
        return self._commitments[unit.tag]

    @staticmethod
    def _worth_chasing(group: list[Unit], target: Point2) -> bool:
        """Lone units can lead us anywhere, so they only matter at the target."""
        return len(group) >= MIN_GROUP_TO_CHASE or any(
            cy_distance_to_squared(u.position, target) < AT_TARGET_DISTANCE**2
            for u in group
        )

    def _should_flee(
        self,
        unit: Unit,
        units: Union[list[Unit], Units],
        armed_enemy: list[Unit],
        detectors: list[Unit],
    ) -> bool:
        """Only worth running when revealed and outnumbered where we are."""
        if not armed_enemy or not self.mediator.get_is_detected(unit=unit):
            return False
        close_friends: int = len(
            [
                u
                for u in units
                if cy_distance_to_squared(u.position, unit.position) < 144.0
            ]
        )
        if detectors:
            return (
                len(armed_enemy) > close_friends * MAX_ENEMY_PER_TEMPLAR_NEAR_DETECTOR
            )
        return len(armed_enemy) >= close_friends
