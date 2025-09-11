from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    WorkerKiteBack,
)
from ares.consts import ALL_STRUCTURES
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot

DANGER_TO_AIR: set[UnitID] = {
    UnitID.VOIDRAY,
    UnitID.PHOTONCANNON,
    UnitID.MISSILETURRET,
    UnitID.SPORECRAWLER,
    UnitID.BUNKER,
}


@dataclass
class WorkerCombat(BaseCombat):
    """Execute behavior for Tempest Combat.

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

    def execute(self, units: Union[list[Unit], Units], **kwargs) -> None:
        close_enemy: Units = kwargs["all_close_enemy"]
        target: Point2 = (
            kwargs["target"] if "target" in kwargs else self.ai.enemy_start_locations[0]
        )
        avoid_grid: np.ndarray = self.mediator.get_air_avoidance_grid
        grid: np.ndarray = self.mediator.get_ground_grid

        close_enemy: list[Unit] = [
            u
            for u in close_enemy
            if (not u.is_cloaked or u.is_cloaked and u.is_revealed)
            and (not u.is_burrowed or u.is_burrowed and u.is_visible)
            and not u.is_memory
            and not u.is_snapshot
        ]

        only_enemy_units: list[Unit] = [
            u for u in close_enemy if u.type_id not in ALL_STRUCTURES
        ]

        for unit in units:
            if unit.is_carrying_minerals:
                unit.return_resource()
                continue

            attacking_maneuver: CombatManeuver = CombatManeuver()
            attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
            attacking_maneuver.add(ShootTargetInRange(unit, only_enemy_units))
            if not only_enemy_units:
                attacking_maneuver.add(ShootTargetInRange(unit, close_enemy))
            if close_enemy:
                if unit.shield_health_percentage < 0.35:
                    attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                else:
                    target_unit: Unit
                    if only_enemy_units:
                        target_unit = cy_closest_to(unit.position, only_enemy_units)
                    else:
                        target_unit = cy_closest_to(unit.position, close_enemy)
                    if not self.mediator.is_position_safe(
                        grid=grid, position=unit.position
                    ):
                        attacking_maneuver.add(
                            WorkerKiteBack(unit=unit, target=target_unit)
                        )
                    else:
                        attacking_maneuver.add(AMove(unit=unit, target=target))

            attacking_maneuver.add(
                PathUnitToTarget(unit=unit, target=target, grid=grid)
            )
            self.ai.register_behavior(attacking_maneuver)
