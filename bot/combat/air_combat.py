from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    AttackTarget,
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    StutterUnitBack,
)
from ares.consts import ALL_STRUCTURES
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import (
    cy_attack_ready,
    cy_center,
    cy_closest_to,
    cy_distance_to,
    cy_in_attack_range,
)
from cython_extensions.geometry import cy_distance_to_squared
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
class AirCombat(BaseCombat):
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
        grid: np.ndarray = self.mediator.get_air_grid

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

        close_batteries: list[Unit] = [
            s
            for s in self.ai.structures
            if s.type_id == UnitID.SHIELDBATTERY
            and s.is_powered
            and cy_distance_to_squared(s.position, cy_center(units)) < 400.0
        ]

        for unit in units:
            type_id: UnitID = unit.type_id
            attacking_maneuver: CombatManeuver = CombatManeuver()
            attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
            if len(close_batteries) > 0 and unit.shield_percentage < 0.25:
                target_battery: Unit = max(
                    close_batteries, key=lambda b: b.energy, default=close_batteries[0]
                )
                attacking_maneuver.add(
                    PathUnitToTarget(
                        unit=unit,
                        target=target_battery.position,
                        grid=grid,
                        success_at_distance=4.0,
                    )
                )
                attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                self.ai.register_behavior(attacking_maneuver)
                continue

            # if in range of anything that can harm tempests
            # shoot them
            if danger_to_air := [
                u
                for u in close_enemy
                if (u.can_attack_air or u.type_id in DANGER_TO_AIR)
                and cy_distance_to(u.position, unit.position)
                <= (
                    (unit.ground_range if not u.is_flying else unit.air_range)
                    + unit.radius
                    + u.radius
                )
            ]:
                if f_danger := [e for e in danger_to_air if e.is_flying]:
                    e_target: Unit = cy_closest_to(unit.position, f_danger)
                else:
                    e_target: Unit = cy_closest_to(unit.position, danger_to_air)
                if e_target and cy_attack_ready(self.ai, unit, e_target):
                    attacking_maneuver.add(AttackTarget(unit=unit, target=e_target))

            # attack any units in range
            if close_enemy:
                if in_attack_range_e := cy_in_attack_range(unit, only_enemy_units):
                    # `ShootTargetInRange` will check weapon is ready
                    # otherwise it will not execute
                    attacking_maneuver.add(
                        ShootTargetInRange(unit=unit, targets=in_attack_range_e)
                    )
                # then anything else
                elif in_attack_range := cy_in_attack_range(unit, close_enemy):
                    attacking_maneuver.add(
                        ShootTargetInRange(unit=unit, targets=in_attack_range)
                    )
                if type_id == UnitID.VOIDRAY:
                    attacking_maneuver.add(AMove(unit=unit, target=target))
                else:
                    attacking_maneuver.add(
                        StutterUnitBack(
                            unit=unit,
                            target=cy_closest_to(unit.position, close_enemy),
                            grid=grid,
                        )
                    )

            else:
                attacking_maneuver.add(AMove(unit=unit, target=target))

            self.ai.register_behavior(attacking_maneuver)
