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
    UseAbility,
)
from ares.consts import ALL_STRUCTURES
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import (
    cy_attack_ready,
    cy_center,
    cy_closest_to,
    cy_distance_to,
    cy_in_attack_range,
    cy_pick_enemy_target,
)
from cython_extensions.geometry import cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
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
# prismatic alignment goes up when an armoured unit is this close to beam range
ALIGNMENT_REACH: float = 2.0
# armoured, but they can't hurt us and zerg nearly always have one about, so they'd
# use up the cooldown before anything that matters turns up
ALIGNMENT_IGNORE: set[UnitID] = {
    UnitID.OVERLORD,
    UnitID.OVERLORDCOCOON,
    UnitID.OVERLORDTRANSPORT,
    UnitID.OVERSEER,
    UnitID.OVERSEERSIEGEMODE,
}
SHIELD_BATTERY_RANGE: float = 6.0
# python-sc2 has no weapon for the void ray beam, so its ranges come back as 0
VOID_RAY_RANGE: float = 6.0


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
                # alignment slows a void ray down, it gets back quicker without it,
                # unless a battery already reaches it
                if unit.has_buff(BuffId.VOIDRAYSWARMDAMAGEBOOST) and not any(
                    cy_distance_to(unit.position, b.position)
                    <= SHIELD_BATTERY_RANGE + unit.radius + b.radius
                    for b in close_batteries
                ):
                    attacking_maneuver.add(
                        UseAbility(AbilityId.CANCEL_VOIDRAYPRISMATICALIGNMENT, unit)
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

            # `UseAbility` only casts when it's off cooldown, so never while it's up
            if type_id == UnitID.VOIDRAY and any(
                u.is_armored
                and u.type_id not in ALIGNMENT_IGNORE
                and cy_distance_to(u.position, unit.position)
                <= VOID_RAY_RANGE + unit.radius + u.radius + ALIGNMENT_REACH
                for u in close_enemy
            ):
                attacking_maneuver.add(
                    UseAbility(AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT, unit)
                )

            extra_range: float = VOID_RAY_RANGE if type_id == UnitID.VOIDRAY else 0.0
            # if in range of anything that can harm tempests
            # shoot them
            if danger_to_air := [
                u
                for u in close_enemy
                if (u.can_attack_air or u.type_id in DANGER_TO_AIR)
                and cy_distance_to(u.position, unit.position)
                <= (
                    (unit.ground_range if not u.is_flying else unit.air_range)
                    + extra_range
                    + unit.radius
                    + u.radius
                )
            ]:
                if f_danger := [e for e in danger_to_air if e.is_flying]:
                    e_target: Unit = cy_closest_to(unit.position, f_danger)
                else:
                    e_target: Unit = cy_closest_to(unit.position, danger_to_air)
                if type_id == UnitID.VOIDRAY:
                    attacking_maneuver.add(
                        self._void_ray_attack(unit, f_danger or danger_to_air, e_target)
                    )
                elif e_target and cy_attack_ready(self.ai, unit, e_target):
                    attacking_maneuver.add(AttackTarget(unit=unit, target=e_target))

            # attack any units in range
            if close_enemy:
                # units first, then anything else
                in_attack_range: list[Unit] = cy_in_attack_range(
                    unit, only_enemy_units, extra_range
                ) or cy_in_attack_range(unit, close_enemy, extra_range)
                if in_attack_range and type_id == UnitID.VOIDRAY:
                    attacking_maneuver.add(
                        self._void_ray_attack(
                            unit, in_attack_range, cy_pick_enemy_target(in_attack_range)
                        )
                    )
                elif in_attack_range:
                    # `ShootTargetInRange` will check weapon is ready
                    # otherwise it will not execute
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

    @staticmethod
    def _void_ray_attack(unit: Unit, targets: list[Unit], pick: Unit) -> AttackTarget:
        """`ShootTargetInRange` and `cy_attack_ready` never fire for a void ray, they
        think it has no weapon, so it's told to attack directly.

        It stays on a target it already has rather than being swapped every step.
        """
        return AttackTarget(
            unit=unit,
            target=next((t for t in targets if t.tag == unit.order_target), pick),
        )
