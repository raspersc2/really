import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import AMove, KeepUnitSafe
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    ProductionController,
    SpawnController,
    UpgradeController,
)
from cython_extensions.units_utils import cy_closest_to
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit

from bot.combat.base_combat import BaseCombat
from bot.openings.opening_base import OpeningBase

REQUIRED_UPGRADES: list[UpgradeId] = [
    UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
    UpgradeId.PROTOSSAIRARMORSLEVEL1,
    UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
    UpgradeId.PROTOSSAIRARMORSLEVEL2,
    UpgradeId.PROTOSSAIRWEAPONSLEVEL3,
    UpgradeId.PROTOSSAIRARMORSLEVEL3,
]


class Carriers(OpeningBase):
    carrier_combat: BaseCombat
    _main_building_location: Point2

    def __init__(self):
        super().__init__()
        self._aggressive: bool = False

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._main_building_location: Point2 = self.ai.start_location

    async def on_step(self) -> None:
        if not self.ai.build_order_runner.build_completed:
            return

        if not self._aggressive and (self.ai.supply_used > 170 or self.ai.time > 900.0):
            self._aggressive = True

        target: Point2 = self.ai.mediator.get_own_nat
        if self._aggressive:
            target = self.attack_target
        elif ground_threats := self.ai.mediator.get_main_ground_threats_near_townhall:
            target = cy_closest_to(
                self.ai.mediator.get_own_nat, ground_threats
            ).position
        elif air_threats := self.ai.mediator.get_main_air_threats_near_townhall:
            target = cy_closest_to(self.ai.mediator.get_own_nat, air_threats).position

        self.micro(target)

        self._macro()

        self._chrono_boosts({UnitTypeId.STARGATE})

        if self.ai.state.game_loop % 16 == 0:
            self._check_building_location()

    def _check_building_location(self):
        if self.ai.time > 540.0 and self.ai.ready_townhalls:
            self._main_building_location = self.ai.ready_townhalls.furthest_to(
                self.ai.start_location
            ).position

    def _macro(self):
        macro_plan: MacroPlan = MacroPlan()
        if self.ai.mediator.get_own_structures_dict[UnitTypeId.FLEETBEACON]:
            macro_plan.add(
                UpgradeController(
                    upgrade_list=REQUIRED_UPGRADES,
                    base_location=self._main_building_location,
                )
            )
        macro_plan.add(
            AutoSupply(
                base_location=self._main_building_location,
                return_true_if_supply_required=True,
            )
        )
        macro_plan.add(BuildWorkers(min(80, 22 * self.ai.townhalls.amount)))
        macro_plan.add(GasBuildingController(to_count=100, max_pending=2))
        # observers
        if (
            self.ai.supply_used > 120
            and len(self.ai.mediator.get_own_army_dict[UnitTypeId.OBSERVER])
            + self.ai.unit_pending(UnitTypeId.OBSERVER)
            < 1
        ):
            army_composition_dict: dict[UnitTypeId, dict[str, float]] = {
                UnitTypeId.OBSERVER: {"proportion": 1.0, "priority": 0}
            }
            if not self.ai.mediator.get_own_structures_dict[
                UnitTypeId.ROBOTICSFACILITY
            ]:
                macro_plan.add(
                    ProductionController(
                        army_composition_dict,
                        base_location=self._main_building_location,
                    )
                )
            macro_plan.add(
                SpawnController(
                    army_composition_dict,
                )
            )
        # carriers
        army_composition_dict: dict[UnitTypeId, dict[str, float]] = {
            UnitTypeId.CARRIER: {"proportion": 1.0, "priority": 0}
        }
        macro_plan.add(SpawnController(army_composition_dict=army_composition_dict))
        macro_plan.add(
            ProductionController(
                army_composition_dict=army_composition_dict,
                base_location=self._main_building_location,
            )
        )

        if self.ai.unit_pending(UnitTypeId.CARRIER) > 0 or self.ai.time > 320.0:
            max_pending: int = 2 if self.ai.supply_used > 170 else 1
            macro_plan.add(ExpansionController(to_count=100, max_pending=max_pending))
        self.ai.register_behavior(macro_plan)

    def micro(self, target: Point2) -> None:
        own_army_dict: dict[UnitTypeId, list[Unit]] = self.ai.mediator.get_own_army_dict
        observers: list[Unit] = own_army_dict[UnitTypeId.OBSERVER]
        carriers: list[Unit] = own_army_dict[UnitTypeId.CARRIER]

        if carriers:
            grid: np.ndarray = self.ai.mediator.get_air_grid
            avoid_grid: np.ndarray = self.ai.mediator.get_air_avoidance_grid
            observer_target: Point2 = cy_closest_to(target, carriers).position
            for observer in observers:
                observer_maneuver: CombatManeuver = CombatManeuver()
                observer_maneuver.add(KeepUnitSafe(unit=observer, grid=avoid_grid))
                if self.ai.mediator.get_is_detected(unit=observer):
                    observer_maneuver.add(KeepUnitSafe(unit=observer, grid=grid))
                observer_maneuver.add(AMove(unit=observer, target=observer_target))
                self.ai.register_behavior(observer_maneuver)

            for carrier in carriers:
                maneuver: CombatManeuver = CombatManeuver()
                maneuver.add(KeepUnitSafe(unit=carrier, grid=avoid_grid))
                if carrier.shield_percentage < 0.3:
                    maneuver.add(KeepUnitSafe(unit=carrier, grid=grid))
                maneuver.add(AMove(unit=carrier, target=target))
                self.ai.register_behavior(maneuver)
