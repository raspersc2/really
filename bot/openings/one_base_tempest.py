from ares import AresBot, UnitRole
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
from ares.consts import UnitTreeQueryType
from ares.managers.squad_manager import UnitSquad
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId
from sc2.units import Units

from bot.combat.air_combat import AirCombat
from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from bot.openings.opening_base import OpeningBase


class OneBaseTempest(OpeningBase):
    tempest_combat: BaseCombat

    def __init__(self):
        super().__init__()

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.tempest_combat = AirCombat(ai, ai.config, ai.mediator)

    async def on_step(self) -> None:
        if self.ai.minerals > 700 and not self.ai.build_order_runner.build_completed:
            self.ai.build_order_runner.set_build_completed()

        # role assignment
        for tempest in self.ai.mediator.get_own_army_dict[UnitID.TEMPEST]:
            self.ai.mediator.assign_role(tag=tempest.tag, role=UnitRole.ATTACKING)

        # macro based stuff
        if self.ai.build_order_runner.build_completed:
            self._do_tempest_macro_plan()
            self._handle_chrono_boosts()
        else:
            self.ai.register_behavior(
                SpawnController(
                    army_composition_dict={
                        UnitID.ZEALOT: {"proportion": 0.5, "priority": 1}
                    },
                    freeflow_mode=True,
                )
            )
            if self.ai.supply_workers >= 16:
                self.ai.register_behavior(BuildWorkers(22))

        # micro
        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=9.0
        )
        if len(squads) == 0:
            return

        for squad in squads:
            all_close_enemy: Units = self.ai.mediator.get_units_in_range(
                start_points=[squad.squad_position],
                distances=13.5,
                query_tree=UnitTreeQueryType.AllEnemy,
            )[0].filter(lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES)
            self.tempest_combat.execute(
                units=squad.squad_units,
                all_close_enemy=all_close_enemy,
                target=self.attack_target,
            )

        for z in self.ai.mediator.get_own_army_dict[UnitID.ZEALOT]:
            z.attack(self.attack_target)

    def _do_tempest_macro_plan(self):
        # Set up macro plan to keep producing Tempests once tech is ready.
        self._macro_plan = MacroPlan()
        self._macro_plan.add(AutoSupply(self.ai.start_location))
        self._macro_plan.add(BuildWorkers(min(80, 22 * self.ai.townhalls.amount)))

        army_comp = {
            UnitID.TEMPEST: {"proportion": 0.9, "priority": 0},
            UnitID.ZEALOT: {"proportion": 0.1, "priority": 1},
        }
        self._macro_plan.add(
            SpawnController(army_composition_dict=army_comp, freeflow_mode=True)
        )
        self._macro_plan.add(
            ProductionController(
                army_composition_dict=army_comp, base_location=self.ai.start_location
            )
        )

        if self.ai.minerals > 600:
            self._macro_plan.add(ExpansionController(to_count=100))

        if len(self.ai.gas_buildings) >= 4:
            self._macro_plan.add(
                UpgradeController(
                    [
                        UpgradeId.TEMPESTGROUNDATTACKUPGRADE,
                        UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
                        UpgradeId.PROTOSSAIRARMORSLEVEL1,
                        UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
                        UpgradeId.PROTOSSAIRARMORSLEVEL2,
                        UpgradeId.PROTOSSAIRWEAPONSLEVEL3,
                        UpgradeId.PROTOSSAIRARMORSLEVEL3,
                    ],
                    self.ai.start_location,
                )
            )

        self._macro_plan.add(GasBuildingController(to_count=100))

        self.ai.register_behavior(self._macro_plan)

    def _handle_chrono_boosts(self):
        if self.ai.build_order_runner.build_completed:
            if available_nexuses := [
                th for th in self.ai.townhalls if th.energy >= 50 and th.is_ready
            ]:
                if targets := [
                    s
                    for s in self.ai.structures
                    if s.is_ready
                    and not s.is_idle
                    and s.type_id == UnitID.STARGATE
                    and not s.has_buff(BuffId.CHRONOBOOSTENERGYCOST)
                    and s.orders
                    and s.orders[0].progress < 0.4
                    and (s.is_powered or s.type_id == UnitID.NEXUS)
                ]:
                    available_nexuses[0](
                        AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, targets[0]
                    )
