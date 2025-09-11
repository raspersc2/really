from ares import AresBot, UnitRole
from ares.behaviors.macro import AutoSupply, BuildWorkers, MacroPlan, SpawnController
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_closest_to, cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import UnitTreeQueryType

from bot.combat.air_combat import AirCombat
from bot.combat.base_combat import BaseCombat
from bot.combat.probe_proxy_builder import ProbeProxyBuilder
from bot.consts import COMMON_UNIT_IGNORE_TYPES, PROXY_VOID_PLAN
from bot.openings.opening_base import OpeningBase


class ProxyVoids(OpeningBase):
    _begin_proxy_construction: bool
    probe_proxy_builder: BaseCombat
    void_combat: BaseCombat
    _proxy_complete: bool
    _proxy_location: Point2
    _max_proxy_workers: int
    _primary_builder_tag: int
    _proxy_plan: list

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.VOIDRAY:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._begin_proxy_construction = False
        self._proxy_location = self.ai.mediator.get_enemy_third
        self._max_proxy_workers = 1
        self._primary_builder_tag = 0
        self._proxy_complete = False
        self._proxy_plan = PROXY_VOID_PLAN
        self.probe_proxy_builder = ProbeProxyBuilder(ai, ai.config, ai.mediator)
        self.void_combat = AirCombat(ai, ai.config, ai.mediator)

    async def on_step(self) -> None:
        if not self._begin_proxy_construction and [
            s
            for s in self.ai.mediator.get_own_structures_dict[UnitTypeId.GATEWAY]
            if s.build_progress > 0.5
        ]:
            self._begin_proxy_construction = True

        if self._begin_proxy_construction and (
            proxy_probes := self._handle_proxy_probe_assignment(
                self._max_proxy_workers, self._proxy_location
            )
        ):
            await self._handle_proxy_stargate_construction(proxy_probes)

        if self.ai.build_order_runner.build_completed:
            self._macro()
        self._micro()

    def _micro(self) -> None:
        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=9.0
        )
        if len(squads) == 0:
            return

        for squad in squads:
            target: Point2 = self.attack_target
            close_enemy: Units = self.ai.mediator.get_units_in_range(
                start_points=[squad.squad_position],
                distances=11.5,
                query_tree=UnitTreeQueryType.AllEnemy,
            )[0].filter(lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES)
            self.void_combat.execute(
                units=squad.squad_units,
                all_close_enemy=close_enemy,
                target=target,
            )

    def _macro(self) -> None:
        self._chrono_boosts({UnitTypeId.STARGATE})
        self._heal_structures()
        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(
            SpawnController(
                army_composition_dict={
                    UnitTypeId.VOIDRAY: {"proportion": 1.0, "priority": 0}
                }
            )
        )

        macro_plan.add(AutoSupply(self.ai.start_location))
        macro_plan.add(BuildWorkers(18))
        self.ai.register_behavior(macro_plan)

    async def _handle_proxy_stargate_construction(self, proxy_probes: Units) -> None:
        if not self._primary_builder_tag and proxy_probes:
            self._primary_builder_tag = cy_closest_to(
                self._proxy_location, proxy_probes
            ).tag

        idle_sg: list[Unit] = [
            s
            for s in self.ai.mediator.get_own_structures_dict[UnitTypeId.STARGATE]
            if s.is_ready and s.is_idle and s.is_powered
        ]
        next_item_to_build: UnitTypeId | None
        if idle_sg:
            next_item_to_build = None
        else:
            next_item_to_build = self._next_build_target(
                self._proxy_plan, self._proxy_location
            )
        if not next_item_to_build and not idle_sg:
            self._proxy_complete = True

        build_location: Point2 | None = None
        if building_worker := self.ai.unit_tag_dict.get(self._primary_builder_tag):
            if (
                next_item_to_build
                and self.ai.tech_requirement_progress(next_item_to_build) >= 0.99
                and cy_distance_to_squared(
                    building_worker.position, self._proxy_location
                )
                < 144.0
            ):
                build_location = await self.ai.find_placement(
                    building=next_item_to_build, near=self._proxy_location
                )

        self.probe_proxy_builder.execute(
            proxy_probes,
            target=self._proxy_location,
            primary_builder_tag=self._primary_builder_tag,
            next_item_to_build=next_item_to_build,
            build_location=build_location,
        )

    def on_unit_destroyed(self, unit_tag: int) -> None:
        if unit_tag == self._primary_builder_tag:
            self._primary_builder_tag = 0

    def _heal_structures(self):
        require_healing: list[Unit] = [
            s
            for s in self.ai.structures
            if s.shield_percentage < 1.0
            and s.type_id in {UnitTypeId.STARGATE, UnitTypeId.PYLON}
        ]
        for battery in self.ai.mediator.get_own_structures_dict[
            UnitTypeId.SHIELDBATTERY
        ]:
            if battery.is_using_ability(
                AbilityId.SHIELDBATTERYRECHARGEEX5_SHIELDBATTERYRECHARGE
            ):
                continue
            if not battery.is_powered or battery.energy < 1:
                continue
            can_heal: list[Unit] = [
                r
                for r in require_healing
                if cy_distance_to_squared(r.position, battery.position) <= 36.0
            ]
            if can_heal:
                battery(
                    AbilityId.SHIELDBATTERYRECHARGEEX5_SHIELDBATTERYRECHARGE,
                    cy_closest_to(battery.position, can_heal),
                )
