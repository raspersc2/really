from sc2.data import Race

from ares import AresBot
from ares.behaviors.macro import AutoSupply, BuildWorkers, MacroPlan, SpawnController
from ares.consts import UnitRole
from cython_extensions import cy_distance_to_squared, cy_towards
from cython_extensions.units_utils import cy_closest_to
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.probe_proxy_builder import ProbeProxyBuilder
from bot.openings.opening_base import OpeningBase
from bot.openings.probe_rush import ProbeRush


class ProxyNexus(OpeningBase):
    _proxy_location: Point2
    _recall_meetup_point: Point2
    probe_proxy_builder: BaseCombat
    probe_rush: OpeningBase

    def __init__(self):
        super().__init__()
        self._proxy_placed: bool = False
        self._start_attack: bool = False
        self._begin_proxy_construction: bool = False
        self._max_proxy_probes: int = 1
        self._proxy_plan: list = [(UnitTypeId.NEXUS, 1)]
        self._recall_complete: bool = False

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.probe_proxy_builder = ProbeProxyBuilder(ai, ai.config, ai.mediator)
        self._proxy_location = self.ai.mediator.get_enemy_third
        if self.ai.enemy_race in {Race.Random, Race.Zerg}:
            if path := self.ai.mediator.find_raw_path(
                start=self.ai.mediator.get_enemy_nat,
                target=self.ai.game_info.map_center,
                grid=self.ai.mediator.get_ground_grid,
                sensitivity=1,
            ):
                if len(path) > 18:
                    self._proxy_location = Point2(path[18])
        else:
            self._proxy_location = Point2(
                cy_towards(
                    self.ai.mediator.get_primary_nydus_enemy_main,
                    self.ai.enemy_start_locations[0],
                    6.0,
                )
            )
        self._recall_meetup_point = Point2(
            cy_towards(self.ai.start_location, self.ai.main_base_ramp.top_center, 6.5)
        )
        self.probe_rush = ProbeRush()
        await self.probe_rush.on_start(ai)

    async def on_step(self) -> None:
        if not self._start_attack and (
            (
                len(self.ai.townhalls) > 1
                and all(th.build_progress > 0.95 for th in self.ai.townhalls)
            )
            or self.ai.time > 180.0
        ):
            self._start_attack = True

        if not self._begin_proxy_construction and [
            s
            for s in self.ai.mediator.get_own_structures_dict[UnitTypeId.GATEWAY]
            if s.build_progress > 0.0
        ]:
            self._begin_proxy_construction = True

        if self._begin_proxy_construction and (
            proxy_probes := self._handle_proxy_probe_assignment(
                self._max_proxy_probes, self._proxy_location
            )
        ):
            await self._handle_proxy_nexus_construction(proxy_probes)

        if len(self.ai.townhalls) > 1 or self.ai.time > 180.0:
            self._proxy_placed = True

        if self._start_attack:
            await self._micro()

        if self.ai.build_order_runner.build_completed and self._proxy_placed:
            can_chrono: bool = True
            if len(self.ai.townhalls.ready) > 1 and not self._recall_complete:
                can_chrono = False
            if can_chrono:
                self._chrono_boosts({UnitTypeId.GATEWAY})
            macro_plan: MacroPlan = MacroPlan()
            macro_plan.add(
                SpawnController(
                    army_composition_dict={
                        UnitTypeId.ZEALOT: {"proportion": 1.0, "priority": 0}
                    }
                )
            )
            if self.ai.supply_used < 25:
                macro_plan.add(AutoSupply(self.ai.start_location))
            macro_plan.add(BuildWorkers(17))
            self.ai.register_behavior(macro_plan)

    async def _micro(self) -> None:
        for worker in self.ai.workers:
            self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.ATTACKING)

        enough_to_recall: bool = (
            len(
                [
                    u
                    for u in self.ai.units
                    if cy_distance_to_squared(u.position, self._recall_meetup_point)
                    < 42.25
                ]
            )
            >= len(self.ai.units) * 0.92
        )
        if not self._recall_complete:
            # just incase something goes wrong
            if self.ai.time > 185.0:
                self._recall_complete = True

            if enough_to_recall and self.ai.townhalls:
                nexus: Unit = cy_closest_to(self._proxy_location, self.ai.townhalls)
                if AbilityId.EFFECT_MASSRECALL_NEXUS in nexus.abilities:
                    nexus(AbilityId.EFFECT_MASSRECALL_NEXUS, self._recall_meetup_point)
                    self._recall_complete = True
            for unit in self.ai.units:
                unit.move(self._recall_meetup_point)
        else:
            await self.probe_rush.on_step()
            target: Point2 = self.attack_target
            zealots = self.ai.mediator.get_own_army_dict[UnitTypeId.ZEALOT]
            for z in zealots:
                if z.is_idle:
                    z.attack(target)

    async def _handle_proxy_nexus_construction(self, proxy_probes: Units):
        next_item_to_build: UnitTypeId | None = self._next_build_target(
            self._proxy_plan, self._proxy_location
        )
        if not next_item_to_build:
            self._max_proxy_probes = 0
        build_location: Point2 | None = None
        building_worker: Unit = proxy_probes.closest_to(self._proxy_location)
        if (
            next_item_to_build
            and self.ai.tech_requirement_progress(next_item_to_build) >= 0.99
            and cy_distance_to_squared(building_worker.position, self._proxy_location)
            < 144.0
        ):
            build_location = await self.ai.find_placement(
                building=next_item_to_build, near=self._proxy_location
            )

        self.probe_proxy_builder.execute(
            proxy_probes,
            target=self._proxy_location,
            primary_builder_tag=building_worker.tag,
            next_item_to_build=next_item_to_build,
            build_location=build_location,
        )

    def on_unit_cancelled(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.NEXUS:
            self._start_attack = True
            self._recall_complete = True
