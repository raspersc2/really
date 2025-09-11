from ares import AresBot
from ares.behaviors.macro import AutoSupply, BuildWorkers, MacroPlan, SpawnController
from cython_extensions import cy_closest_to, cy_distance_to_squared, cy_towards
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.probe_proxy_builder import ProbeProxyBuilder
from bot.consts import PROXY_ZEALOT_PLAN, PROXY_ZEALOT_PLAN_3G
from bot.openings.opening_base import OpeningBase

PATH_THRESHOLD: int = 100


class ProxyZealot(OpeningBase):
    probe_proxy_builder: BaseCombat
    _proxy_complete: bool
    _proxy_location: Point2
    _max_proxy_workers: int
    _primary_builder_tag: int
    _current_build_location: Point2 | None
    _proxy_plan: list

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.probe_proxy_builder = ProbeProxyBuilder(ai, ai.config, ai.mediator)
        self._max_proxy_workers = 1
        self._proxy_complete = False
        self._proxy_location = self._calculate_proxy_location()
        self._primary_builder_tag = 0
        self._current_build_location = None
        if self.ai.build_order_runner.chosen_opening == "ProxyZealot":
            self._proxy_plan = PROXY_ZEALOT_PLAN
        else:
            self._proxy_plan = PROXY_ZEALOT_PLAN_3G

    async def on_step(self) -> None:
        if self._proxy_complete:
            self._max_proxy_workers = 1
        elif self.ai.time > 30.0:
            self._max_proxy_workers = 2

        if proxy_probes := self._handle_proxy_probe_assignment(
            self._max_proxy_workers, self._proxy_location
        ):
            await self._handle_proxy_zealot_construction(proxy_probes)

        self._chrono_boosts({UnitTypeId.GATEWAY})

        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(
            SpawnController(
                army_composition_dict={
                    UnitTypeId.ZEALOT: {"proportion": 1.0, "priority": 0}
                }
            )
        )
        if self.ai.time > 180.0:
            macro_plan.add(AutoSupply(self.ai.start_location))
        macro_plan.add(BuildWorkers(15))
        self.ai.register_behavior(macro_plan)

        target: Point2 = self.attack_target
        for z in self.ai.mediator.get_own_army_dict[UnitTypeId.ZEALOT]:
            if z.is_idle:
                z.attack(target)

    async def _handle_proxy_zealot_construction(self, proxy_probes: Units) -> None:
        if not self._primary_builder_tag and proxy_probes:
            self._primary_builder_tag = cy_closest_to(
                self._proxy_location, proxy_probes
            ).tag

        next_item_to_build: UnitTypeId | None = self._next_build_target(
            self._proxy_plan, self._proxy_location
        )
        if not next_item_to_build:
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

    def _calculate_proxy_location(self):
        if path := self.ai.mediator.find_raw_path(
            start=self.ai.mediator.get_own_nat,
            target=self.ai.mediator.get_enemy_nat,
            grid=self.ai.mediator.get_ground_grid,
            sensitivity=1,
        ):
            if len(path) <= PATH_THRESHOLD:
                return Point2(
                    cy_towards(
                        self.ai.mediator.get_primary_nydus_enemy_main,
                        self.ai.enemy_start_locations[0],
                        2.0,
                    )
                )
            else:
                if path := self.ai.mediator.find_raw_path(
                    start=self.ai.mediator.get_enemy_nat,
                    target=self.ai.game_info.map_center,
                    grid=self.ai.mediator.get_ground_grid,
                    sensitivity=1,
                ):
                    if len(path) > 18:
                        return Point2(path[18])
        return self.ai.mediator.get_enemy_nat
