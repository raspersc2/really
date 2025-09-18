from ares import AresBot
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
from cython_extensions import cy_distance_to_squared
from cython_extensions.general_utils import cy_pylon_matrix_covers, cy_unit_pending
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import UnitRole

from bot.combat.base_combat import BaseCombat
from bot.combat.ground_range_combat import GroundRangeCombat
from bot.combat.probe_proxy_builder import ProbeProxyBuilder
from bot.consts import PROXY_4G_PLAN
from bot.openings.opening_base import OpeningBase
from bot.openings.probe_rush import ProbeRush


class MightBeAWorkerRush(OpeningBase):
    _worker_rush_activated: bool
    _main_building_location: Point2
    ground_range_combat: BaseCombat
    probe_rush: OpeningBase
    probe_proxy_builder: BaseCombat
    _proxy_location: Point2
    _proxy_plan: list
    _proxy_finished: bool

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.ground_range_combat = GroundRangeCombat(ai, ai.config, ai.mediator)
        self.probe_proxy_builder = ProbeProxyBuilder(ai, ai.config, ai.mediator)
        self.probe_rush = ProbeRush()
        await self.probe_rush.on_start(ai)
        self._main_building_location = self.ai.start_location
        self._proxy_finished = False
        self._proxy_plan = PROXY_4G_PLAN
        self._proxy_location = self.ai.mediator.get_enemy_third
        if path := self.ai.mediator.find_raw_path(
            start=self.ai.mediator.get_enemy_nat,
            target=self.ai.game_info.map_center,
            grid=self.ai.mediator.get_ground_grid,
            sensitivity=1,
        ):
            if len(path) > 45:
                self._proxy_location = Point2(path[45])

    async def on_step(self) -> None:
        await self.probe_rush.on_step()

        target: Point2 = self.attack_target
        for dt in self.ai.mediator.get_own_army_dict[UnitTypeId.DARKTEMPLAR]:
            dt.attack(target)

        next_item_to_build: UnitTypeId | None = None
        if not self._proxy_finished:
            next_item_to_build = self._next_build_target(
                self._proxy_plan, self._proxy_location
            )
        if not next_item_to_build:
            self._proxy_finished = True
        max_workers: int = 0 if self._proxy_finished else 1
        proxy_probe_building: bool = False
        if self.ai.build_order_runner.build_completed and (
            proxy_probes := self._handle_proxy_probe_assignment(
                max_workers, self._proxy_location
            )
        ):
            if (
                next_item_to_build
                and cy_distance_to_squared(
                    proxy_probes[0].position, self._proxy_location
                )
                < 81.0
            ):
                proxy_probe_building = True
            await self._handle_proxy_gateway_construction(
                next_item_to_build, proxy_probes
            )

        if self.ai.build_order_runner.build_completed and not proxy_probe_building:
            self._macro()

        _target: Point2 = self.attack_target
        self.ground_range_combat.execute(
            self.ai.mediator.get_own_army_dict[UnitTypeId.STALKER], target=_target
        )

    def _macro(self):
        self._chrono_boosts({UnitTypeId.CYBERNETICSCORE, UnitTypeId.WARPGATE})
        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(
            UpgradeController(
                upgrade_list=[UpgradeId.WARPGATERESEARCH],
                base_location=self._main_building_location,
            )
        )
        macro_plan.add(
            AutoSupply(
                base_location=self._main_building_location,
                return_true_if_supply_required=True,
            )
        )
        num_gatherers: int = len(
            self.ai.mediator.get_units_from_role(role=UnitRole.GATHERING)
        )
        if num_gatherers + cy_unit_pending(self.ai, UnitTypeId.PROBE) < 22:
            macro_plan.add(BuildWorkers(31))
        if len(self.ai.structures({UnitTypeId.GATEWAY, UnitTypeId.WARPGATE})) >= 3:
            macro_plan.add(GasBuildingController(to_count=100, max_pending=2))

        # stalkers
        army_composition_dict: dict[UnitTypeId, dict[str, float]] = {
            UnitTypeId.STALKER: {"proportion": 1.0, "priority": 0}
        }
        macro_plan.add(
            SpawnController(
                army_composition_dict=army_composition_dict,
                spawn_target=self.attack_target,
            )
        )
        macro_plan.add(
            ProductionController(
                army_composition_dict=army_composition_dict,
                base_location=self._main_building_location,
            )
        )

        if self.ai.minerals > 600:
            macro_plan.add(ExpansionController(to_count=100, max_pending=1))
        self.ai.register_behavior(macro_plan)

    async def _handle_proxy_gateway_construction(
        self, next_item_to_build: UnitTypeId, proxy_probes: Units
    ) -> None:
        building_worker: Unit = proxy_probes[0]

        build_location: Point2 | None = None
        if (
            next_item_to_build
            and self.ai.tech_requirement_progress(next_item_to_build) >= 0.99
            and cy_distance_to_squared(building_worker.position, self._proxy_location)
            < 144.0
            and (
                next_item_to_build in {UnitTypeId.NEXUS, UnitTypeId.PYLON}
                or cy_pylon_matrix_covers(
                    self._proxy_location,
                    self.ai.mediator.get_own_structures_dict[UnitTypeId.PYLON],
                    self.ai.game_info.terrain_height.data_numpy,
                    pylon_build_progress=1.0,
                )
            )
        ):
            build_location = await self.ai.find_placement(
                building=next_item_to_build, near=self._proxy_location, placement_step=1
            )
        self.probe_proxy_builder.execute(
            proxy_probes,
            target=self._proxy_location,
            primary_builder_tag=building_worker.tag,
            next_item_to_build=next_item_to_build,
            build_location=build_location,
        )
