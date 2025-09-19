import numpy as np
from ares import AresBot
from ares.behaviors.combat.individual import KeepUnitSafe, PathUnitToTarget
from ares.consts import UnitRole, UnitTreeQueryType
from ares.managers.squad_manager import UnitSquad
from cython_extensions.geometry import cy_distance_to_squared
from cython_extensions.units_utils import cy_closest_to
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.worker_combat import WorkerCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from bot.openings.opening_base import OpeningBase


class ProbeRush(OpeningBase):
    worker_combat: BaseCombat
    attack_commenced: bool = False

    def __init__(self):
        super().__init__()
        self._low_shield_tags: set[int] = set()
        self._start_attack_at_time: float = 10
        self._initial_assignment: bool = False
        self._max_probes_in_attack: int = 200
        self._keep_assigning: bool = True
        self._stack_for: float = 1.85

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.worker_combat = WorkerCombat(ai, ai.config, ai.mediator)
        self._opening_specific_settings()

    async def on_step(self) -> None:
        if self.ai.supply_used < 1:
            await self.ai.client.leave()

        if not self.attack_commenced:
            if self.ai.time >= self._start_attack_at_time:
                self.attack_commenced = True
            return

        self._assign_workers()

        # micro
        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=9.0
        )
        if len(squads) == 0:
            return

        pos_of_main_squad: Point2 = self.ai.mediator.get_position_of_main_squad(
            role=UnitRole.ATTACKING
        )

        for squad in squads:
            if self.ai.time < self._start_attack_at_time + self._stack_for:
                mf: Unit = cy_closest_to(self.ai.start_location, self.ai.mineral_field)
                for unit in squad.squad_units:
                    if unit.is_carrying_resource:
                        unit.return_resource()
                    else:
                        unit.gather(mf)
                continue

            target: Point2 = (
                self.attack_target if squad.main_squad else pos_of_main_squad
            )
            close_ground_enemy: Units = self.ai.mediator.get_units_in_range(
                start_points=[squad.squad_position],
                distances=12.5,
                query_tree=UnitTreeQueryType.EnemyGround,
            )[0].filter(lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES)
            self.worker_combat.execute(
                units=squad.squad_units,
                all_close_enemy=close_ground_enemy,
                target=target,
            )

        grid: np.ndarray = self.ai.mediator.get_ground_grid
        mf: Unit = cy_closest_to(self.ai.start_location, self.ai.mineral_field)
        for worker in self.ai.mediator.get_units_from_role(
            role=UnitRole.CONTROL_GROUP_ONE
        ):
            nearby_friendlies: list[Unit] = [
                w
                for w in self.ai.workers
                if cy_distance_to_squared(w.position, worker.position) < 7.5
                and w.tag != worker.tag
            ]

            if len(nearby_friendlies) > 4:
                worker.gather(mf)

            else:
                self.ai.register_behavior(
                    PathUnitToTarget(
                        unit=worker, grid=grid, target=self.ai.game_info.map_center
                    )
                )

    def _opening_specific_settings(self):
        if self.ai.build_order_runner.chosen_opening == "MightBeAWorkerRush":
            self._keep_assigning = False
            self._max_probes_in_attack = 9
            self._start_attack_at_time = 9.0

    def _assign_workers(self):
        if not self._initial_assignment:
            num_assigned: int = 0
            for worker in self.ai.workers:
                if num_assigned >= self._max_probes_in_attack:
                    break

                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.ATTACKING)
                self.ai.mediator.remove_worker_from_mineral(worker_tag=worker.tag)
                num_assigned += 1
            self._initial_assignment = True

        else:
            for worker in self.ai.workers:
                if self._keep_assigning and worker.tag not in self._low_shield_tags:
                    self.ai.mediator.assign_role(
                        tag=worker.tag, role=UnitRole.ATTACKING
                    )
                shield_perc: float = worker.shield_percentage
                if shield_perc < 0.3:
                    self._low_shield_tags.add(worker.tag)
                    self.ai.mediator.assign_role(
                        tag=worker.tag, role=UnitRole.CONTROL_GROUP_ONE
                    )
                elif worker.tag in self._low_shield_tags and shield_perc > 0.99:
                    self._low_shield_tags.remove(worker.tag)
                    self.ai.mediator.assign_role(
                        tag=worker.tag, role=UnitRole.ATTACKING
                    )
