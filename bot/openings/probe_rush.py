from ares import AresBot, UnitRole
from ares.behaviors.macro import BuildWorkers, Mining
from ares.consts import UnitTreeQueryType
from ares.managers.squad_manager import UnitSquad
from cython_extensions.units_utils import cy_closest_to
from sc2.ids.unit_typeid import UnitTypeId
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

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.worker_combat = WorkerCombat(ai, ai.config, ai.mediator)

    async def on_step(self) -> None:
        if self.ai.supply_workers < 1:
            await self.ai.client.leave()

        if self.ai.supply_left and self.ai.can_afford(UnitTypeId.PROBE):
            self.ai.register_behavior(BuildWorkers(200))

        if not self.attack_commenced:
            if self.ai.time >= 10:
                self.attack_commenced = True
            return

        for worker in self.ai.workers:
            self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.ATTACKING)
            shield_perc: float = worker.shield_percentage
            if shield_perc < 0.05:
                self._low_shield_tags.add(worker.tag)
                self.ai.mediator.assign_role(
                    tag=worker.tag, role=UnitRole.CONTROL_GROUP_ONE
                )
            elif worker.tag in self._low_shield_tags and shield_perc > 0.99:
                self._low_shield_tags.remove(worker.tag)
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.ATTACKING)

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
            if self.ai.time < 11.0:
                mf: Unit = cy_closest_to(self.ai.start_location, self.ai.mineral_field)
                for unit in squad.squad_units:
                    unit.gather(mf)
                continue

            target: Point2 = (
                self.attack_target if squad.main_squad else pos_of_main_squad
            )
            close_ground_enemy: Units = self.ai.mediator.get_units_in_range(
                start_points=[squad.squad_position],
                distances=11.5,
                query_tree=UnitTreeQueryType.EnemyGround,
            )[0].filter(lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES)
            self.worker_combat.execute(
                units=squad.squad_units,
                all_close_enemy=close_ground_enemy,
                target=target,
            )

        for worker in self.ai.mediator.get_units_from_role(
            role=UnitRole.CONTROL_GROUP_ONE
        ):
            if not worker.is_gathering:
                worker.gather(
                    cy_closest_to(self.ai.start_location, self.ai.mineral_field)
                )
