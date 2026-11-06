from ares import AresBot, UnitRole
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    GasBuildingController,
    MacroPlan,
    SpawnController,
)
from ares.consts import UnitTreeQueryType
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_closest_to, cy_distance_to_squared, cy_towards
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.air_combat import AirCombat
from bot.combat.base_combat import BaseCombat
from bot.combat.probe_proxy_builder import ProbeProxyBuilder
from bot.consts import COMMON_UNIT_IGNORE_TYPES, NATURAL_TAKEOVER_PLAN
from bot.openings.one_base_tempest import OneBaseTempest
from bot.openings.opening_base import OpeningBase

# a second probe follows the first, so a pull or a lost probe doesn't stop building
PROXY_PROBES: int = 2
SECOND_PROBE_TIME: float = 30.0
# everything but the blocking pylon is placed around a spot this far from their
# townhall spot away from their ramp, so less of it is in range of the high ground
RAMP_CLEARANCE: float = 3.0
# cannons and batteries sit on the ramp side of that, covering the pylons and
# production further back
FRONT_DISTANCE: float = 5.0
BACK_DISTANCE: float = 4.0
# anything of ours closer than this to their townhall spot counts as at the natural
NATURAL_RADIUS: float = 18.0
# nothing left at the natural after this and the cheese is over, go tempests at home
TAKEOVER_LOST_TIME: float = 150.0
# a placement search asks the game about every ring out to 20, so one that fails
# isn't tried again every step
PLACEMENT_RETRY_SECONDS: float = 2.0
# gas waits on the first cannon so the first structures aren't short of minerals,
# or this long if cannons are held up, so the stargate isn't kept waiting for gas
GAS_BUILDINGS: int = 2
GAS_DEADLINE: float = 120.0
WORKER_LIMIT: int = 20
THREE_BY_THREE: set[UnitTypeId] = {
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.FORGE,
    UnitTypeId.GATEWAY,
    UnitTypeId.STARGATE,
}
FRONT_LINE: set[UnitTypeId] = {UnitTypeId.PHOTONCANNON, UnitTypeId.SHIELDBATTERY}
HEALED_STRUCTURES: set[UnitTypeId] = {
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.FORGE,
    UnitTypeId.GATEWAY,
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.PYLON,
    UnitTypeId.SHIELDBATTERY,
    UnitTypeId.STARGATE,
}


class NaturalTakeover(OpeningBase):
    """Block the enemy natural with a pylon, then build everything else there too.

    Plays the same against every race: a probe leaves straight away, puts its first
    pylon on their townhall spot, and fills the natural with cannons, batteries and
    a stargate making void rays.
    """

    air_combat: BaseCombat
    probe_proxy_builder: BaseCombat
    tempest_transition: OpeningBase
    _gas_started: bool
    _takeover_lost: bool
    _primary_builder_tag: int
    _placement_retry_at: float
    _planned_spot: tuple[UnitTypeId, Point2] | None
    _natural: Point2
    _middle: Point2
    _front: Point2
    _back: Point2

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.VOIDRAY:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    def on_unit_destroyed(self, unit_tag: int) -> None:
        if unit_tag == self._primary_builder_tag:
            self._primary_builder_tag = 0

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.air_combat = AirCombat(ai, ai.config, ai.mediator)
        self.probe_proxy_builder = ProbeProxyBuilder(ai, ai.config, ai.mediator)
        self.tempest_transition = OneBaseTempest()
        await self.tempest_transition.on_start(ai)
        self._gas_started = False
        self._takeover_lost = False
        self._primary_builder_tag = 0
        self._placement_retry_at = 0.0
        self._planned_spot = None
        self._natural = self.ai.mediator.get_enemy_nat
        ramp: Point2 = self.ai.mediator.get_enemy_ramp.bottom_center
        self._middle = Point2(cy_towards(self._natural, ramp, -RAMP_CLEARANCE))
        self._front = Point2(cy_towards(self._middle, ramp, FRONT_DISTANCE))
        self._back = Point2(cy_towards(self._middle, ramp, -BACK_DISTANCE))

    async def on_step(self) -> None:
        self._check_takeover_lost()
        if self._takeover_lost:
            await self.tempest_transition.on_step()
            return

        await self._handle_takeover_construction()
        if self.ai.build_order_runner.build_completed:
            self._macro()
        self._micro()

    def _check_takeover_lost(self) -> None:
        if self._takeover_lost or self.ai.time < TAKEOVER_LOST_TIME:
            return
        if not [
            s
            for s in self.ai.structures
            if cy_distance_to_squared(s.position, self._natural) < NATURAL_RADIUS**2
        ]:
            self._takeover_lost = True
            for probe in self.ai.mediator.get_units_from_role(
                role=UnitRole.PROXY_WORKER
            ):
                self.ai.mediator.assign_role(tag=probe.tag, role=UnitRole.GATHERING)

    async def _handle_takeover_construction(self) -> None:
        """Runs from the first step, the first probe leaves as the game starts.

        Destroyed structures drop the count at the natural, so the plan rebuilds them.
        """
        proxy_probes: Units = self._handle_proxy_probe_assignment(
            PROXY_PROBES if self.ai.time > SECOND_PROBE_TIME else 1, self._natural
        )
        if not proxy_probes:
            return
        if self._primary_builder_tag not in self.ai.unit_tag_dict:
            self._primary_builder_tag = cy_closest_to(self._natural, proxy_probes).tag

        next_item_to_build: UnitTypeId | None = self._next_build_target(
            NATURAL_TAKEOVER_PLAN, self._natural
        )
        # a stargate sitting idle gets everything, even while saving up for a void ray
        if [
            s
            for s in self.ai.mediator.get_own_structures_dict[UnitTypeId.STARGATE]
            if s.is_ready and s.is_idle and s.is_powered
        ]:
            next_item_to_build = None

        build_location: Point2 | None = None
        if (
            (building_worker := self.ai.unit_tag_dict.get(self._primary_builder_tag))
            and next_item_to_build
            and self.ai.can_afford(next_item_to_build)
            and self.ai.tech_requirement_progress(next_item_to_build) >= 0.99
            and cy_distance_to_squared(building_worker.position, self._natural) < 144.0
            # nothing else can go down until a pylon there finishes
            and (next_item_to_build == UnitTypeId.PYLON or self._has_power)
            and self.ai.time >= self._placement_retry_at
        ):
            build_location = await self._placement(next_item_to_build)
            if not build_location:
                self._placement_retry_at = self.ai.time + PLACEMENT_RETRY_SECONDS

        self.probe_proxy_builder.execute(
            proxy_probes,
            target=self._natural,
            primary_builder_tag=self._primary_builder_tag,
            next_item_to_build=next_item_to_build,
            build_location=build_location,
        )

    @property
    def _has_power(self) -> bool:
        return any(
            p.is_ready
            and cy_distance_to_squared(p.position, self._natural) < NATURAL_RADIUS**2
            for p in self.ai.mediator.get_own_structures_dict[UnitTypeId.PYLON]
        )

    async def _placement(self, structure: UnitTypeId) -> Point2 | None:
        """Keeps the spot while the probe walks there, checking it is still free.

        That's one query a step rather than a whole search.
        """
        if (
            self._planned_spot
            and self._planned_spot[0] == structure
            and await self.ai.can_place_single(structure, self._planned_spot[1])
        ):
            return self._planned_spot[1]
        spot: Point2 | None = await self._find_spot(structure)
        self._planned_spot = (structure, spot) if spot else None
        return spot

    async def _find_spot(self, structure: UnitTypeId) -> Point2 | None:
        """The first pylon on their townhall spot, then defence in front of the rest."""
        if structure == UnitTypeId.PYLON and not self._count_started_at_proxy(
            UnitTypeId.PYLON, self._natural, NATURAL_RADIUS
        ):
            # anywhere on the spot stops a townhall going down, search no further
            if spot := await self.ai.find_placement(
                UnitTypeId.PYLON,
                near=self._natural.rounded,
                max_distance=3,
                placement_step=1,
            ):
                return spot

        near: Point2 = self._middle
        if structure in FRONT_LINE:
            near = self._front
        elif structure == UnitTypeId.PYLON:
            near = self._back
        # 3x3 structures sit on half coordinates, a search from a whole one never fits
        near = (
            near.rounded.offset((0.5, 0.5))
            if structure in THREE_BY_THREE
            else near.rounded
        )
        return await self.ai.find_placement(structure, near=near, placement_step=1)

    def _macro(self) -> None:
        self._chrono_boosts({UnitTypeId.STARGATE})
        self._heal_structures(HEALED_STRUCTURES)
        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(
            SpawnController(
                army_composition_dict={
                    UnitTypeId.VOIDRAY: {"proportion": 1.0, "priority": 0}
                }
            )
        )
        # once started it stays started, the cannon can die and be rebuilt later
        self._gas_started = (
            self._gas_started
            or self.ai.time > GAS_DEADLINE
            or bool(
                self._count_started_at_proxy(
                    UnitTypeId.PHOTONCANNON, self._natural, NATURAL_RADIUS
                )
            )
        )
        if self._gas_started:
            macro_plan.add(GasBuildingController(to_count=GAS_BUILDINGS))
        # until then the pylons at the natural are our supply, a pylon at home this
        # early would hold the forge back
        if self.ai.mediator.get_own_structures_dict[UnitTypeId.STARGATE]:
            macro_plan.add(AutoSupply(self.ai.start_location))
        macro_plan.add(BuildWorkers(WORKER_LIMIT))
        self.ai.register_behavior(macro_plan)

    def _micro(self) -> None:
        # each void ray goes in as soon as it's out, the cannons hold the natural
        target: Point2 = self.attack_target

        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=9.0
        )
        for squad in squads:
            close_enemy: Units = self.ai.mediator.get_units_in_range(
                start_points=[squad.squad_position],
                distances=11.5,
                query_tree=UnitTreeQueryType.AllEnemy,
            )[0].filter(lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES)
            self.air_combat.execute(
                units=squad.squad_units, all_close_enemy=close_enemy, target=target
            )
