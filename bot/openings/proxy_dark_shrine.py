import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    DropCargo,
    KeepUnitSafe,
    PathUnitToTarget,
    PickUpCargo,
)
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    ProductionController,
    SpawnController,
)
from ares.consts import (
    ID,
    TARGET,
    TOWNHALL_TYPES,
    WORKER_TYPES,
    UnitRole,
    UnitTreeQueryType,
)
from cython_extensions import (
    cy_center,
    cy_closest_to,
    cy_distance_to_squared,
    cy_towards,
    cy_unit_pending,
)
from cython_extensions.general_utils import cy_pylon_matrix_covers
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from scipy.ndimage import label

from bot.combat.base_combat import BaseCombat
from bot.combat.dark_templar_combat import DarkTemplarCombat
from bot.combat.probe_proxy_builder import ProbeProxyBuilder
from bot.consts import DETECTION_STRUCTURES, PROXY_DARK_SHRINE_PLAN
from bot.openings.one_base_tempest import OneBaseTempest
from bot.openings.opening_base import OpeningBase

# zealots only get what gas leaves over, so the bank doesn't sit unused
ARMY_COMPOSITION: dict[UnitTypeId, dict[str, float]] = {
    UnitTypeId.DARKTEMPLAR: {"proportion": 0.8, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.2, "priority": 1},
}
# below this a force at home isn't worth giving up a wave for
HOME_DEFENCE_SUPPLY: float = 4.0
# warp ins closer to the enemy than this get killed before they finish, mains can
# be small with only a couple of pylons in them, so this can't be too cautious
HOME_WARP_SAFE_DISTANCE: float = 8.0
# how far out from a townhall counts as at home, for pylons and for defenders
HOME_DISTANCE: float = 30.0
# threats come and go around the limit, defenders shouldn't turn round each time
HOME_DEFENCE_HOLD_SECONDS: float = 5.0
# the build order can't place everything on some maps, `_ensure_tech` takes over
BUILD_ORDER_DEADLINE: float = 170.0
# all warped in at once, which is also exactly what the prism can carry
FIRST_WAVE_SIZE: int = 4
# stop holding the first wave back for gates or resources at this point
FIRST_WAVE_DEADLINE: float = 270.0
# detection is what actually stops Dark Templar, so it counts for more than any unit
DETECTOR_DEFENCE: int = 4
# don't let the target flip while units are walking there
RETARGET_SECONDS: float = 10.0
# tempests get first call on everything, only what's left over goes on zealots
ZEALOT_BANK_MINERALS: int = 700
MAX_HARASS_ZEALOTS: int = 8
HARASS_COMPOSITION: dict[UnitTypeId, dict[str, float]] = {
    UnitTypeId.ZEALOT: {"proportion": 1.0, "priority": 0},
}
# gives gates a moment to start on a new field, after that it only waits on warp ins
PRISM_PHASE_SECONDS: float = 2.0
# cells either side of the prism that are always inside its field
PRISM_FIELD_CELLS: int = 3
# warp ins crowd into what ground there is, and land back home if none of it fits
MIN_WARP_ROOM: float = 0.8
# warping units are visible, closer than this to a base or an army they get seen
WARP_CLEAR_DISTANCE: float = 15.0
# how far from the main to look for a warp spot, nearest first
DROP_SPOT_REACHES: tuple[float, ...] = (7.0, 14.0, 21.0, 28.0, 35.0)
# the prism beats the shrine there, this far back it doesn't sit among their units
PRISM_HOLD_DISTANCE: float = 40.0
# anti air can keep the prism away, don't hold gates back for it longer than this
PRISM_WAIT_SECONDS: float = 30.0
# once the enemy shows detection after this we move on to tempests
TRANSITION_IF_DETECTED_TIME: float = 300.0
# and we move on regardless at this point
TRANSITION_TIME: float = 420.0


class ProxyDarkShrine(OpeningBase):
    dark_templar_combat: BaseCombat
    probe_proxy_builder: BaseCombat
    tempest_transition: OpeningBase
    _begin_proxy_construction: bool
    _detection_seen: bool
    _first_wave_sent: bool
    _initial_attack_over: bool
    _proxy_finished: bool
    _transitioned: bool
    _warp_at_prism: bool
    _enemy_main_points: set[tuple[int, int]]
    _scouted: set[Point2]
    _proxy_location: Point2
    _attack_base: Point2
    _defend_at: Point2
    _pickup_spot: Point2
    _prism_hold: Point2
    _prism_hover: Point2
    _prism_stage: Point2
    _warp_spot: Point2
    _phased_at: float
    _prism_deadline: float
    _defend_until: float
    _retarget_at: float

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.dark_templar_combat = DarkTemplarCombat(ai, ai.config, ai.mediator)
        self.probe_proxy_builder = ProbeProxyBuilder(ai, ai.config, ai.mediator)
        self.tempest_transition = OneBaseTempest()
        await self.tempest_transition.on_start(ai)
        self.tempest_transition.controls_zealots = False
        self._begin_proxy_construction = False
        self._detection_seen = False
        self._first_wave_sent = False
        self._initial_attack_over = False
        self._proxy_finished = False
        self._transitioned = False
        self._warp_at_prism = False
        self._proxy_location = self._find_hidden_corner()
        self._enemy_main_points = self.ai.mediator.get_flood_fill_area(
            start_point=self.ai.enemy_start_locations[0], max_dist=45
        )
        self._warp_spot, self._prism_hover = self._find_drop_spots()
        self._prism_stage = self._closest_map_edge(self._warp_spot)
        self._prism_hold = Point2(
            cy_towards(self._warp_spot, self.ai.start_location, PRISM_HOLD_DISTANCE)
        )
        self._pickup_spot = self._warp_spot
        self._attack_base = self.ai.enemy_start_locations[0]
        self._defend_at = self.ai.start_location
        self._defend_until = 0.0
        self._scouted = set()
        self._phased_at = 0.0
        self._prism_deadline = 0.0
        self._retarget_at = 0.0

    async def on_step(self) -> None:
        if (
            self.ai.minerals > 500 or self.ai.time > BUILD_ORDER_DEADLINE
        ) and not self.ai.build_order_runner.build_completed:
            self.ai.build_order_runner.set_build_completed()

        if (
            not self._begin_proxy_construction
            and self.ai.mediator.get_own_structures_dict[UnitTypeId.CYBERNETICSCORE]
        ):
            self._begin_proxy_construction = True

        if self._begin_proxy_construction:
            await self._handle_proxy_construction()

        own_army: dict = self.ai.mediator.get_own_army_dict
        # warp ins can still be cancelled, the first wave isn't sent until one lands
        self._first_wave_sent = self._first_wave_sent or any(
            u.is_ready for u in own_army[UnitTypeId.DARKTEMPLAR]
        )
        self._check_initial_attack()
        self._check_tempest_transition()
        targets: list[Point2] = self._attack_targets
        drop_point: Point2 = (
            targets[0] if self._initial_attack_over else self._prism_hover
        )
        units_to_drop: list[Unit] = self._units_to_drop(drop_point)
        # decided before anything spends the money, or the prism packs up on the same
        # step as a warp in starts and cancels it. Gates come back one by one, so
        # anyone who has landed goes rather than waiting on the next
        self._warp_at_prism = (
            self._warp_ready
            and not self._home_warp_spot
            and not any(u.is_ready for u in units_to_drop)
        )
        if self._transitioned:
            await self.tempest_transition.on_step()
            if home_warp_spot := self._home_warp_spot:
                self.ai.register_behavior(self._spawn_army(home_warp_spot))
            elif own_army[UnitTypeId.WARPPRISMPHASING] and self._warp_at_prism:
                self.ai.register_behavior(
                    self._spawn_army(
                        own_army[UnitTypeId.WARPPRISMPHASING][0].position,
                        HARASS_COMPOSITION,
                    )
                )
        elif self.ai.build_order_runner.build_completed:
            await self._ensure_tech()
            self._macro()

        self._handle_warp_prisms(units_to_drop, drop_point)

        drop_tags: set[int] = {u.tag for u in units_to_drop}
        if self._force_at_home:
            self._defend_at = Point2(cy_center(self._home_threats))
            self._defend_until = self.ai.time + HOME_DEFENCE_HOLD_SECONDS
        defend_at: Point2 | None = (
            self._defend_at if self.ai.time < self._defend_until else None
        )
        defenders: list[Unit] = []
        attackers: list[Unit] = []
        for dark_templar in own_army[UnitTypeId.DARKTEMPLAR]:
            if dark_templar.tag not in drop_tags:
                (
                    defenders
                    if defend_at and self._is_home(dark_templar)
                    else attackers
                ).append(dark_templar)
        retreat_targets: list[Point2] = [*self._enemy_bases, self._proxy_location]
        if attackers:
            self.dark_templar_combat.execute(
                attackers, targets=targets, retreat_targets=retreat_targets
            )
        if defenders:
            self.dark_templar_combat.execute(
                defenders, targets=[defend_at], retreat_targets=retreat_targets
            )
        for zealot in own_army[UnitTypeId.ZEALOT]:
            if zealot.is_idle and zealot.tag not in drop_tags:
                zealot.attack(
                    defend_at if defend_at and self._is_home(zealot) else targets[0]
                )

    def _check_initial_attack(self) -> None:
        """The first wave goes for the main until it dies or the main does."""
        if self._initial_attack_over or not self._first_wave_sent:
            return
        own_army: dict = self.ai.mediator.get_own_army_dict
        enemy_main: Point2 = self.ai.enemy_start_locations[0]
        main_destroyed: bool = self.ai.is_visible(enemy_main) and not [
            s
            for s in self.ai.enemy_structures
            if s.type_id in TOWNHALL_TYPES and s.position == enemy_main
        ]
        # anything inside a prism isn't in the army dict
        self._initial_attack_over = main_destroyed or not (
            own_army[UnitTypeId.DARKTEMPLAR]
            or any(p.has_cargo for p in own_army[UnitTypeId.WARPPRISM])
        )

    @property
    def _enemy_bases(self) -> list[Point2]:
        """Every base we know about, or any building at all if we know of none."""
        enemy_structures: list[Unit] = [
            s for s in self.ai.enemy_structures if not s.is_flying
        ]
        bases: list[Unit] = [s for s in enemy_structures if s.type_id in TOWNHALL_TYPES]
        return sorted(
            [s.position for s in bases or enemy_structures],
            key=lambda p: cy_distance_to_squared(p, self.ai.enemy_start_locations[0]),
        )

    @property
    def _attack_targets(self) -> list[Point2]:
        # never `attack_target`, that chases the enemy army rather than their bases
        if not self._initial_attack_over:
            return [self.ai.enemy_start_locations[0]]
        if bases := self._enemy_bases:
            if self._attack_base not in bases or self.ai.time > self._retarget_at:
                self._attack_base = self._least_defended(bases)
                self._retarget_at = self.ai.time + RETARGET_SECONDS
            return [self._attack_base]
        # nothing known, split up over every base on the map to find what's left
        return [
            p
            for p in (
                self.ai.enemy_start_locations[0],
                self.ai.mediator.get_enemy_nat,
            )
            if not self.ai.is_visible(p)
        ] or list(self.ai.expansion_locations_list)

    def _least_defended(self, bases: list[Point2]) -> Point2:
        def defence(enemy: Units) -> int:
            return sum(
                DETECTOR_DEFENCE
                if u.is_detector or u.type_id in DETECTION_STRUCTURES
                else 1
                for u in enemy
                if u.is_detector
                or u.type_id in DETECTION_STRUCTURES
                or (u.can_attack_ground and u.type_id not in WORKER_TYPES)
            )

        near_bases: list[Units] = self.ai.mediator.get_units_in_range(
            start_points=bases,
            distances=15,
            query_tree=UnitTreeQueryType.AllEnemy,
        )
        defences: dict[Point2, int] = {
            base: defence(enemy) for base, enemy in zip(bases, near_bases)
        }
        least_defended: Point2 = min(defences, key=defences.get)
        # enemy we can't see any more fade from this, a small gap isn't worth the walk
        if (
            self._attack_base in defences
            and defences[self._attack_base] - defences[least_defended]
            < DETECTOR_DEFENCE
        ):
            return self._attack_base
        return least_defended

    @property
    def _home_threats(self) -> list[Unit]:
        return [
            u
            for u in self.ai.mediator.get_main_ground_threats_near_townhall
            if u.type_id not in WORKER_TYPES
        ]

    @property
    def _force_at_home(self) -> bool:
        return (
            sum(
                self.ai.game_data.units[u.type_id.value]._proto.food_required
                for u in self._home_threats
            )
            > HOME_DEFENCE_SUPPLY
        )

    def _is_home(self, unit: Unit) -> bool:
        return (
            bool(self.ai.townhalls)
            and self.ai.townhalls.closest_distance_to(unit.position) < HOME_DISTANCE
        )

    @property
    def _home_warp_spot(self) -> Point2 | None:
        """The pylon at home furthest from a force there, if any is safe to use."""
        if not self._force_at_home:
            return None
        threats: list[Unit] = self._home_threats
        distances: dict[Point2, float] = {
            p.position: cy_distance_to_squared(
                p.position, cy_closest_to(p.position, threats).position
            )
            for p in self.ai.mediator.get_own_structures_dict[UnitTypeId.PYLON]
            if p.is_ready and self._is_home(p)
        }
        if not distances:
            return None
        furthest: Point2 = max(distances, key=distances.get)
        if distances[furthest] < HOME_WARP_SAFE_DISTANCE**2:
            return None
        return furthest

    def _units_to_drop(self, drop_point: Point2) -> list[Unit]:
        """Warped in by the prism, waiting for it to lift them to `drop_point`.

        Nothing is held once the prism is dead, so these go back to walking in, or
        if a base is close enough that lifting them would just be slower. The first
        drop is up a cliff, it can be a long walk round however close that looks.
        """
        own_army: dict = self.ai.mediator.get_own_army_dict
        if not (
            own_army[UnitTypeId.WARPPRISM] or own_army[UnitTypeId.WARPPRISMPHASING]
        ) or (
            self._initial_attack_over
            and cy_distance_to_squared(self._pickup_spot, drop_point) < 400.0
        ):
            return []
        return [
            u
            for u in [*own_army[UnitTypeId.DARKTEMPLAR], *own_army[UnitTypeId.ZEALOT]]
            if cy_distance_to_squared(u.position, self._pickup_spot) < 81.0
            and not self._in_enemy_main(u.position)
        ]

    def _in_enemy_main(self, position: Point2) -> bool:
        return (round(position.x), round(position.y)) in self._enemy_main_points

    def _can_warp_at(self, position: Point2) -> bool:
        """Units are visible while warping in, so stay clear of armies and bases.

        Having to pull the prism out early cancels everything part way through.
        """
        # the spot itself is the best we know of, so there's no looking any further
        has_room: bool = (
            self._warp_room(position) >= MIN_WARP_ROOM
            or cy_distance_to_squared(position, self._warp_spot) < 4.0
        )
        return (
            has_room
            and self.ai.mediator.is_position_safe(
                grid=self.ai.mediator.get_ground_grid, position=position
            )
            and not [
                u
                for u in self.ai.mediator.get_units_in_range(
                    start_points=[position],
                    distances=WARP_CLEAR_DISTANCE,
                    query_tree=UnitTreeQueryType.AllEnemy,
                )[0]
                if u.type_id in TOWNHALL_TYPES
                or (u.can_attack_ground and u.type_id not in WORKER_TYPES)
            ]
        )

    def _warp_room(self, position: Point2) -> float:
        """Share of the prism's field that's ground, `is_position_safe` can't say.

        Unpathable cells are infinite on the grid, and that counts as safe.
        """
        x, y = round(position.x), round(position.y)
        field: np.ndarray = self.ai.mediator.get_ground_grid[
            max(x - PRISM_FIELD_CELLS, 0) : x + PRISM_FIELD_CELLS + 1,
            max(y - PRISM_FIELD_CELLS, 0) : y + PRISM_FIELD_CELLS + 1,
        ]
        return float(np.mean(field < np.inf))

    @property
    def _warping_in(self) -> bool:
        """Morphing the prism back would cancel anything still warping in on it."""
        own_army: dict = self.ai.mediator.get_own_army_dict
        return any(
            not u.is_ready
            and cy_distance_to_squared(u.position, self._pickup_spot) < 81.0
            for u in [*own_army[UnitTypeId.DARKTEMPLAR], *own_army[UnitTypeId.ZEALOT]]
        )

    @property
    def _ready_warp_gates(self) -> list[Unit]:
        return [
            gate
            for gate in self.ai.mediator.get_own_structures_dict[UnitTypeId.WARPGATE]
            if AbilityId.WARPGATETRAIN_STALKER in gate.abilities
        ]

    @property
    def _first_wave_ready(self) -> bool:
        """Everything lands together, four at once is much harder to clean up."""
        if self._first_wave_sent or self.ai.time > FIRST_WAVE_DEADLINE:
            return True
        cost = self.ai.calculate_cost(UnitTypeId.DARKTEMPLAR)
        return (
            len(self._ready_warp_gates) >= FIRST_WAVE_SIZE
            and self.ai.minerals >= cost.minerals * FIRST_WAVE_SIZE
            and self.ai.vespene >= cost.vespene * FIRST_WAVE_SIZE
            and self.ai.supply_left >= 2 * FIRST_WAVE_SIZE
        )

    @property
    def _warp_ready(self) -> bool:
        if not self._ready_warp_gates or self.ai.supply_left < 2:
            return False
        if self._transitioned:
            return (
                self.ai.minerals > ZEALOT_BANK_MINERALS
                and len(self.ai.mediator.get_own_army_dict[UnitTypeId.ZEALOT])
                < MAX_HARASS_ZEALOTS
            )
        return (
            self.ai.tech_ready_for_unit(UnitTypeId.DARKTEMPLAR)
            and self.ai.can_afford(UnitTypeId.DARKTEMPLAR)
            and self._first_wave_ready
        )

    @property
    def _spawn_target(self) -> Point2 | None:
        """Where to warp in, `None` if we should hold for the prism to get in place."""
        own_army: dict = self.ai.mediator.get_own_army_dict
        if home_warp_spot := self._home_warp_spot:
            return home_warp_spot
        if not self.ai.tech_ready_for_unit(UnitTypeId.DARKTEMPLAR):
            return None
        if not self._first_wave_ready:
            return None
        if phasing_prisms := own_army[UnitTypeId.WARPPRISMPHASING]:
            return phasing_prisms[0].position if self._warp_at_prism else None
        if self.ai.time < self._prism_deadline and (
            own_army[UnitTypeId.WARPPRISM]
            or cy_unit_pending(self.ai, UnitTypeId.WARPPRISM)
        ):
            return None
        # prism can't help right now, warp in at the proxy and walk
        return self._proxy_location

    def _find_hidden_corner(self) -> Point2:
        """Somewhere buildable near a map corner that isn't either main."""
        area = self.ai.game_info.playable_area
        corners: list[Point2] = [
            Point2((x, y))
            for x in (area.x, area.x + area.width)
            for y in (area.y, area.y + area.height)
        ]
        for main in (self.ai.start_location, self.ai.enemy_start_locations[0]):
            corners.remove(min(corners, key=lambda c: cy_distance_to_squared(c, main)))

        ground_grid: np.ndarray = self.ai.mediator.get_ground_grid
        own_nat: Point2 = self.ai.mediator.get_own_nat
        enemy_third: Point2 = self.ai.mediator.get_enemy_third
        ys, xs = np.nonzero(self.ai.game_info.placement_grid.data_numpy)
        spots: np.ndarray = np.column_stack((xs, ys))
        expansions: np.ndarray = np.array(self.ai.expansion_locations_list)
        # anyone scouting base locations shouldn't see this
        spots = spots[
            np.linalg.norm(spots[:, None] - expansions[None], axis=2).min(axis=1) > 16.0
        ]
        for corner in sorted(
            corners,
            key=lambda c: cy_distance_to_squared(c, enemy_third),
            reverse=True,
        ):
            closest_first = np.linalg.norm(spots - corner, axis=1).argsort()
            # corners can hold pockets of ground that a probe can't walk to
            for x, y in spots[closest_first[::20][:15]]:
                spot: Point2 = Point2((float(x), float(y)))
                if self.ai.mediator.find_raw_path(
                    start=own_nat, target=spot, grid=ground_grid, sensitivity=1
                ):
                    return spot
        return enemy_third

    def _find_drop_spots(self) -> tuple[Point2, Point2]:
        """Ground just outside the enemy main, then where to drop into the main.

        Returns the spot outside as far from their bases and ramp as possible, so
        warp ins go unseen, and the closest point to it well inside the main.
        """
        enemy_start: Point2 = self.ai.enemy_start_locations[0]
        ground_grid: np.ndarray = self.ai.mediator.get_ground_grid
        main: np.ndarray = np.array(list(self._enemy_main_points))
        # if the prism dies these units have to be able to walk in, and rocks or mineral
        # walls can seal off ground that looks open
        walkable: np.ndarray = label(ground_grid < np.inf)[0]
        outside: np.ndarray = np.argwhere(
            walkable == walkable[self.ai.mediator.get_enemy_ramp.bottom_center.rounded]
        )
        outside = outside[np.linalg.norm(outside - enemy_start, axis=1) < 55.0]
        outside = outside[
            [(x, y) not in self._enemy_main_points for x, y in outside.tolist()]
        ]
        to_main: np.ndarray = np.linalg.norm(
            outside[:, None] - main[::3][None], axis=2
        ).min(axis=1)
        avoid: np.ndarray = np.array(
            [
                *[e for e in self.ai.expansion_locations_list if e != enemy_start],
                self.ai.mediator.get_enemy_ramp.bottom_center,
            ]
        )
        cover: np.ndarray = np.linalg.norm(outside[:, None] - avoid[None], axis=2).min(
            axis=1
        )
        roomy: np.ndarray = (
            np.array([self._warp_room(Point2(p)) for p in outside]) >= MIN_WARP_ROOM
        )
        # `_can_warp_at` refuses anywhere near a base, and a natural can sit right
        # under the cliff, so walking further to the main beats never warping in
        clear: np.ndarray = roomy & (cover > WARP_CLEAR_DISTANCE + 2.0)
        spots: np.ndarray = next(
            spots
            for spots in (
                *[clear & (to_main < reach) for reach in DROP_SPOT_REACHES],
                roomy & (to_main < DROP_SPOT_REACHES[0]),
                to_main < DROP_SPOT_REACHES[0],
            )
            if spots.any()
        )
        warp_spot: Point2 = Point2(
            tuple(float(v) for v in outside[spots][cover[spots].argmax()])
        )

        closest_edge: np.ndarray = main[
            np.linalg.norm(main - np.array(warp_spot), axis=1).argmin()
        ]
        inside: Point2 = Point2(
            cy_towards(Point2(closest_edge.tolist()), enemy_start, 7.0)
        )
        hover: np.ndarray = main[
            np.linalg.norm(main - np.array(inside), axis=1).argmin()
        ]
        return warp_spot, Point2(tuple(float(v) for v in hover))

    def _closest_map_edge(self, position: Point2) -> Point2:
        area = self.ai.game_info.playable_area
        return min(
            [
                Point2((area.x + 2.0, position.y)),
                Point2((area.x + area.width - 2.0, position.y)),
                Point2((position.x, area.y + 2.0)),
                Point2((position.x, area.y + area.height - 2.0)),
            ],
            key=lambda p: cy_distance_to_squared(p, position),
        )

    async def _ensure_tech(self) -> None:
        """Ares can run out of powered spots on some maps and stall the build order.

        Its fallback spot can be unpowered too, which holds the probe there until a
        two minute timeout, so anything stuck like that is placed with a game query.
        """
        pylons: list[Unit] = [
            p
            for p in self.ai.mediator.get_own_structures_dict[UnitTypeId.PYLON]
            if p.is_ready
        ]
        if not pylons:
            return
        own_structures: dict = self.ai.mediator.get_own_structures_dict
        tracker: dict = self.ai.mediator.get_building_tracker_dict
        for structure, wanted in (
            (UnitTypeId.TWILIGHTCOUNCIL, 1),
            (UnitTypeId.ROBOTICSFACILITY, 1),
            (UnitTypeId.GATEWAY, FIRST_WAVE_SIZE),
        ):
            if not self.ai.can_afford(structure):
                continue
            tracked: list[int] = [t for t in tracker if tracker[t][ID] == structure]
            stuck: list[int] = [
                t for t in tracked if not self._is_powered(tracker[t][TARGET])
            ]
            owned: int = len(own_structures[structure])
            if structure == UnitTypeId.GATEWAY:
                owned += len(own_structures[UnitTypeId.WARPGATE])
            if owned + len(tracked) >= wanted and not stuck:
                continue
            # 3x3 structures sit on half coordinates, Ares rejects anything else
            if build_location := await self.ai.find_placement(
                structure,
                near=cy_closest_to(self._macro_location, pylons).position.offset(
                    (0.5, 0.5)
                ),
            ):
                if stuck:
                    tracker[stuck[0]][TARGET] = build_location
                elif worker := self.ai.mediator.select_worker(
                    target_position=build_location
                ):
                    self.ai.mediator.build_with_specific_worker(
                        worker=worker, structure_type=structure, pos=build_location
                    )
            return

    def _is_powered(self, position: Point2) -> bool:
        return cy_pylon_matrix_covers(
            position,
            self.ai.mediator.get_own_structures_dict[UnitTypeId.PYLON],
            self.ai.game_info.terrain_height.data_numpy,
            pylon_build_progress=1.0,
        )

    def _check_tempest_transition(self) -> None:
        if self._transitioned or not self.ai.build_order_runner.build_completed:
            return
        self._detection_seen = self._detection_seen or bool(self.ai.enemy_detectors)
        if self.ai.time > TRANSITION_TIME or (
            self._detection_seen and self.ai.time > TRANSITION_IF_DETECTED_TIME
        ):
            self._transitioned = True

    async def _handle_proxy_construction(self) -> None:
        next_item_to_build: UnitTypeId | None = self._next_build_target(
            PROXY_DARK_SHRINE_PLAN, self._proxy_location
        )
        if self._proxy_finished or not next_item_to_build:
            self._proxy_finished = True
            self._scout_for_bases()
            return

        proxy_probes = self._handle_proxy_probe_assignment(1, self._proxy_location)
        if not proxy_probes:
            return

        building_worker: Unit = cy_closest_to(self._proxy_location, proxy_probes)
        build_location: Point2 | None = None
        if (
            self.ai.can_afford(next_item_to_build)
            and self.ai.tech_requirement_progress(next_item_to_build) >= 0.99
            and cy_distance_to_squared(building_worker.position, self._proxy_location)
            < 144.0
            and (
                next_item_to_build == UnitTypeId.PYLON
                or self._is_powered(self._proxy_location)
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

    def _scout_for_bases(self) -> None:
        """Walk the map so `_enemy_bases` knows about more than the main and natural.

        Runs on the same probe that placed the proxy, one lost here isn't replaced.
        """
        probes: Units = self.ai.mediator.get_units_from_role(role=UnitRole.PROXY_WORKER)
        if not probes:
            return
        expansions: list[Point2] = [
            e
            for e in self.ai.expansion_locations_list
            if e != self.ai.enemy_start_locations[0]
        ]
        self._scouted |= {e for e in expansions if self.ai.is_visible(e)}
        unscouted: list[Point2] = [e for e in expansions if e not in self._scouted]
        if not unscouted:
            # the enemy keeps expanding, so go round again
            self._scouted = set()
            return

        avoid_grid: np.ndarray = self.ai.mediator.get_ground_avoidance_grid
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        for probe in probes:
            target: Point2 = min(
                unscouted, key=lambda e: cy_distance_to_squared(e, probe.position)
            )
            # not worth dying over, the rest of the map is more useful
            if not self.ai.mediator.is_position_safe(
                grid=grid, position=probe.position
            ):
                self._scouted.add(target)
            maneuver: CombatManeuver = CombatManeuver()
            maneuver.add(KeepUnitSafe(unit=probe, grid=avoid_grid))
            maneuver.add(KeepUnitSafe(unit=probe, grid=grid))
            maneuver.add(PathUnitToTarget(unit=probe, target=target, grid=grid))
            self.ai.register_behavior(maneuver)

    def _handle_warp_prisms(
        self, units_to_drop: list[Unit], drop_point: Point2
    ) -> None:
        """Warp in as soon as it's safe, lift it all in, go back for the next wave."""
        own_army: dict = self.ai.mediator.get_own_army_dict
        air_grid: np.ndarray = self.ai.mediator.get_air_grid
        avoid_grid: np.ndarray = self.ai.mediator.get_air_avoidance_grid
        warp_ready: bool = self._warp_at_prism
        if not warp_ready or own_army[UnitTypeId.WARPPRISMPHASING]:
            self._prism_deadline = self.ai.time + PRISM_WAIT_SECONDS

        # not in the build order so a stalled build or a dead prism can't end the plan
        if not (
            own_army[UnitTypeId.WARPPRISM]
            or own_army[UnitTypeId.WARPPRISMPHASING]
            or cy_unit_pending(self.ai, UnitTypeId.WARPPRISM)
            or (self._transitioned and self.ai.minerals <= ZEALOT_BANK_MINERALS)
        ) and self.ai.can_afford(UnitTypeId.WARPPRISM):
            for robo in self.ai.mediator.get_own_structures_dict[
                UnitTypeId.ROBOTICSFACILITY
            ]:
                if robo.is_ready and robo.is_idle and robo.is_powered:
                    robo.train(UnitTypeId.WARPPRISM)
                    break

        for prism in own_army[UnitTypeId.WARPPRISMPHASING]:
            if not self.ai.mediator.is_position_safe(
                grid=air_grid, position=prism.position
            ) or (
                not warp_ready
                and not self._warping_in
                and self.ai.time > self._phased_at + PRISM_PHASE_SECONDS
            ):
                prism(AbilityId.MORPH_WARPPRISMTRANSPORTMODE)

        for prism in own_army[UnitTypeId.WARPPRISM]:
            safe: bool = self.ai.mediator.is_position_safe(
                grid=air_grid, position=prism.position
            )
            maneuver: CombatManeuver = CombatManeuver()
            maneuver.add(KeepUnitSafe(unit=prism, grid=avoid_grid))
            # the hover spot is the only ground in the main we know to be open, a
            # base after that can have its townhall sat right on the drop point
            at_drop_point: bool = cy_distance_to_squared(prism.position, drop_point) < (
                100.0 if self._initial_attack_over else 4.0
            )
            # warping units can't be loaded yet, zealots and templar both take up 2,
            # and a load that has set off is delivered or it just rides back and forth
            if (
                (loadable := [u for u in units_to_drop if u.is_ready])
                and prism.cargo_left >= 2
                and (
                    not prism.has_cargo
                    or cy_distance_to_squared(prism.position, self._pickup_spot) < 225.0
                )
            ):
                maneuver.add(
                    PickUpCargo(unit=prism, grid=air_grid, pickup_targets=loadable)
                )
            elif prism.has_cargo:
                # better to unload early than lose everything, but not next to the
                # pickup or the prism would only lift the same units straight back up
                if at_drop_point or (
                    not safe
                    and cy_distance_to_squared(prism.position, self._pickup_spot) > 81.0
                ):
                    maneuver.add(DropCargo(unit=prism, target=prism.position))
                maneuver.add(
                    PathUnitToTarget(
                        unit=prism,
                        target=drop_point,
                        grid=air_grid,
                        success_at_distance=1.0,
                    )
                )
            elif warp_ready and safe and self._can_warp_at(prism.position):
                prism(AbilityId.MORPH_WARPPRISMPHASINGMODE)
                self._phased_at = self.ai.time
                self._pickup_spot = prism.position
                continue
            else:
                maneuver.add(KeepUnitSafe(unit=prism, grid=air_grid))
                wait_at: Point2 = (
                    self._prism_stage if self._first_wave_sent else self._prism_hold
                )
                maneuver.add(
                    PathUnitToTarget(
                        unit=prism,
                        target=self._warp_spot if warp_ready else wait_at,
                        grid=air_grid,
                        success_at_distance=1.0,
                    )
                )
            self.ai.register_behavior(maneuver)

    def _spawn_army(
        self,
        spawn_target: Point2,
        composition: dict[UnitTypeId, dict[str, float]] = ARMY_COMPOSITION,
    ) -> SpawnController:
        return SpawnController(
            army_composition_dict=composition,
            freeflow_mode=True,
            spawn_target=spawn_target,
        )

    def _macro(self) -> None:
        self._chrono_boosts({UnitTypeId.CYBERNETICSCORE, UnitTypeId.ROBOTICSFACILITY})
        macro_plan: MacroPlan = MacroPlan()
        if spawn_target := self._spawn_target:
            macro_plan.add(self._spawn_army(spawn_target))
        macro_plan.add(AutoSupply(self._macro_location))
        macro_plan.add(BuildWorkers(min(80, 22 * self.ai.townhalls.amount)))
        # by now this adds gates, or replaces the shrine at home if the proxy failed
        if self._first_wave_sent and self.ai.time > 240.0:
            macro_plan.add(
                ProductionController(
                    army_composition_dict=ARMY_COMPOSITION,
                    base_location=self._macro_location,
                )
            )
            macro_plan.add(GasBuildingController(to_count=100))
        if self._first_wave_sent and self.ai.minerals > 600:
            macro_plan.add(ExpansionController(to_count=100, max_pending=1))
        self.ai.register_behavior(macro_plan)
