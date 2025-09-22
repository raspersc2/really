from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.unit_typeid import UnitTypeId as UnitID

ATTACK_TARGET_IGNORE: set[UnitID] = {
    UnitID.SCV,
    UnitID.DRONE,
    UnitID.PROBE,
    UnitID.MULE,
    UnitID.LARVA,
    UnitID.EGG,
    UnitID.CHANGELING,
    UnitID.CHANGELINGMARINE,
    UnitID.CHANGELINGMARINESHIELD,
    UnitID.CHANGELINGZEALOT,
    UnitID.CHANGELINGZERGLING,
    UnitID.CHANGELINGZERGLINGWINGS,
    UnitID.REAPER,
}

COMMON_UNIT_IGNORE_TYPES: set[UnitID] = {UnitID.EGG, UnitID.LARVA}

PROXY_4G_PLAN: list[tuple[UnitTypeId, int]] = [
    (UnitTypeId.PYLON, 1),
    (UnitTypeId.GATEWAY, 1),
]

PROXY_VOID_PLAN: list[tuple[UnitTypeId, int]] = [
    (UnitTypeId.PYLON, 2),
    (UnitTypeId.STARGATE, 1),
    (UnitTypeId.PYLON, 2),
    (UnitTypeId.SHIELDBATTERY, 9),
]

PROXY_ZEALOT_PLAN: list[tuple[UnitTypeId, int]] = [
    (UnitTypeId.PYLON, 2),
    (UnitTypeId.GATEWAY, 1),
    (UnitTypeId.PYLON, 1),
    (UnitTypeId.GATEWAY, 2),
    (UnitTypeId.PYLON, 1),
    (UnitTypeId.GATEWAY, 1),
]

PROXY_ZEALOT_PLAN_3G: list[tuple[UnitTypeId, int]] = [
    (UnitTypeId.PYLON, 1),
    (UnitTypeId.GATEWAY, 3),
    (UnitTypeId.PYLON, 1),
]
