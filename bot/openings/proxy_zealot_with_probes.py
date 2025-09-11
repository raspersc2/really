from ares import AresBot
from cython_extensions import cy_unit_pending
from sc2.ids.unit_typeid import UnitTypeId

from bot.openings.opening_base import OpeningBase
from bot.openings.probe_rush import ProbeRush
from bot.openings.proxy_zealot import ProxyZealot


class ProxyZealotWithProbes(OpeningBase):
    _worker_rush_activated: bool
    probe_rush: OpeningBase
    proxy_zealot: OpeningBase

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._worker_rush_activated = False
        self.probe_rush = ProbeRush()
        await self.probe_rush.on_start(ai)
        self.proxy_zealot = ProxyZealot()
        await self.proxy_zealot.on_start(ai)

    async def on_step(self) -> None:
        if self.ai.supply_used < 1:
            await self.ai.client.leave()
        if not self._worker_rush_activated:
            zealots = self.ai.mediator.get_own_army_dict[UnitTypeId.ZEALOT]
            if (len(zealots) + cy_unit_pending(self.ai, UnitTypeId.ZEALOT)) >= 3:
                self._worker_rush_activated = True

        await self.proxy_zealot.on_step()
        if self._worker_rush_activated:
            await self.probe_rush.on_step()
