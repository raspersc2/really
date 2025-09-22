from ares import AresBot
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
        if not self._worker_rush_activated:
            if (
                len(
                    [
                        g
                        for g in self.ai.mediator.get_own_structures_dict[
                            UnitTypeId.GATEWAY
                        ]
                        if g.build_progress > 0.75
                    ]
                )
                > 0
            ):
                self._worker_rush_activated = True

        await self.proxy_zealot.on_step()
        if self._worker_rush_activated:
            await self.probe_rush.on_step()

    def on_unit_destroyed(self, unit_tag):
        self.proxy_zealot.on_unit_destroyed(unit_tag)
