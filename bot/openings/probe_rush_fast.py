from ares import AresBot
from bot.openings.opening_base import OpeningBase
from bot.openings.probe_rush import ProbeRush


class ProbeRushFast(OpeningBase):
    _worker_rush_activated: bool
    probe_rush: OpeningBase

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.probe_rush = ProbeRush()
        await self.probe_rush.on_start(ai)

    async def on_step(self) -> None:
        await self.probe_rush.on_step()
