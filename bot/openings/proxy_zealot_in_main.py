from ares import AresBot

from bot.openings.opening_base import OpeningBase
from bot.openings.proxy_zealot import ProxyZealot


class ProxyZealotInMain(OpeningBase):
    proxy_zealot: OpeningBase

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self.proxy_zealot = ProxyZealot()
        await self.proxy_zealot.on_start(ai)

    async def on_step(self) -> None:
        await self.proxy_zealot.on_step()

    def on_unit_destroyed(self, unit_tag):
        self.proxy_zealot.on_unit_destroyed(unit_tag)
