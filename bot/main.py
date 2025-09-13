import importlib
from typing import Any, Optional

from ares import AresBot
from ares.behaviors.macro import Mining
from sc2.unit import Unit


def _to_snake(name: str) -> str:
    # Convert e.g. "OneBaseTempest" -> "one_base_tempest"
    out = []
    for i, c in enumerate(name):
        if c.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(c.lower())
    return "".join(out)


class MyBot(AresBot):
    def __init__(self, game_step_override: Optional[int] = None):
        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)
        self.opening_handler: Optional[Any] = None
        self.opening_chat_tag: bool = False

    def load_opening(self, opening_name: str) -> None:
        """Load opening from bot.openings.<snake_case> with class <PascalCase>"""
        module_path = f"bot.openings.{_to_snake(opening_name)}"
        module = importlib.import_module(module_path)
        opening_cls = getattr(module, opening_name, None)
        if opening_cls is None:
            raise ImportError(
                f"Opening class '{opening_name}' not found in '{module_path}'"
            )
        self.opening_handler = opening_cls()

    async def on_start(self) -> None:
        await super(MyBot, self).on_start()
        # Ares has initialized BuildOrderRunner at this point
        try:
            self.load_opening(self.build_order_runner.chosen_opening)
            if hasattr(self.opening_handler, "on_start"):
                await self.opening_handler.on_start(self)
        except Exception as exc:
            print(f"Failed to load opening: {exc}")

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)
        self.register_behavior(Mining())

        if self.opening_handler and hasattr(self.opening_handler, "on_step"):
            await self.opening_handler.on_step()

        if not self.opening_chat_tag and self.time > 5.0:
            await self.chat_send(f"Tag:  {self.build_order_runner.chosen_opening}")
            self.opening_chat_tag = True

    async def on_unit_created(self, unit: Unit) -> None:
        await super(MyBot, self).on_unit_created(unit)
        if self.opening_handler and hasattr(self.opening_handler, "on_unit_created"):
            self.opening_handler.on_unit_created(unit)

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        await super(MyBot, self).on_unit_destroyed(unit_tag)
        if self.opening_handler and hasattr(self.opening_handler, "on_unit_destroyed"):
            self.opening_handler.on_unit_destroyed(unit_tag)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            self.mediator.cancel_structure(structure=unit)
            if self.opening_handler and hasattr(
                self.opening_handler, "on_unit_cancelled"
            ):
                self.opening_handler.on_unit_cancelled(unit)

    async def on_building_construction_complete(self, unit: Unit) -> None:
        await super(MyBot, self).on_building_construction_complete(unit)
        if self.opening_handler and hasattr(
            self.opening_handler, "on_building_construction_complete"
        ):
            self.opening_handler.on_building_construction_complete(unit)
