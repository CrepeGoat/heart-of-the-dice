from dataclasses import dataclass
from typing import Literal

import taipy.gui.builder as tgb
from taipy.gui import Gui

import calc


@dataclass(kw_only=True, slots=True)
class DiceStruct(object):
    _count: int
    _sides: int
    drop: Literal["low", "none", "high"]

    def default():
        return DiceStruct(_count=1, _sides=6, drop="none")

    # UI stores numbers as floats -> auto-convert them to ints
    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = int(value)

    @property
    def sides(self):
        return self._sides

    @sides.setter
    def sides(self, value):
        self._sides = int(value)


def make_dice_controls(tgb):
    tgb.text("{int(die.count)}d{die.sides} drop {die.drop}")
    with tgb.layout(columns="1 1 1"):
        tgb.number("{die.count}", on_change=update_chart)
        tgb.slider(
            "{die.sides}", lov=[2, 4, 6, 8, 10, 12, 20, 100], on_change=update_chart
        )
        tgb.toggle("{die.drop}", lov=["low", "none", "high"], on_change=update_chart)


# TODO allow for:
# - heterogeneous dice
# - multiplicative factors (e.g., double)
def calc_prob(d_count, d_sides, d_bias, drop):
    if drop == "high":
        dist = calc.kdn_drophigh(d_count, d_sides)
    elif drop == "low":
        dist = calc.kdn_droplow(d_count, d_sides)
    else:
        dist = calc.kdn(d_count, d_sides)

    return dist.bias_by(d_bias).to_labeled()


def update_chart(state):
    state.d_prob = calc_prob(
        state.die.count, state.die.sides, int(state.d_bias), state.die.drop
    )


if __name__ == "__main__":
    die = DiceStruct.default()
    d_bias = 0
    d_prob = calc_prob(die.count, die.sides, d_bias, die.drop)

    # Definition of the page
    with tgb.Page() as page:
        tgb.text("# D&D Dice Calculator", mode="md")
        make_dice_controls(tgb)

        tgb.text("+ {int(d_bias)}")
        tgb.number("{d_bias}", on_change=update_chart)

        tgb.chart(data="{d_prob}", x="x", y="y", type="bar")

    Gui(page).run(debug=True)
