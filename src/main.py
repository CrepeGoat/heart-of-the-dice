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


# TODO allow for:
# - heterogeneous dice
# - multiplicative factors (e.g., double)
def calc_prob(d_count, d_sides, d_bias, drop):
    dist1 = calc.roll_1dn(d_sides)
    if drop == "high":
        dist = calc.roll_k_drophigh(dist1, d_count, 1)
    elif drop == "low":
        dist = calc.roll_k_droplow(dist1, d_count, 1)
    else:
        dist = calc.roll_k(dist1, d_count)

    return dist.bias_by(d_bias).scaled_to_prob().to_labeled()


def update_chart(state):
    state.d_prob = calc_prob(
        state.die.count, state.die.sides, int(state.d_bias), state.die.drop
    )


if __name__ == "__main__":
    die = DiceStruct.default()
    d_bias = 0
    d_prob = calc_prob(die.count, die.sides, d_bias, die.drop)

    chart_properties = dict(
        layout=dict(
            xaxis=dict(title="dice outcome"),
            yaxis=dict(title="probability (per whole)"),
        )
    )

    # Definition of the page
    with tgb.Page() as page:
        tgb.text("# D&D Dice Calculator", mode="md")

        with tgb.layout(columns="100px 300px 200px 100px"):
            tgb.text("# of dice")
            tgb.text("# of sides")
            tgb.text("drop die")
            tgb.text("then add")

            tgb.number("{die.count}", min=0, on_change=update_chart)
            tgb.slider(
                "{die.sides}", lov=[2, 4, 6, 8, 10, 12, 20, 100], on_change=update_chart
            )
            tgb.toggle(
                "{die.drop}", lov=["low", "none", "high"], on_change=update_chart
            )
            tgb.number("{d_bias}", on_change=update_chart)

        tgb.text(
            "## -> {int(die.count)}d{die.sides} drop {die.drop} + {int(d_bias)}",
            mode="md",
        )
        tgb.chart(
            data="{d_prob}", x="x", y="y", type="bar", properties="{chart_properties}"
        )

    Gui(page).run(debug=True)
