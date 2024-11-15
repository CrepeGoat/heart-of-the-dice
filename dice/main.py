from dataclasses import dataclass

import taipy.gui.builder as tgb
from taipy.gui import Gui, navigate

from dice import calc


@dataclass(kw_only=True, slots=True)
class DiceStruct(object):
    _count_keep: int
    _sides: int
    _count_drop: int

    def default():
        return DiceStruct(_count_keep=1, _sides=6, _count_drop=0)

    # UI stores numbers as floats -> auto-convert them to ints
    @property
    def count_keep(self):
        return self._count_keep

    @count_keep.setter
    def count_keep(self, value):
        self._count_keep = int(value)

    @property
    def sides(self):
        return self._sides

    @sides.setter
    def sides(self, value):
        self._sides = int(value)

    @property
    def count_drop(self):
        return self._count_drop

    @count_drop.setter
    def count_drop(self, value):
        self._count_drop = int(value)


def make_dice_label(count_keep, count_drop, sides, bias):
    result = f"{die.count_keep + abs(die.count_drop)}d{die.sides}"

    if die.count_drop < 0:
        result += f" keep highest {die.count_keep}"
    elif die.count_drop > 0:
        result += f" keep lowest {die.count_keep}"

    bias = int(bias)
    if count_drop != 0 and bias != 0:
        result = f"({result})"
    if bias < 0:
        result += f" - {abs(bias)}"
    elif bias > 0:
        result += f" + {bias}"

    return result


# TODO allow for:
# - heterogeneous dice
# - multiplicative factors (e.g., double)
def calc_prob(keep, d_sides, d_bias, drop):
    dist1 = calc.roll_1dn(d_sides)
    if drop > 0:
        dist = calc.roll_k_drophigh(dist1, keep + drop, drop)
    elif drop < 0:
        dist = calc.roll_k_droplow(dist1, keep - drop, -drop)
    else:
        dist = calc.roll_k(dist1, keep)

    return dist.bias_by(d_bias).scaled_to_prob().to_labeled()


def update_chart(state):
    state.d_prob = calc_prob(
        state.die.count_keep, state.die.sides, int(state.d_bias), state.die.count_drop
    )


def set_params(state, count_keep, count_drop, sides, bias: int):
    state.die.count_keep = count_keep
    state.die.count_drop = count_drop
    state.die.sides = sides
    state.d_bias = bias
    state.d_prob = calc_prob(count_keep, sides, d_bias, count_drop)

    state.refresh("die")
    state.refresh("d_bias")


def buy_me_a_coffee_button():
    with open("assets/bmc-button.png", mode="rb") as f:
        image_content = f.read()
    tgb.image(
        image_content,
        width="164px",
        height="46px",
        label="buy me a coffee!",
        on_action=lambda state, id, payload: navigate(
            state, to="http://buymeacoffee.com/awqtopus"
        ),
    )


die = DiceStruct.default()
d_bias = 0
d_prob = calc_prob(die.count_keep, die.sides, d_bias, die.count_drop)

chart_properties = dict(
    layout=dict(
        xaxis=dict(title="dice outcome"),
        yaxis=dict(title="probability (per whole)"),
    )
)

# Definition of the page
with tgb.Page() as page:
    tgb.text("# D&D Dice Calculator", mode="md")

    tgb.text("### common configurations", mode="md")
    with tgb.layout(columns="150px 150px 150px"):
        tgb.button(
            label="roll w/ advantage",
            on_action=lambda state: set_params(
                state, count_keep=1, count_drop=-1, sides=20, bias=0
            ),
        )
        tgb.button(
            label="roll w/ disadvantage",
            on_action=lambda state: set_params(
                state, count_keep=1, count_drop=1, sides=20, bias=0
            ),
        )
        tgb.button(
            label="roll a stat",
            on_action=lambda state: set_params(
                state, count_keep=3, count_drop=-1, sides=6, bias=0
            ),
        )

    tgb.text("### full configuration", mode="md")
    with tgb.layout(columns="100px 300px 100px 100px"):
        with tgb.part():
            tgb.text("# of dice")
            tgb.number("{die.count_keep}", min=0, on_change=update_chart)
        with tgb.part():
            tgb.text("# of sides")
            tgb.slider(
                "{die.sides}", lov=[2, 4, 6, 8, 10, 12, 20, 100], on_change=update_chart
            )
        with tgb.part():
            tgb.text("drop die")
            tgb.number("{die.count_drop}", on_change=update_chart)
        with tgb.part():
            tgb.text("then add")
            tgb.number("{d_bias}", on_change=update_chart)

    tgb.text(
        "## -> {make_dice_label(die.count_keep, die.count_drop, die.sides, d_bias)}",
        mode="md",
    )
    tgb.chart(
        data="{d_prob}", x="x", y="y", type="bar", properties="{chart_properties}"
    )

    buy_me_a_coffee_button()


app_gui = Gui(page)

if __name__ == "__main__":
    app_gui.run(debug=True)
