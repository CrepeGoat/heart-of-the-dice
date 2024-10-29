import taipy.gui.builder as tgb
from taipy.gui import Gui

import calc


# TODO allow for:
# - heterogeneous dice
# - multiplicative factors (e.g., double)
def make_prob(d_count, d_sides, d_bias):
    return calc.add_bias(calc.kdn(d_count, d_sides), d_bias)


def update_chart(state):
    state.d_prob = make_prob(int(state.d_count), int(state.d_sides), int(state.d_bias))


if __name__ == "__main__":
    d_count = 1
    d_sides = 2
    d_bias = 0
    d_prob = make_prob(d_count, d_sides, d_bias)

    # Definition of the page
    with tgb.Page() as page:
        tgb.text("# Getting started with Taipy GUI", mode="md")
        tgb.text("{int(d_count)}d{d_sides} + {int(d_bias)}")
        tgb.number("{d_count}", on_change=update_chart)
        tgb.slider(
            "{d_sides}", lov=[2, 4, 6, 8, 10, 12, 20, 100], on_change=update_chart
        )
        tgb.number("{d_bias}", on_change=update_chart)

        tgb.chart(data="{d_prob}", x="x", y="y", type="bar")

    Gui(page).run(debug=True)
