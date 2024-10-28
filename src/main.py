
import taipy.gui.builder as tgb
from taipy.gui import Gui


# TODO allow for:
# - multiple dice
# - biases
# - heterogeneous dice
# - multiplicative factors (e.g., double)
# TODO move to separate file
def make_prob(d_sides):
    return [1/d_sides for _ in range(d_sides)]

def on_slider_action(state):
    state.d_prob = make_prob(int(state.d_sides))


if __name__ == "__main__":
    d_sides = 2
    d_prob = make_prob(d_sides)

    # Definition of the page
    with tgb.Page() as page:
        tgb.text("# Getting started with Taipy GUI", mode="md")
        tgb.text("1d{d_sides}")
        tgb.slider("{d_sides}",  lov=[2, 4, 6, 8, 10, 12, 20, 100], on_change=on_slider_action)

        tgb.chart(data="{d_prob}", type="bar")

    Gui(page).run(debug=True)
