use leptos::prelude::*;

fn main() {
    let (dice_sides, set_dice_sides) = signal(6u32);
    mount_to_body(HomogeneousDiceInputPanel {
        dice_sides: set_dice_sides,
    });
}

#[component]
fn HomogeneousDiceInputPanel(dice_sides: WriteSignal<u32>) -> impl IntoView {
    // let dice_sides_str = move || dice_sides.get().to_string();
    // let set_dice_sides_str = set_dice_sides.clone();
    // let (dice_count, set_dice_count) = signal(1u32);
    // let (dice_drop_count, set_dice_drop_count) = signal(0i32);
    // let (dice_modifier, set_dice_modifier) = signal(0i32);

    view! {
        <input type="number" value="4" min="2" max="100" list="commonDice" />
        <datalist id="commonDice">
            <option value="2"></option>
            <option value="4"></option>
            <option value="6"></option>
            <option value="8"></option>
            <option value="10"></option>
            <option value="12"></option>
            <option value="20"></option>
            <option value="100"></option>
        </datalist>
    }
}
