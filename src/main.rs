use leptos::prelude::*;
use std::num::ParseIntError;

fn main() {
    let (dice_sides, set_dice_sides) = signal(Ok(6));
    let dice_sides_str = move || {
        dice_sides.with(|x| {
            x.as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Err".to_string())
        })
    };

    mount_to_body(move || {
        view! {
            <p>"Number of sides: " {dice_sides_str}</p>
            <HomogeneousDiceInputPanel set_dice_sides=set_dice_sides dice_sides_init=6 />
        }
    });
}

#[component]
fn HomogeneousDiceInputPanel(
    set_dice_sides: WriteSignal<Result<u32, ParseIntError>>,
    dice_sides_init: u32,
) -> impl IntoView {
    let dice_sides_str = RwSignal::new(dice_sides_init.to_string());
    Effect::new(move |_| {
        let x = dice_sides_str.with(|x| x.parse::<u32>());
        set_dice_sides.set(x);
    });
    // let set_dice_sides_str = set_dice_sides.clone();
    // let (dice_count, set_dice_count) = signal(1u32);
    // let (dice_drop_count, set_dice_drop_count) = signal(0i32);
    // let (dice_modifier, set_dice_modifier) = signal(0i32);

    view! {
        <input type="number" bind:value=dice_sides_str min="2" max="100" />
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
