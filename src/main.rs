use leptos::prelude::*;

fn main() {
    let (dice_sides, set_dice_sides) = signal(Some(6));
    let dice_sides_str =
        move || dice_sides.with(|x| x.map(|x| x.to_string()).unwrap_or("None".to_string()));

    mount_to_body(move || {
        view! {
            <p>"Number of sides: " {dice_sides_str}</p>
            <HomogeneousDiceInputPanel
                set_dice_sides=set_dice_sides
                dice_sides_init="6".to_string()
            />
        }
    });
}

#[component]
fn HomogeneousDiceInputPanel(
    set_dice_sides: WriteSignal<Option<u32>>,
    dice_sides_init: String,
) -> impl IntoView {
    let dice_sides_str = RwSignal::new(dice_sides_init);
    Effect::new(move |_| {
        let x = dice_sides_str.with(|x| x.parse::<u32>().ok());
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
