use leptos::prelude::*;
use std::num::ParseIntError;

fn main() {
    let dice_sides = RwSignal::new(Ok(6));
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
            <HomogeneousDiceInputPanel dice_sides=dice_sides />
        }
    });
}

#[component]
fn HomogeneousDiceInputPanel(dice_sides: RwSignal<Result<u32, ParseIntError>>) -> impl IntoView {
    // let set_dice_sides_str = set_dice_sides.clone();
    // let (dice_count, set_dice_count) = signal(1u32);
    // let (dice_drop_count, set_dice_drop_count) = signal(0i32);
    // let (dice_modifier, set_dice_modifier) = signal(0i32);

    view! {
        <DiscreteRangeAndNumberInput
            value=dice_sides
            options=&[2, 4, 6, 8, 10, 12, 20, 100]
            min=2
            max=100
        />
    }
}

#[component]
fn DiscreteRangeAndNumberInput(
    value: RwSignal<Result<u32, ParseIntError>>,
    options: &'static [u32],
    #[prop(default = u32::MIN)] min: u32,
    #[prop(default = u32::MAX)] max: u32,
) -> impl IntoView {
    let options_len = options.len();
    let range_index = RwSignal::new(0);
    let update_range_index = move || {
        value.with(|v| {
            v.as_ref().map_or(range_index.get(), |v| {
                options
                    .iter()
                    .position(|o| *o == *v)
                    .unwrap_or(range_index.get())
            })
        })
    };

    view! {
        <input
            type="range"
            on:input:target=move |ev| {
                range_index.set(ev.target().value().parse::<usize>().unwrap());
                value.set(Ok(options[range_index.get()]));
            }
            prop:value=update_range_index
            min="0"
            max=options_len - 1
            step="1"
        />
        <NumberInput value=value min=min max=max />
    }
}

#[component]
fn NumberInput(
    value: RwSignal<Result<u32, ParseIntError>>,
    #[prop(default = u32::MIN)] min: u32,
    #[prop(default = u32::MAX)] max: u32,
) -> impl IntoView {
    view! {
        <input
            type="number"
            on:input:target=move |ev| {
                value.set(ev.target().value().parse::<u32>());
            }
            prop:value=move || value.with(|x| x.as_ref().unwrap_or(&min).to_string())
            min=min
            max=max
        />
    }
}
