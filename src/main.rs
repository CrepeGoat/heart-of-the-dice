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
    let dice_count = RwSignal::new(Ok(1));
    let dice_count_str = move || {
        dice_count.with(|x| {
            x.as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Err".to_string())
        })
    };
    let adv_dice_count = RwSignal::new(Ok(0));
    let adv_dice_count_str = move || {
        adv_dice_count.with(|x| {
            x.as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Err".to_string())
        })
    };
    let modifier = RwSignal::new(Ok(0));
    let modifier_str = move || {
        modifier.with(|x| {
            x.as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Err".to_string())
        })
    };

    mount_to_body(move || {
        view! {
            <p>
                {dice_count_str}"d"{dice_sides_str} " adv"{adv_dice_count_str} " + "{modifier_str}
            </p>
            <HomogeneousDiceInputPanel
                dice_sides=dice_sides
                dice_count=dice_count
                adv_dice_count=adv_dice_count
                modifier=modifier
            />
        }
    });
}

#[component]
fn HomogeneousDiceInputPanel(
    dice_sides: RwSignal<Result<i32, ParseIntError>>,
    dice_count: RwSignal<Result<i32, ParseIntError>>,
    adv_dice_count: RwSignal<Result<i32, ParseIntError>>,
    modifier: RwSignal<Result<i32, ParseIntError>>,
) -> impl IntoView {
    view! {
        <DiscreteRangeAndNumberInput
            value=dice_sides
            options=&[2, 4, 6, 8, 10, 12, 20, 100]
            min=2
            max=100
        />
        <NumberInput value=dice_count min=0 max=999 />
        <NumberInput value=adv_dice_count min=-999 max=999 />
        <NumberInput value=modifier min=-999 max=999 />
    }
}

#[component]
fn DiscreteRangeAndNumberInput(
    value: RwSignal<Result<i32, ParseIntError>>,
    options: &'static [i32],
    #[prop(default = i32::MIN)] min: i32,
    #[prop(default = i32::MAX)] max: i32,
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
    value: RwSignal<Result<i32, ParseIntError>>,
    #[prop(default = i32::MIN)] min: i32,
    #[prop(default = i32::MAX)] max: i32,
) -> impl IntoView {
    view! {
        <input
            type="number"
            on:input:target=move |ev| {
                value.set(ev.target().value().parse::<i32>());
            }
            prop:value=move || value.with(|x| x.as_ref().unwrap_or(&min).to_string())
            min=min
            max=max
        />
    }
}
