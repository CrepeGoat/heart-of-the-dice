use leptos::prelude::*;
use leptos_chartistry::*;
use std::num::ParseIntError;

mod distributions;

fn main() {
    let dice_sides = RwSignal::new(Ok(6));
    let dice_count = RwSignal::new(Ok(1));
    let adv_dice_count = RwSignal::new(Ok(0));
    let modifier = RwSignal::new(Ok(0));

    mount_to_body(move || {
        view! {
            <HomogeneousDiceInputPanel
                dice_sides=dice_sides
                dice_count=dice_count
                adv_dice_count=adv_dice_count
                modifier=modifier
            />
            <ProbabilityChart />
        }
    });
}

#[component]
fn ProbabilityChart() -> impl IntoView {
    #[derive(PartialEq, Clone)]
    struct TestDatum {
        x: f64,
        y: f64,
    }
    let test_data = RwSignal::new(vec![
        TestDatum { x: 0., y: 0.25 },
        TestDatum { x: 1., y: 0.5 },
        TestDatum { x: 2., y: 0.25 },
        TestDatum { x: 3., y: 1. },
    ]);
    let line_data = Memo::new(move |_| {
        let mut data = test_data.get();
        let mut result = vec![];

        let p1 = data[0].clone();
        let pn = data[data.len() - 1].clone();

        result.push(TestDatum {
            x: p1.x - 0.51,
            y: 0.,
        });
        result.push(TestDatum {
            x: p1.x - 0.5,
            y: p1.y,
        });
        result.append(&mut data);
        result.push(TestDatum {
            x: pn.x + 0.5,
            y: pn.y,
        });
        result.push(TestDatum {
            x: pn.x + 0.51,
            y: 0.,
        });

        result
    });
    let series = Series::new(|data: &TestDatum| data.x).line(
        Line::new(|data: &TestDatum| data.y)
            .with_interpolation(Interpolation::Step(Step::HorizontalMiddle)),
    );
    view! {
        <Chart
            aspect_ratio=AspectRatio::from_outer_height(300.0, 1.2)
            series=series
            data=line_data
            // debug=true

            left=TickLabels::aligned_floats()
            inner=[
                AxisMarker::left_edge().into_inner(),
                AxisMarker::bottom_edge().into_inner(),
                YGridLine::default().into_inner(),
                YGuideLine::over_mouse().into_inner(),
            ]
        />
    }
}

#[component]
fn HomogeneousDiceInputPanel(
    dice_sides: RwSignal<Result<i32, ParseIntError>>,
    dice_count: RwSignal<Result<i32, ParseIntError>>,
    adv_dice_count: RwSignal<Result<i32, ParseIntError>>,
    modifier: RwSignal<Result<i32, ParseIntError>>,
) -> impl IntoView {
    let dice_sides_str = move || {
        dice_sides.with(|x| {
            x.as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Err".to_string())
        })
    };
    let dice_count_str = move || {
        dice_count.with(|x| {
            x.as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Err".to_string())
        })
    };
    let adv_dice_count_str = move || {
        adv_dice_count.with(|x| {
            x.as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Err".to_string())
        })
    };
    let modifier_str = move || {
        modifier.with(|x| {
            x.as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Err".to_string())
        })
    };

    let label_str = move || {
        format!(
            "{dcount}d{dsides} adv{adv_dcount} + {modifier}",
            dcount = dice_count_str(),
            dsides = dice_sides_str(),
            adv_dcount = adv_dice_count_str(),
            modifier = modifier_str(),
        )
    };

    view! {
        <p>{label_str}</p>
        <div>
            <DiscreteRangeAndNumberInput
                value=dice_sides
                options=&[2, 4, 6, 8, 10, 12, 20, 100]
                min=2
                max=100
            />
            <NumberInput value=dice_count min=0 max=999 />
            <NumberInput value=adv_dice_count min=-999 max=999 />
            <NumberInput value=modifier min=-999 max=999 />
        </div>
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
