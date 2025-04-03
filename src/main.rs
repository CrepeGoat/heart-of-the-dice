use leptos::prelude::*;
use leptos_chartistry::*;

fn main() {
    mount_to_body(move || {
        view! {
            <ProbabilityChart/>
        }
    });
}

#[component]
fn ProbabilityChart() -> impl IntoView {
    struct TestDatum {
        x: f64,
        y: f64,
    }
    let test_data = RwSignal::new(vec![
        TestDatum { x: 0., y: 0.25 },
        TestDatum { x: 1., y: 0.5 },
        TestDatum { x: 2., y: 0.25 },
    ]);
    let series = Series::new(|data: &TestDatum| data.x).bar(|data: &TestDatum| data.y);
    view! {
        <Chart
            aspect_ratio=AspectRatio::from_outer_height(300.0, 1.2)
            series=series
            data=test_data
            debug=true
        />
    }
}
