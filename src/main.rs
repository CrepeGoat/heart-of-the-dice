use leptos::mount::mount_to_body;

fn main() {
    mount_to_body(App);
}

use leptos::prelude::*;

#[component]
fn ProgressBar(
    #[prop(optional)] progress: Option<Box<dyn Fn() -> i32 + Send + Sync>>,
) -> impl IntoView {
    progress.map(|progress| {
        view! {
            <progress max=100 value=progress />
            <br />
        }
    })
}

#[component]
pub fn App() -> impl IntoView {
    view! { <ProgressBar /> }
}
