use leptos::mount::mount_to_body;

fn main() {
    mount_to_body(App);
}

use leptos::prelude::*;

#[component]
fn App() -> impl IntoView {
    let (count, set_count) = signal(0);
    let double_count = move || count.get() * 2;

    view! {
        <button
            on:click=move |_| {
                *set_count.write() += 1;
            }
            // the class: syntax reactively updates a single class
            // here, we'll set the `red` class when `count` is odd
            class:red=move || count.get() % 2 == 1
        >
            "Click me"
        </button>

        <progress
            max="50"
            // we use it once here
            value=double_count
        />
        <p>
            "Double Count: " // and again here
            {double_count}
        </p>
    }
}
