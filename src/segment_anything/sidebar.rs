use dioxus::prelude::*;
use dioxus::logger::tracing::info;
use image::{ImageBuffer, Rgba, Rgb, DynamicImage};
use super::model::{SegmentAnythingData, ImageState, MaskState, MessageState, sam_resize_image};
use web_sys::wasm_bindgen::JsCast;

#[derive(Debug, Clone, PartialEq, Props)]
pub struct SideBarProps {
    pub image: Signal<ImageState>,
    pub mask: Signal<MaskState>,
    pub message_state: Signal<MessageState>,
    pub toggle_mask: Signal<bool>,
    pub data: SegmentAnythingData
}



#[component]
pub fn SideBar(mut props: SideBarProps) -> Element {



    rsx! {
        div {
            class: "w-64 bg-gray-100 p-4 border-r border-gray-200",
            div {
                class: "sa-container-wrapper",
                SideBarDescription { data: props.data }
                ControlBar { 
                    toggle_mask: props.toggle_mask,
                    data: props.data,
                    message_state: props.message_state,
                    image: props.image,
                    mask: props.mask,
                }
                ImageUploader {
                    image: props.image
                }

            }
        }
    }
}

#[component]
pub fn SideBarDescription(mut data: SegmentAnythingData) -> Element {

    let select_model = move |ev: Event<FormData>| {
        let selection = ev.value();
        let map_selection = match selection.as_str() {
            "sam_mobile_tiny" => Some("mobile_sam-tiny-vitt.safetensors".to_string()),
            "sam_base" => Some("sam_vit_b_01ec64.safetensors".to_string()),
            _ => None
        };

        data.model_name.set(map_selection);
    };

    rsx! {
        div {
            class: "container max-w-4xl gap-20 p-4",
            div {
                class: "grid grid-cols-1 gap-8 relative",
                div {
                    class: "gap-24",
                    label {
                        r#for: "model",
                        class: "font-medium",
                        "Models Options: "
                    }
                    select {
                        id: "model",
                        class: "border-2 border-gray-500 rounded-md font-light",
                        oninput: select_model,
                        option {
                            value: "",
                            selected: true,
                            "Select Model"
                        }
                        option {
                            value: "sam_mobile_tiny",
                            selected: false,
                            "Mobile SAM Tiny (40.6 MB)"
                        }
                        option {
                            value: "sam_base",
                            "SAM Base (375 MB)"
                        }
                        
                    }
                }
                div {
                    p {
                        class: "text-xs italic max-w-lg",
                        b { "Note:" }
                        " The model's first run may take a few seconds as it loads and caches the model in the browser, and then creates the image embeddings. Any subsequent clicks on points will be significantly faster."
                    }
                }
            }
        }
    }
}



#[component]
pub fn ControlBar(
    toggle_mask: Signal<bool>,
    data: SegmentAnythingData, 
    message_state: Signal<MessageState>,
    image: Signal<ImageState>,
    mask: Signal<MaskState>
    ) -> Element {
    // Derive whether buttons should be disabled based on data state
    // let data = data;

    let is_empty = data.clone().goal_points.read().is_empty();

    let cutout_mask_image = move || {
        if let ImageState::Ready(img) = &*image.read() {
            if let MaskState::Ready(msk) = &*mask.read() {
                Some(create_cutout(img.clone(), msk.clone()))
            }
            else {
                None
            }
        }
        else {
            None
        }
    };

    let download_image = move |_| {
        if let Some(cutout) = &cutout_mask_image() {
            let mut png_bytes = Vec::new();
            cutout
                .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
                .expect("Failed to encode PNG");
            let array = js_sys::Uint8Array::from(png_bytes.as_slice());
            let blob = web_sys::Blob::new_with_u8_array_sequence(&js_sys::Array::of1(&array))
                .expect("Failed to create Blob");
            let url = web_sys::Url::create_object_url_with_blob(&blob).expect("Failed to create URL");
            let window = web_sys::window().expect("No window");
            let document = window.document().expect("No document");
            let anchor = document
                .create_element("a")
                .expect("Failed to create anchor")
                .dyn_into::<web_sys::HtmlAnchorElement>()
                .expect("Failed to cast to HtmlAnchorElement");
            anchor.set_href(&url);
            anchor.set_download("cutout.png");
            anchor.click();
            web_sys::Url::revoke_object_url(&url).expect("Failed to revoke URL");
        }
    };

    let get_message_state_class = move |msg| {
        match msg {
            MessageState::Waiting => "waiting",
            MessageState::ModelLoading => "loading",
            MessageState::ModelLoaded => "loading",
            MessageState::EmbeddingsLoading => "loading",
            MessageState::Ready => "success",
        }
    };

    let get_message_state_parse = move |msg| {
        match msg {
            MessageState::Waiting => "Waiting".to_string(),
            MessageState::ModelLoading => "Model loading...".to_string(),
            MessageState::ModelLoaded => "Model loaded...".to_string(),
            MessageState::EmbeddingsLoading => "Creating embeddings...".to_string(),
            MessageState::Ready => "Model ready".to_string(),
        }
    };

    rsx! {
        div {
            class: "control-bar-container",
            // Status indicator
            div {
                class: "status-indicator",
                span {
                    id: "output-status",
                    class: {
                        let msg = (message_state.read()).clone();
                        get_message_state_class(msg)
                    },
                    {
                        let msg = (message_state.read()).clone();
                        get_message_state_parse(msg)
                    }
                }
            }            
            div {
                class: "control-bar",
        
                // Mask toggle button
                button {
                    id: "mask-btn",
                    aria_label: "Toggle mask points",
                    class: format_args!(
                        "control-button {}",
                        if *toggle_mask.read() { "mask-active" } else { "" }
                    ),
                    onclick: move |_| {
                        toggle_mask.set(true);
                    },
                    div {
                        class: "control-button-icon",
                        svg {
                            xmlns: "http://www.w3.org/2000/svg",
                            view_box: "0 0 512 512",
                            path {
                                d: "M256 512a256 256 0 1 0 0-512 256 256 0 1 0 0 512z",
                                fill: if *toggle_mask.read() { "#000" } else { "#666" },
                            }
                        }
                    },
                    span { "Goal Points" }
                },

                // Background toggle button
                button {
                    id: "background-btn",
                    aria_label: "Toggle background points",
                    class: format_args!(
                        "control-button {}",
                        if !*toggle_mask.read() { "background-active" } else { "" }
                    ),
                    onclick: move |_| {
                        toggle_mask.set(false);
                    },
                    div {
                        class: "control-button-icon",
                        svg {
                            xmlns: "http://www.w3.org/2000/svg",
                            view_box: "0 0 512 512",
                            path {
                                d: "M464 256a208 208 0 1 0-416 0 208 208 0 1 0 416 0zM0 256a256 256 0 1 1 512 0 256 256 0 1 1-512 0z",
                                fill: if !*toggle_mask.read() { "#000" } else { "#666" },
                            }
                        }
                    },
                    span { "Avoid Points" }
                }
                // Undo button
                button {
                    id: "undo-btn",
                    disabled: is_empty,
                    class: "control-button",
                    onclick: move |_| {
                        info!("Removing last point");
                        if *toggle_mask.read() {
                            data.remove_goal_point();
                        }
                        else {
                            data.remove_avoid_point();
                        }
                    },
                    div {
                        class: "control-button-icon",
                        svg {
                            xmlns: "http://www.w3.org/2000/svg",
                            class: "w-full h-full fill-current",
                            view_box: "0 0 512 512",
                            path {
                                d: "M48.5 224H40a24 24 0 0 1-24-24V72a24 24 0 0 1 41-17l41.6 41.6a224 224 0 1 1-1 317.8 32 32 0 0 1 45.3-45.3 160 160 0 1 0 1-227.3L185 183a24 24 0 0 1-17 41H48.5z"
                            }
                        }
                    },
                    span { "Undo" }
                }
        
                // Clear button
                button {
                    id: "clear-btn",
                    disabled: is_empty,
                    class: "control-button",
                    onclick: move |_| {
                        info!("Clearing all points");
                        data.clear_all_points();
                    },
                    div {
                        class: "control-button-icon",
                        svg {
                            xmlns: "http://www.w3.org/2000/svg",
                            class: "w-full h-full stroke-current",
                            view_box: "0 0 13 12",
                            fill: "none",
                            path {
                                d: "M1.6.7 12 11.1M12 .7 1.6 11.1",
                                stroke: "currentColor",
                                stroke_width: "2"
                            }
                        }
                    },
                    span { "Clear" }
                }
            }
            div {
                class: "download-section",
                button {
                    class: "download-button",
                    disabled: is_empty,
                    onclick: download_image,
                    "Download Cutout PNG"
                }
            }

            
 
        }        
    }
}

pub fn create_cutout(
    base_image: DynamicImage,
    mask: DynamicImage,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    // Convert base_image to RGBA for consistent processing
    let base_image = match base_image {
        DynamicImage::ImageRgba8(img) => img,
        _ => base_image.to_rgba8(),
    };

    // Ensure mask is RGB for threshold checking (or use alpha if RGBA)
    let mask_rgb = mask.to_rgb8();
    let (width, height) = base_image.dimensions();
    let mut cutout_image = base_image.clone();

    // Iterate over each pixel
    for y in 0..height {
        for x in 0..width {
            let mask_pixel = mask_rgb.get_pixel(x, y);
            // Use red channel as indicator (adjust threshold as needed)
            if mask_pixel[0] == 0 {
                // Set non-masked pixels to fully transparent
                cutout_image.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
            // Otherwise, keep the original base_image pixel
        }
    }

    cutout_image
}


#[component]
pub fn ImageUploader(image: Signal<ImageState>) -> Element {


    let on_file_upload = move |evt: Event<FormData>| {
        if let Some(file_engine) = evt.files() {
            let file_names = file_engine.files();
            if let Some(name) = file_names.first() {
                let name = name.clone();
                spawn(async move {
                    match file_engine.read_file(&name).await {
                        Some(bytes) => {
                            info!("Read file: {name} ({} bytes)", bytes.len());
                            // Handle the file bytes here
                            match image::load_from_memory(&bytes) {
                                Ok(dynamic_image) => {
                                    info!("Successfully converted to DynamicImage");

                                    let dynamic_image = sam_resize_image(dynamic_image.clone());

                                    image.set(ImageState::Ready(dynamic_image));
                                    // Use dynamic_image here (type: DynamicImage)
                                }
                                Err(e) => {
                                    image.set(ImageState::Error(format!("Failed to convert to DynamicImage: {:?}", e)));
                                    info!("Failed to convert to DynamicImage: {:?}", e);
                                }
                            }
                        }
                        None => {
                            image.set(ImageState::Error(format!("Failed to read file {}", name)));
                            info!("Failed to read file {name}");
                        }
                    }
                });
            }
        }
    };

    let on_example_upload = move |url: String| {
        async move {
            let client = reqwest::Client::new();
            let response = client
                .get(url)
                .send()
                .await.unwrap();
            
            let bytes = response.bytes().await.unwrap();
            let img = image::load_from_memory(&bytes).unwrap();
            let img = sam_resize_image(img.clone());
            
            image.set(ImageState::Ready(img));
        }
    };

    rsx! {
        div {class: "image-uploader-sidebar",
        div {
            class: "image-uploader-container",

            div {
                class: "image-uploader-row",
                div {
                    class: "upload-section",
                    // Drag and drop window
                    div {
                        class: "drop-image-container",
                        label {
                            r#for: "drop-image",
                            class: "drop-image-label",
                            div {
                                class: "drop-image-text",
                                span { "Drag & drop" }
                                span { 
                                    class: "drop-image-subtext", 
                                    "or click to upload" 
                                }
                            }
                            input {
                                id: "drop-image",
                                name: "drop-image",
                                r#type: "file",
                                accept: "image/*",
                                class: "sr-only",
                                onchange: on_file_upload
                            }
                        }
                    }
                    // Example images with title above
                    div {
                        class: "example-section",
                        h3 {
                            class: "example-title",
                            "Examples:"
                        }
                        // Example images
                        div {
                            class: "example-images",
                            img {
                                class: "example-image",
                                src: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/candle/examples/sf.jpg",
                                onclick: move |_| {
                                    info!("Selected SF image");
                                    on_example_upload("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/candle/examples/sf.jpg".to_string())
                                }
                            }
                            img {
                                class: "example-image",
                                src: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/candle/examples/bike.jpeg",
                                onclick: move |_| {
                                    info!("Selected bike image");
                                    on_example_upload("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/candle/examples/bike.jpeg".to_string())
                                }
                            }
                            img {
                                class: "example-image",
                                src: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/candle/examples/000000000077.jpg",
                                onclick: move |_| {
                                    info!("Selected example image");
                                    on_example_upload("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/candle/examples/000000000077.jpg".to_string())
                                }
                            }
                        }
                    }
            }
        }
        }
    }}
}