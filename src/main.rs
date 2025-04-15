pub mod segment_anything;

use segment_anything::sidebar::SideBar;
use segment_anything::ui::ImageContainer;
use segment_anything::model::{ImageState, ModelState, MaskState, MessageState};
use segment_anything::model::{SegmentAnythingModel, SegmentAnythingData};
use segment_anything::model::sam_resize_image;
use segment_anything::mask::{create_example_mask, overlay_mask};
use rand::{rng, Rng, random_range};
use base64::{engine::general_purpose, Engine as _};
use std::io::Cursor;


use dioxus::router::prelude::*;
use dioxus::prelude::*;
use dioxus::logger::tracing::info;
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgba};

fn main() {
    
    dioxus::launch(app);
}

fn app() -> Element {
    
    rsx! { Router::<Route> {} }
}

#[derive(Clone, Routable, Debug, PartialEq)]
enum Route {
    #[route("/")]
    SegmentAnythingApp {}
}

/// An error that can occur when loading a [`SegmentAnything`] model.
#[derive(Debug, thiserror::Error)]
pub enum LoadSegmentAnythingError {
    #[error("Failed to load model into device: {0}")]
    LoadModel(#[from] candle_core::Error),
    #[error("Failed to fetch model from URL: {0}")]
    UrlFetchError(#[from] reqwest::Error),
}


#[component]
fn SegmentAnythingApp() -> Element {
    info!("launch");
    // let base_image_resource = use_resource(move || async {
    //     let client = reqwest::Client::new();
    //     let response = client
    //         .get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/candle/examples/sf.jpg")
    //         .send()
    //         .await?;
        
    //     let bytes = response.bytes().await?;
    //     let img = image::load_from_memory(&bytes)?;
    //     Ok::<_, anyhow::Error>(img)
    // });


    let is_model_loaded = use_signal(|| false);
    let should_set_embeddings = use_signal(|| false);

    let model_name = use_signal(|| None as Option<String>);
    // let undo_last_point = use_signal(|| true);
    let toggle_mask = use_signal(|| true);
    // let model_name = use_signal(|| Some(String::from("mobile_sam-tiny-vitt.safetensors")));
    let goal_points = use_signal(|| Vec::new());
    let avoid_points = use_signal(|| Vec::new());

    let base_image = use_signal(|| ImageState::Loading);
    let base_image_bytes = use_signal(|| String::new());
    let mask_image = use_signal(|| MaskState::Waiting);
    let message_state = use_signal(|| MessageState::Waiting);


    let sam_model = use_signal(|| ModelState::Loading);
    let sam_data = SegmentAnythingData{
                                            model_name: model_name,
                                            goal_points: goal_points,
                                            avoid_points: avoid_points 
                                        };


    // // Update base_image, base_image_bytes, and embeddings
    // use_effect(move || {

    //     let image_resource = &*base_image_resource.read();

    //     let mut sam_data = sam_data.clone();
    //     let mut base_image = base_image.clone();
    //     let mut base_image_bytes = base_image_bytes.clone();

    //     let mut mask_image = mask_image.clone();
        
    //     match &*base_image_resource.read() {
    //         Some(Ok(img)) => {
    //                 base_image.set(ImageState::Ready(img.clone()));

    //                 // Set Mask
    //                 let (width, height) = img.clone().dimensions();
    //                 mask_image.set(MaskState::Ready(DynamicImage::new_rgba8(width, height)));

    //                 let mut buf = Vec::new();
    //                 match img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png) {
    //                     Ok(_) => {
    //                         let encoded = general_purpose::STANDARD.encode(&buf);

    //                         base_image_bytes.set(format!("data:image/png;base64,{}", encoded));

                            

    //                     }
    //                     Err(e) => {
    //                         info!("Failed to encode image to PNG: {}", e);
    //                         base_image.set(ImageState::Error(format!("Failed to encode image: {}", e)));
    //                     }
    //                 }
    //         }
    //         Some(Err(e)) => {
    //             info!("Image resource error: {}", e);
    //             base_image.set(ImageState::Error(e.to_string()));
    //         }
    //         None => {
    //             info!("Image resource not ready");
    //             base_image.set(ImageState::Loading);
    //         }
    //     }
    // });

    use_effect(move|| {

        let maybe_model_name = (*sam_data.model_name.read()).clone();

        let mut sam_model = sam_model.clone();
        let mut is_model_loaded = is_model_loaded.clone();
        let mut should_set_embeddings = should_set_embeddings.clone();
        let mut message_state = message_state.clone();

        let should_load = match maybe_model_name {
            None => {
                sam_model.set(ModelState::Waiting);
                is_model_loaded.set(false);
                false
            },
            Some(filename) => {
                if filename.is_empty() {
                    sam_model.set(ModelState::Waiting);
                    is_model_loaded.set(false);
                    false
                } else {
                    info!("sending model load");
                    true
                }
            }
        };
    });

    use_effect(move|| {

        let mut mask_image = mask_image.clone();
        let mut base_image_bytes = base_image_bytes.clone();
        let mut should_set_embeddings = should_set_embeddings.clone();

        match &*base_image.read() {
            ImageState::Ready(img) => {
                // Set Mask
                let (width, height) = img.clone().dimensions();
                mask_image.set(MaskState::Ready(DynamicImage::new_rgba8(width, height)));

                let mut buf = Vec::new();
                match img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png) {
                    Ok(_) => {
                        let encoded = general_purpose::STANDARD.encode(&buf);
                        should_set_embeddings.set(true);
                        base_image_bytes.set(format!("data:image/png;base64,{}", encoded));

                    }
                    Err(e) => {
                        info!("Failed to encode image to PNG: {}", e);
                    }
                }
            },
            _ => {
                info!("Image not ready");
            }
        };
    });


    let load_model_resource = use_resource(move || {

        let mut sam_data = sam_data.clone();
        let mut sam_model = sam_model.clone();
        let mut is_model_loaded = is_model_loaded.clone();
        let mut message_state = message_state.clone();

        let maybe_model_name = sam_data.model_name.read().clone();
        
        async move {
            let base_url = "https://huggingface.co/lmz/candle-sam/resolve/main/";

            match maybe_model_name {
                None => sam_model.set(ModelState::Waiting),
                Some(filename) => {
            
            
            // let filename = model_name
            info!("filename: {}", filename);
            
            let model_url = format!("{}{}", base_url, filename);
            
            message_state.set(MessageState::ModelLoading);
            is_model_loaded.set(false);

            let response_result = reqwest::get(model_url).await;
            match response_result {
                Ok(response) => {
                    match response.bytes().await {
                        Ok(bytes) => {

                        let weights = bytes.to_vec();
                        match SegmentAnythingModel::builder().weights(weights).use_model(filename).build().await {
                            Ok(model) => {
                                is_model_loaded.set(true);
                                message_state.set(MessageState::ModelLoaded);
                                sam_model.set(ModelState::Loaded(model))
                            },
                            Err(e) => {
                                info!("Building model failed: {:?}", e);
                                is_model_loaded.set(false);
                                sam_model.set(ModelState::Error(e));
                            }
                        }
                    }
                     Err(err) => {
                            sam_model.set(ModelState::Error(err.to_string()));
                        }
                    }
                }
                Err(err) => {
                    sam_model.set(ModelState::Error(err.to_string()));
                }
            }
        }
    }
    }});


    // Update settings reactively
    let load_embeddings = use_resource(move || {
        // let mut sam_settings = sam_settings.clone();

        let should_set_embeddings = *should_set_embeddings.read();
        let is_model_loaded = *is_model_loaded.read();
        let mut sam_model = sam_model.clone();
        let mut message_state = message_state.clone();

        async move {
            if is_model_loaded && should_set_embeddings {
            
            // let base_image2 = sam_data.write().image.clone();
            match &*base_image.read() {
                ImageState::Ready(image) => {

                    let image = sam_resize_image(image.clone());
                    let image_clone = image.clone();

                    let height = image_clone.height();
                    let width = image_clone.width();
                    let size = image_clone.as_bytes().len();
                    info!("dheight {height} dwidth {width} dsize {size}");
                    // let new_img = ImageBuffer::from_raw(image.width(), image.height(), image.as_raw().clone())
                    //     .unwrap_or_else(|| ImageBuffer::new(1, 1));
                    info!("[Inference Settings] set");
                    // sam_settings.set(SegmentAnythingInferenceSettings::new(new_img));
                    
                    message_state.set(MessageState::EmbeddingsLoading);
                    // Update embeddings
                    match &mut *sam_model.write() {
                        ModelState::Loaded(model) => {
                            
                            if let Err(e) = model.set_image_embeddings(image_clone).await {
                                info!("Set embeddings failed: {}", e);
                            }
                            info!("[Model State] loaded");
                            message_state.set(MessageState::Ready);
 
                        }
                        ModelState::Error(e) => info!("Model error: {}", e),
                        _ => info!("Model not ready for embeddings"),
                    }
                }
                _ => {
                    // sam_settings.set(SegmentAnythingInferenceSettings::new(
                    //     ImageBuffer::<Rgba<u8>, Vec<u8>>::new(1, 1),
                    // ));
                }
            }
            }
        }

    });

    rsx! {
        document::Link {
            rel: "stylesheet",
            href: asset!("/assets/segment_anything.css")
        }
        document::Link {
            rel: "stylesheet",
            href: asset!("/assets/dist.css")
        }
        document::Link {
            rel: "stylesheet",
            href: asset!("/assets/sidebar.css")
        }

            div {
                class: "centered-container", // Full width up to 8xl
                div {
                    class: "flex flex-col items-end", // Right-aligns content
                    // Candle icon
                    div {
                        class: "flex justify-end", // Right-aligns candle
                        span {
                            class: "text-5xl",
                            "🕯️"
                        }
                    },
                    // Main content
                    div {
                        class: "text-right", // Right-aligns all text
                        h1 {
                            class: "text-5xl font-bold",
                            "Candle Segment Anything"
                        },
                        h2 {
                            class: "text-2xl font-bold mt-2",
                            "Rust/WASM Demo"
                        },
                        p {
                            class: "mt-4 max-w-xlg ml-auto", // Right-aligned paragraph
                            "Zero-shot image segmentation with ",
                            a {
                                href: "https://segment-anything.com",
                                class: "underline hover:text-blue-500",
                                target: "_blank",
                                "Segment Anything Model (SAM)"
                            },
                            " and ",
                            a {
                                href: "https://github.com/ChaoningZhang/MobileSAM",
                                class: "underline hover:text-blue-500",
                                target: "_blank",
                                "MobileSAM"
                            },
                            ". It runs in the browser with a WASM runtime built with ",
                            a {
                                href: "https://github.com/huggingface/candle",
                                class: "underline hover:text-blue-500",
                                target: "_blank",
                                "Candle"
                            }
                        }
                    }
                }
        }
        div {
            class: "flex h-screen",
            SideBar { 
                image: base_image, 
                mask: mask_image,
                message_state: message_state,
                toggle_mask: toggle_mask,
                data: sam_data
             }
            // match *download_state.read() {
            //     Some(Ok(ref bytes)) => rsx! { "Downloaded {bytes.len()} bytes" },
            //     Some(Err(ref err)) => rsx! { "Error: {err}" },
            //     None => rsx! { "Waiting to start..." },
            // }

            ImageContainer { 
                image: base_image,
                image_bytes: base_image_bytes,
                mask_image: mask_image,
                toggle_mask: toggle_mask,
                data: sam_data,
                model: sam_model,
                // settings: sam_settings
                } 
            }
         

        // ImageContainer { 
        //     image: base_image,
        //     image_url: base_image_bytes,
        //     model: sam_model,
        //     settings: sam_settings
        // }
    }
}