use dioxus::prelude::*;

use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgba};
use rand::{rng, Rng, random_range};
use base64::{engine::general_purpose, Engine as _};
use std::io::Cursor;
use dioxus::logger::tracing::info;

use super::model::*;

#[derive(Props, Debug, PartialEq, Clone)]
pub struct ImageContainerProps {
    pub image: Signal<ImageState>,
    pub image_bytes: Signal<String>,
    pub mask_image: Signal<MaskState>,
    pub toggle_mask: Signal<bool>,
    pub data: SegmentAnythingData,
    pub model: Signal<ModelState>,
    // pub settings: Signal<SegmentAnythingInferenceSettings>
}

struct RenderedPosition {
    x: u32,
    y: u32
}

#[component]
pub fn ImageContainer(mut props: ImageContainerProps) -> Element {

    let reactive_read_data = props.data.model_name.read();
    // Signal to hold the display-ready image
    let display_image = use_signal(|| None::<DynamicImage>);

    // Update display_image and settings reactively based on props.image
    use_effect(move || {
        let mut display_image = display_image.clone();
        // let mut settings = props.settings.clone();

        match &*props.image.read() {
            ImageState::Ready(image) => {
                let height = image.height();
                let width = image.width();
                let size = image.as_bytes().len();
                info!("bheight {height} bwidth {width} bsize {size}");
                display_image.set(Some(image.clone()));
                // settings.set(SegmentAnythingInferenceSettings::new(image.clone()));
            }
            _ => {
                display_image.set(None);
                // settings.set(SegmentAnythingInferenceSettings::new(
                //     ImageBuffer::<Rgba<u8>, Vec<u8>>::new(1, 1),
                // ));
            }
        };

    });

    // let mut click_positions = use_signal(|| Vec::<(f64, f64)>::new());
    let mut rendered_position = use_signal(|| RenderedPosition { x: 0, y: 0 });

    let handle_click = move |evt: MouseEvent| {


        info!("[Handle Click] start");
        let image_dims = match &*props.image.read() {
            ImageState::Ready(img) => (img.width(), img.height()),
            _ => (1, 1),
        };
        let (image_width, image_height) = image_dims;

        let pixel_position = rendered_position.read();
        let pixel_x = pixel_position.x;
        let pixel_y = pixel_position.y;

        let point = evt.element_coordinates();
        let x = (point.x as u32) * image_width / pixel_x.max(1);
        let y = (point.y as u32) * image_height / pixel_y.max(1);

        let x_normalized = x as f64 / image_width as f64;
        let y_normalized = y as f64 / image_height as f64;

        info!("coordinates {:?}", point);
        info!("calculated {:?}", (x, y));
        info!("normalized {:?}", (x_normalized, y_normalized));
        info!("bounding box {:?}", (pixel_x, pixel_y));
        info!("image {:?}", (image_width, image_height));

        if *props.toggle_mask.read() {
            props.data.add_goal_point(x_normalized, y_normalized);
        }
        else {
            props.data.add_avoid_point(x_normalized, y_normalized);
        }
        
        // click_positions.push((x_normalized, y_normalized));

        
        
            // let positions = click_positions.read().clone();
            // info!("Processing click positions: {:?}", positions);

            // {
            //     let mut settings = props.settings.write();
            //     settings.set_goal_points(positions);
            //     info!("Updated settings: {:?}", settings.goal_points);
            // }

            // let current_settings = props.settings.read().clone();
        
    };


    let segment_points = use_resource(move || {

        let data = props.data;
        let goal_points = &data.goal_points;
        let goal_points = goal_points.read().to_vec();

        let goal_points_empty = goal_points.clone().is_empty();

        let avoid_points = &data.avoid_points;
        let avoid_points = avoid_points.read().to_vec();

        let points = {
            let mut points = Vec::new();
            for (x, y) in goal_points {
                points.push((x, y, true));
            }
            for (x, y) in avoid_points {
                points.push((x, y, false));
            }
            points
        };
        
        async move {
            if goal_points_empty {
                info!("Goal points empty");
                if let Some(image) = &*display_image.read() {
                    let mut buf = Vec::new();
                    image.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
                        .map_err(|e| format!("Encoding error: {}", e)).unwrap();

                    info!("Model Read masked image");
                    let base64 = general_purpose::STANDARD.encode(buf);
                    props.image_bytes.set(format!("data:image/png;base64,{}", base64));
                }
            }
            else {

                match &*props.model.read() {
                    ModelState::Loaded(model) => {  

                        let mask = model.segment_from_points(points.clone()).await.unwrap();
                        // Save the base mask 
                        props.mask_image.set(MaskState::Ready(mask.clone()));

                        if let Some(img) = &*display_image.read() {
                            
                            let masked_image = overlay_mask(img.clone(), mask);

                            let masked_image = overlay_points_on_image(masked_image.clone(), points);

                            // masked_image.save("rust_with_overlay.png").expect("Failed to save overlaid image");
                            let mut buf = Vec::new();
                            masked_image.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
                                .map_err(|e| format!("Encoding error: {}", e)).unwrap();

                            info!("Model Read masked image");
                            // Ok::<_, String>()
                            props.image_bytes.set(format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(buf)));
                            }

                    }
                    _ => {

                    }
                }; 
            }
        }
    });

    // Render based on display_image
    rsx! {

            div {
                class: "sa-resizable-image-container flex-1 p-4",
                div {
                    class: "image-wrapper",
                
                match *display_image.read() {
                    Some(ref img) => rsx! {
                        img {
                            id: "sam-img",
                            class: "sa-img",
                            src: "{*props.image_bytes.read()}",
                            onclick: handle_click,
                            onresize: move |element| {
                                if let Ok(rect) = element.get_border_box_size() {
                                    rendered_position.set(RenderedPosition {
                                        x: rect.width as u32,
                                        y: rect.height as u32,
                                    });
                                } else {
                                    info!("Failed to get client rect");
                                }
                            },
                            style: "cursor: crosshair;"
                        }
                    },
                    None => rsx! { div {
                        class: "image-placeholder",
                        "Select or Upload an Image"
                    } },
                }
            }
        }
    }
}



pub fn overlay_mask(
    base_image: DynamicImage,
    mask: DynamicImage,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {

    let base_image = match base_image {
        DynamicImage::ImageRgba8(img) => img,
        _ => base_image.to_rgba8(),
    };

    let (width, height) = base_image.dimensions();
    let mask_rgb = mask.to_rgb8();
    let mut modified_image = base_image.clone();

    // Create a random overlay color.
    let mut rng = rand::thread_rng();
    let overlay_color = [
        rng.gen_range(0..=255) as u8,
        rng.gen_range(0..=255) as u8,
        rng.gen_range(0..=255) as u8,
    ];
    let overlay_alpha = 0.6_f32;

    // For each pixel, if the mask indicates a value above a threshold, overlay the color.
    for y in 0..height {
        for x in 0..width {
            let mask_pixel = mask_rgb.get_pixel(x, y);
            // Here we use the red channel as an indicator.
            if mask_pixel[0] > 100 {
                let base_pixel = *modified_image.get_pixel(x, y);
                let overlay_px = Rgba([
                    overlay_color[0],
                    overlay_color[1],
                    overlay_color[2],
                    (overlay_alpha * 255.0) as u8,
                ]);
                let blended = blend_with_alpha(base_pixel, overlay_px);
                modified_image.put_pixel(x, y, blended);
            }
        }
    }
    modified_image
}


/// Blends `src` and `overlay` using the overlay's alpha channel.
/// The simplest approach:
///   result = (overlay_alpha * overlay) + (1 - overlay_alpha)*src
pub fn blend_with_alpha(src: Rgba<u8>, overlay: Rgba<u8>) -> Rgba<u8> {
    let oa = overlay[3] as f32 / 255.0;
    let sa = src[3] as f32 / 255.0;

    // Composite alpha
    let out_a = oa + sa * (1.0 - oa);

    // For each channel: (r, g, b)
    let blend_channel = |sc, oc| {
        let sf = sc as f32 / 255.0;
        let of = oc as f32 / 255.0;
        // Overlay formula:
        ((of * oa) + (sf * sa * (1.0 - oa))) / out_a
    };

    if out_a > 0.0 {
        let r = blend_channel(src[0], overlay[0]);
        let g = blend_channel(src[1], overlay[1]);
        let b = blend_channel(src[2], overlay[2]);
        let a = out_a;
        Rgba([
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
            (a * 255.0) as u8,
        ])
    } else {
        // If alpha is zero, just return transparent
        Rgba([0, 0, 0, 0])
    }
}


fn overlay_points_on_image(
    mut image: ImageBuffer<Rgba<u8>, Vec<u8>>,
    points: Vec<(f64, f64, bool)>, // Normalized points (x, y) in [0, 1]
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    // Get image dimensions
    let (width, height) = (image.width(), image.height());
    // Circle color
    let goal_color = Rgba([0, 220, 100, 255]); // Red, fully opaque
    let avoid_color = Rgba([220, 0, 100, 255]); // Red, fully opaque

    // Calculate diameter: 0.005 * sqrt(height^2 + width^2)
    let diagonal = ((height as f64).powi(2) + (width as f64).powi(2)).sqrt();
    let diameter = 0.005 * diagonal;
    let circle_radius = (diameter / 2.0).max(1.0).round() as u32; // Ensure at least 1 pixel
    
     // Draw a filled circle for each point
     for (x, y, mask) in points {
        // Scale normalized coordinates to image dimensions
        let pixel_x = (x * width as f64).round() as i32;
        let pixel_y = (y * height as f64).round() as i32;

        let circle_color = {
            if mask {
                goal_color
            }
            else {
                avoid_color
            }
        };
        
        // Draw a filled circle
        for dy in -(circle_radius as i32)..=(circle_radius as i32) {
            for dx in -(circle_radius as i32)..=(circle_radius as i32) {
                // Check if the pixel is within the circle
                if dx * dx + dy * dy <= (circle_radius as i32).pow(2) {
                    let px = pixel_x + dx;
                    let py = pixel_y + dy;
                    

                    // Ensure the pixel is within image bounds
                    if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                        image.put_pixel(px as u32, py as u32, circle_color);
                    }
                }
            }
        }
    }
    
    image
}

