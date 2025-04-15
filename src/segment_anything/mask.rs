
use image::{ImageBuffer, Rgba, GenericImageView};
use rand::{rng  , Rng};
use std::io::Cursor;

/// Generate a mask vector (u8) the same size as the given dimensions, 
/// with either random or uniform values.
/// 
/// In this example, 0 means "no overlay" and 255 means "overlay this pixel."
pub fn create_example_mask(width: u32, height: u32) -> Vec<u8> {
    // Example #1: A random binary mask
    //  - 0 = no overlay, 255 = overlay
    //  - Here, ~20% chance to overlay a given pixel
    let mut rng = rng();
    let mut mask = Vec::with_capacity((width * height) as usize);
    for _ in 0..(width * height) {
        if rng.random_bool(0.2) {
            mask.push(255);
        } else {
            mask.push(0);
        }
    }

    // Example #2 (uncomment if you want a uniform mask):
    // let mut mask = vec![255; (width * height) as usize];

    mask
}

/// Overlay a mask onto an RGBA image in-place. 
/// - `mask` is expected to have the same width/height as `base_image`.
/// - A mask value of 255 indicates "apply overlay color", 0 means "no overlay."
pub fn overlay_mask(
    base_image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    mask: &[u8],
    overlay_color: [u8; 4], // e.g., RGBA of the mask color
) {
    let (width, height) = base_image.dimensions();

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if mask[idx] > 0 {
                // If the mask says we should overlay this pixel, 
                // then blend with the overlay color.
                let px = base_image.get_pixel_mut(x, y);
                // Simple approach: replace with overlay color 
                // (i.e., ignoring original pixel).
                *px = blend_with_alpha(*px, Rgba(overlay_color));
            }
        }
    }
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