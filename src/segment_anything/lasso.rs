use dioxus::prelude::*;
use dioxus::logger::tracing::info;
use std::error::Error;

use image::{DynamicImage, GrayImage, RgbImage, ImageBuffer, Rgb};
use imageproc::{
    drawing::draw_filled_circle_mut,
    morphology::{dilate, erode},
    distance_transform::Norm,
};
use ndarray::{Array3, s};
use std::io::Cursor;


// #[server]
pub async fn lasso(
    image: &DynamicImage,
    mask: &DynamicImage,
    points: Vec<(f64, f64, bool)>,
) -> Vec<u8> {
    // Convert inputs to ndarray
    let image_rgb = image.to_rgb8();
    let mask_gray = mask.to_luma8();

    // Convert to ndarray for processing
    let image_array = Array3::from_shape_vec(
        (image_rgb.height() as usize, image_rgb.width() as usize, 3),
        image_rgb.clone().into_vec(),
    )
    .map_err(|e| {
        info!("Failed to convert image to ndarray: {}", e);
        // ServerFnError::ServerError("Image conversion failed".to_string())
    }).unwrap();
    let mask_array = Array3::from_shape_vec(
        (mask_gray.height() as usize, mask_gray.width() as usize, 1),
        mask_gray.into_vec(),
    )
    .map_err(|e| {
        info!("Failed to convert mask to ndarray: {}", e);
        // ServerFnError::ServerError("Mask conversion failed".to_string())
    }).unwrap();

    // Extract cutout (compute bounding box with padding)
    let padding = 20;
    let (min_x, min_y, w, h) = compute_bounding_box(&mask_array, padding, image_rgb.width(), image_rgb.height());
    let img_cut = image_array.slice(s![min_y..min_y+h, min_x..min_x+w, ..]).to_owned();
    let mut mask_cut = mask_array.slice(s![min_y..min_y+h, min_x..min_x+w, 0]).to_owned();

    // Verify mask_cut shape
    let mask_cut_shape = mask_cut.shape();
    if mask_cut_shape != &[h, w] {
        info!(
            "Invalid mask_cut shape: expected ({}, {}), got {:?}",
            h,
            w,
            mask_cut_shape
        );
        return get_image_bytes(DynamicImage::ImageRgb8(RgbImage::new(
            image_rgb.width(),
            image_rgb.height(),
        )))
        .await;
    }

    // Convert mask_cut to GrayImage for drawing
    let mask_vec = mask_cut
        .into_shape(h * w)
        .map_err(|e| {
            info!("Failed to reshape mask_cut: {}", e);
            // ServerFnError::ServerError("Mask reshape failed".to_string())
        }).unwrap()
        .into_raw_vec();


    let mut mask_luma = GrayImage::from_vec(w as u32, h as u32, mask_vec).ok_or_else(|| {
        info!("Failed to convert mask to GrayImage");
        // ServerFnError::ServerError("Mask conversion failed".to_string())
    }).unwrap();
    

    // Apply points to mask
    for (x, y, is_fg) in points {
        let cx = (x.round() as i32 - min_x as i32).max(0).min(w as i32 - 1);
        let cy = (y.round() as i32 - min_y as i32).max(0).min(h as i32 - 1);
        draw_filled_circle_mut(
            &mut mask_luma,
            (cx, cy),
            1,
            image::Luma([if is_fg { 255 } else { 0 }]),
        );
    }

    // Morphological operations
    let _kernel = GrayImage::from_vec(3, 3, vec![1u8; 9]).unwrap(); // 3x3 ones kernel
    let cleaned_mask = erode(&mask_luma, Norm::L1, 3);
    let final_mask = dilate(&cleaned_mask, Norm::L1, 3);

    let mut full_mask = RgbImage::new(image_rgb.width(), image_rgb.height());
    for y in 0..h {
        for x in 0..w {
            let mask_value = final_mask[(x as u32, y as u32)][0]; // Use (y, x) for GrayImage indexing
            full_mask.put_pixel(
                (min_x + x) as u32,
                (min_y + y) as u32,
                Rgb([mask_value, 0, 0]), // Red channel holds mask value, green/blue are 0
            );
        }
    }

    // Convert to DynamicImage and encode as PNG
    let output_image = DynamicImage::ImageRgb8(full_mask);
    get_image_bytes(output_image).await

    // // Apply mask to image
    // let mut out_cut = Array3::zeros((h, w, 3));
    // for y in 0..h {
    //     for x in 0..w {
    //         let mask_value = final_mask[(x as u32, y as u32)][0];
    //         let pixel = if mask_value > 50 {
    //             img_cut.slice(s![y, x, ..]).to_owned()
    //         } else {
    //             ndarray::arr1(&[0u8, 0u8, 0u8])
    //         };
    //         out_cut.slice_mut(s![y, x, ..]).assign(&pixel);
    //     }
    // }

    // // Paste back to full image
    // let mut full = Array3::zeros((image_rgb.height() as usize, image_rgb.width() as usize, 3));
    // let mut roi = full.slice_mut(s![min_y..min_y+h, min_x..min_x+w, ..]);
    // roi.assign(&out_cut);

    // // Convert back to DynamicImage
    // let full_vec = full.into_shape(image_rgb.width() as usize * image_rgb.height() as usize * 3)
    //     .map_err(|e| {
    //         info!("Failed to reshape ndarray: {}", e);
    //         // ServerFnError::ServerError("Image reshaping failed".to_string())
    //     }).unwrap()
    //     .into_raw_vec();
    // let output_image = ImageBuffer::<Rgb<u8>, _>::from_vec(
    //     image_rgb.width(),
    //     image_rgb.height(),
    //     full_vec,
    // )
    // .map(DynamicImage::ImageRgb8)
    // .ok_or_else(|| {
    //     info!("Failed to convert ndarray to DynamicImage");
    //     // ServerFnError::ServerError("Image conversion failed".to_string())
    // }).unwrap();

    // get_image_bytes(image::DynamicImage::ImageLuma8(final_mask)).await
}

// Helper function to compute bounding box with padding
fn compute_bounding_box(
    mask: &Array3<u8>,
    padding: i32,
    max_width: u32,
    max_height: u32,
) -> (usize, usize, usize, usize) {
    let (h, w, _) = mask.dim();
    let mut min_x = w;
    let mut max_x = 0;
    let mut min_y = h;
    let mut max_y = 0;

    for y in 0..h {
        for x in 0..w {
            if mask[[y, x, 0]] > 0 {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }

    let min_x = (min_x as i32 - padding).max(0) as usize;
    let min_y = (min_y as i32 - padding).max(0) as usize;
    let max_x = (max_x as i32 + padding).min(max_width as i32 - 1) as usize;
    let max_y = (max_y as i32 + padding).min(max_height as i32 - 1) as usize;
    let w = max_x - min_x + 1;
    let h = max_y - min_y + 1;
    (min_x, min_y, w, h)
}

async fn get_image_bytes(dynamic_image: DynamicImage) -> Vec<u8> {
    let mut bytes = Vec::new();
    dynamic_image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap_or_default();
    bytes
}

// // Helper to convert DynamicImage to OpenCV Mat
// fn dynamic_image_to_mat(image: &DynamicImage) -> Result<Mat, Box<dyn Error>> {
//     let rgb = image.to_rgb8();
//     let (width, height) = (rgb.width() as i32, rgb.height() as i32);
//     let mut mat = Mat::new_rows_cols_with_default(height, width, CV_8UC3, Scalar::all(0.0))?;
//     let data = rgb.into_raw();
//     let mat_data = mat.data_mut()?;
//     mat_data.copy_from_slice(&data);
//     Ok(mat)
// }

// // Helper to convert OpenCV Mat to DynamicImage
// fn mat_to_dynamic_image(mat: &Mat) -> Result<DynamicImage, Box<dyn Error>> {
//     if mat.typ() != CV_8UC3 {
//         return Err("Mat must be CV_8UC3".into());
//     }
//     let size = mat.size()?;
//     let mut rgb = RgbImage::new(size.width as u32, size.height as u32);
//     let data = mat.data_bytes()?;
//     for y in 0..size.height {
//         for x in 0..size.width {
//             let idx = (y * size.width * 3 + x * 3) as usize;
//             rgb.put_pixel(x as u32, y as u32, Rgb([data[idx], data[idx + 1], data[idx + 2]]));
//         }
//     }
//     Ok(DynamicImage::ImageRgb8(rgb))
// }

// // Helper to convert DynamicImage mask to OpenCV Mat (grayscale)
// fn mask_to_mat(mask: &DynamicImage) -> Result<Mat, Box<dyn Error>> {
//     let gray = mask.to_luma8();
//     let (width, height) = (gray.width() as i32, gray.height() as i32);
//     let mut mat = Mat::new_rows_cols_with_default(height, width, CV_8U, Scalar::all(0.0))?;
//     let data = gray.into_raw();
//     let mat_data = mat.data_mut()?;
//     mat_data.copy_from_slice(&data);
//     Ok(mat)
// }

// // Contour tracing using OpenCV
// fn trace_contour(mask: &Mat) -> Result<Vec<(i32, i32)>, Box<dyn Error>> {
//     let mut contours = VectorOfVectorOfPoint::new();
//     let mut hierarchy = Mat::default();
//     find_contours(
//         mask,
//         &mut contours,
//         &mut hierarchy,
//         imgproc::RETR_EXTERNAL,
//         imgproc::CHAIN_APPROX_SIMPLE,
//         Point::default(),
//     )?;

//     let mut contour_points = vec![];
//     if !contours.is_empty() {
//         let contour = contours.get(0)?;
//         for point in contour.iter() {
//             contour_points.push((point.x, point.y));
//         }
//         info!("Traced contour with {} points", contour_points.len());
//     } else {
//         info!("No contours found");
//     }
//     Ok(contour_points)
// }

// // Cutout extraction with padding
// fn extract_cutout(
//     image: &Mat,
//     fg_mask: &Mat,
//     padding: i32,
// ) -> Result<(Mat, Mat, (i32, i32, i32, i32)), Box<dyn Error>> {
//     let mut contours = VectorOfVectorOfPoint::new();
//     let mut hierarchy = Mat::default();
//     find_contours(
//         fg_mask,
//         &mut contours,
//         &mut hierarchy,
//         imgproc::RETR_EXTERNAL,
//         imgproc::CHAIN_APPROX_SIMPLE,
//         Point::default(),
//     )?;

//     if contours.is_empty() {
//         info!("No foreground pixels found in fg_mask");
//         return Err("No foreground pixels".into());
//     }

//     let contour = contours.get(0)?;
//     let bbox = bounding_rect(&contour)?;
//     let size = image.size()?;

//     // Apply padding, clamp to image bounds
//     let min_x = (bbox.x - padding).max(0);
//     let min_y = (bbox.y - padding).max(0);
//     let max_x = (bbox.x + bbox.width + padding).min(size.width - 1);
//     let max_y = (bbox.y + bbox.height + padding).min(size.height - 1);
//     let cutout_width = max_x - min_x;
//     let cutout_height = max_y - min_y;

//     info!(
//         "Cutout: x={}..{}, y={}..{}, size={}x{}",
//         min_x, max_x, min_y, max_y, cutout_width, cutout_height
//     );

//     let roi = Rect::new(min_x, min_y, cutout_width, cutout_height);
//     let image_cutout = Mat::roi(image, roi)?;
//     let mask_cutout = Mat::roi(fg_mask, roi)?;

//     Ok((image_cutout, mask_cutout, (min_x, min_y, cutout_width, cutout_height)))
// }

// // Main GrabCut function using OpenCV
// #[server]
// pub async fn grabcut(image: &DynamicImage, fg_mask: &DynamicImage) -> Result<DynamicImage, ServerFnError> {
//     info!("Starting grabcut with image dimensions {}x{}", image.width(), image.height());

//     // Convert inputs to OpenCV Mats
//     let image_mat = match dynamic_image_to_mat(image) {
//         Ok(mat) => mat,
//         Err(e) => {
//             info!("Failed to convert image to Mat: {}", e);
//             return DynamicImage::ImageRgb8(RgbImage::new(image.width(), image.height()));
//         }
//     };
//     let mut mask_mat = match mask_to_mat(fg_mask) {
//         Ok(mat) => mat,
//         Err(e) => {
//             info!("Failed to convert mask to Mat: {}", e);
//             return DynamicImage::ImageRgb8(RgbImage::new(image.width(), image.height()));
//         }
//     };

//     // Extract cutout
//     let padding = 50;
//     let cutout = match extract_cutout(&image_mat, &mask_mat, padding) {
//         Ok(cutout) => cutout,
//         Err(e) => {
//             info!("Failed to extract cutout: {}", e);
//             return DynamicImage::ImageRgb8(RgbImage::new(image.width(), image.height()));
//         }
//     };
//     let (image_cutout, mut mask_cutout, (min_x, min_y, cutout_width, cutout_height)) = cutout;

//     // Prepare for OpenCV GrabCut
//     let mut bgd_model = Mat::default();
//     let mut fgd_model = Mat::default();
//     let iter_count = 5;

//     // Run GrabCut
//     match grab_cut(
//         &image_cutout,
//         &mut mask_cutout,
//         Rect::default(), // Use mask instead of rectangle
//         &mut bgd_model,
//         &mut fgd_model,
//         iter_count,
//         imgproc::GC_INIT_WITH_MASK,
//     ) {
//         Ok(_) => info!("GrabCut completed successfully"),
//         Err(e) => {
//             info!("GrabCut failed: {}", e);
//             return DynamicImage::ImageRgb8(RgbImage::new(image.width(), image.height()));
//         }
//     }

//     // Create output mask
//     let mut output_cutout = Mat::new_rows_cols_with_default(
//         cutout_height,
//         cutout_width,
//         CV_8U,
//         Scalar::all(0.0),
//     ).unwrap();
//     for y in 0..cutout_height {
//         for x in 0..cutout_width {
//             let mask_value = mask_cutout.at_2d::<u8>(y, x).unwrap();
//             let value = if *mask_value == GC_FGD || *mask_value == GC_PR_FGD {
//                 GC_FGD
//             } else {
//                 GC_BGD
//             };
//             *output_cutout.at_2d_mut::<u8>(y, x).unwrap() = value;
//         }
//     }

//     // Map back to full image
//     let mut full_output = Mat::new_rows_cols_with_default(
//         image.height() as i32,
//         image.width() as i32,
//         CV_8U,
//         Scalar::all(0.0),
//     ).unwrap();
//     let roi = Rect::new(min_x, min_y, cutout_width, cutout_height);
//     let mut full_roi = Mat::roi(&mut full_output, roi).unwrap();
//     output_cutout.copy_to(&mut full_roi).unwrap();

//     // Convert to binary RGB output
//     let mut rgb_output = RgbImage::new(image.width(), image.height());
//     for y in 0..image.height() as i32 {
//         for x in 0..image.width() as i32 {
//             let value = if *full_output.at_2d::<u8>(y, x).unwrap() == GC_FGD {
//                 Rgb([255, 255, 255])
//             } else {
//                 Rgb([0, 0, 0])
//             };
//             rgb_output.put_pixel(x as u32, y as u32, value);
//         }
//     }

//     DynamicImage::ImageRgb8(rgb_output)
// }

