

use candle_core::{DType, IndexOp, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam;

use dioxus::prelude::*;
use image::{ImageBuffer, Rgba, Rgb};
use image::DynamicImage;
use dioxus::logger::tracing::info;
use serde;

#[derive(serde::Serialize, serde::Deserialize)]
struct Mask {
    iou: f32,
    mask_shape: Vec<usize>,
    mask_data: Vec<u8>,
}
struct Image {
    original_width: u32,
    original_height: u32,
    width: u32,
    height: u32,
}

struct MaskImage {
    mask: Mask,
    image: Image,
}

#[derive(Debug)]
struct Embeddings {
    original_width: usize,
    original_height: usize,
    width: usize,
    height: usize,
    data: Tensor,
}


#[derive(Debug)]
pub struct SegmentAnythingModel {
    sam: sam::Sam,
    embeddings: Option<Embeddings>,
    device: Device
}

impl SegmentAnythingModel {
    pub fn builder() -> SegmentAnythingModelBuilder {
        SegmentAnythingModelBuilder {
            weights: Vec::new(),
            model_name: String::new(),
        }
    }

    pub async fn set_image_embeddings(&mut self, image: DynamicImage) -> Result<(), String> {
        
        let (original_height, original_width) = (image.height() as usize, image.width() as usize);
        let img = {
                let resize_longest = sam::IMAGE_SIZE;
                let (height, width) = (image.height(), image.width());
                let resize_longest = resize_longest as u32;
                let (height, width) = if height < width {
                    let h = (resize_longest * height) / width;
                    (h, resize_longest)
                } else {
                    let w = (resize_longest * width) / height;
                    (resize_longest, w)
                };
                image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        };        
        let (height, width) = (img.height() as usize, img.width() as usize);
        let img_data = img.to_rgb8().into_raw();
        let img_tensor = Tensor::from_vec(
            img_data,
            (img.height() as usize, img.width() as usize, 3),
            &Device::Cpu,
        )
        .map_err(|e| e.to_string())?
        .permute((2, 0, 1))
        .map_err(|e| e.to_string())?;
    
        let data = self.sam.embeddings(&img_tensor).map_err(|e| e.to_string())?;
        info!("[Set Image Embedding] collected embeddings");


        self.embeddings = Some(Embeddings {
            original_width,
            original_height, 
            width,
            height,
            data,
        });
        Ok(())
    }

    pub async fn segment_from_points(&self, transformed_points: Vec<(f64, f64, bool)>) -> Result<DynamicImage, String> {
        // let transformed_points: Vec<(f64, f64, bool)> = goal_points.into_iter().map(|(x, y, mask)| (x, y, mask)).collect();
        for &(x, y, _) in &transformed_points {
            if !(0.0..=1.0).contains(&x) {
                return Err(format!("x has to be between 0 and 1, got {}", x));
            }
            if !(0.0..=1.0).contains(&y) {
                return Err(format!("y has to be between 0 and 1, got {}", y));
            }
        }
        let embeddings = self.embeddings.as_ref().ok_or("image embeddings have not been set")?;

        let original_height = embeddings.original_height as usize;
        let original_width = embeddings.original_width as usize;
        let height = embeddings.height as usize;
        let width = embeddings.width as usize;
        info!("embedding height {:?}", embeddings.original_height);
        info!("embedding width {:?}", embeddings.original_width);
        let (low_res_mask, iou_predictions) = self.sam.forward_for_embeddings(
            &embeddings.data,
            embeddings.height as usize,
            embeddings.width as usize,
            &transformed_points,
            false,
        ).unwrap();
        info!("low res mask {:?}", low_res_mask);
        let mask = low_res_mask
        .upsample_nearest2d(sam::IMAGE_SIZE,sam::IMAGE_SIZE).unwrap()
        .get(0).unwrap()
        .i((.., ..height, ..width)).unwrap();

        let mask = (mask.ge(0f32).unwrap() * 255.).unwrap();
        let (_one, h, w) = mask.dims3().unwrap();
        let mask = mask.expand((3, h, w)).unwrap();

        let mask_pixels = mask.permute((1, 2, 0)).unwrap().flatten_all().unwrap().to_vec1::<u8>().unwrap();
        let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels).unwrap();

        Ok(DynamicImage::from(mask_img).resize_to_fill(
            original_width as u32,
            original_height as u32,
            image::imageops::FilterType::CatmullRom,
        ))
    }
}

pub fn sam_resize_image(image: DynamicImage) -> DynamicImage {
    let resize_longest = sam::IMAGE_SIZE;
    let (height, width) = (image.height(), image.width());
    let resize_longest = resize_longest as u32;
    let (height, width) = if height < width {
        let h = (resize_longest * height) / width;
        (h, resize_longest)
    } else {
        let w = (resize_longest * width) / height;
        (resize_longest, w)
    };
    image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
}


pub struct SegmentAnythingModelBuilder {
    weights: Vec<u8>,
    model_name: String,
}

impl SegmentAnythingModelBuilder {
    pub fn weights(mut self, weights: Vec<u8>) -> Self {
        self.weights = weights;
        self
    }

    pub fn use_model(mut self, model_name: String) -> Self {
        self.model_name = model_name;
        self
    }

    pub async fn build(self) -> Result<SegmentAnythingModel, String> {
        let device = Device::Cpu;
        // let device = Device::new_webgpu(0).await?;
        
        let vb = VarBuilder::from_buffered_safetensors(self.weights, DType::F32, &device).map_err(|e| e.to_string())?;
        let sam = if self.model_name == "mobile_sam-tiny-vitt.safetensors" {
            sam::Sam::new_tiny(vb)
        } else {
            sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)
        }.map_err(|e| e.to_string())?;
        Ok(SegmentAnythingModel {
            sam,
            embeddings: None,
            device
        })
    }
}

// Define an enum to represent the model state
#[derive(Debug)]
pub enum ModelState {
    Waiting,
    Loading,                      // Initial state while loading
    Loaded(SegmentAnythingModel),  // Successful load with the model
    Ready(SegmentAnythingModel),
    Error(String),                // Error state with a message
}

#[derive(Debug, Clone)]
pub enum ImageState {
    Empty,                            // Initial empty state (e.g., 1x1 image)
    Loading,                          // While loading the image
    Ready(DynamicImage), // Successfully loaded image
    Error(String),                    // Error during loading
}

#[derive(Debug, Clone)]
pub enum MaskState {
    Waiting,                          // While loading the image
    Ready(DynamicImage), // Successfully loaded image
}


#[derive(Debug, Clone, PartialEq, Copy)]
pub struct SegmentAnythingData {
    pub model_name: Signal<Option<String>>,
    pub goal_points: Signal<Vec<(f64, f64)>>,
    pub avoid_points: Signal<Vec<(f64, f64)>>,
}

impl SegmentAnythingData {
    pub fn set_goal_points(&mut self, points: Vec<(f64, f64)>) {
        self.goal_points.set(points);
    }
    /// Add a point to the list of points to segment.
    pub fn add_goal_point(&mut self, x: impl Into<f64>, y: impl Into<f64>) {
        self.goal_points.write().push((x.into(), y.into()));
    }
    pub fn remove_goal_point(&mut self) -> bool {
        self.goal_points.write().pop().is_some()
    }    
    pub fn clear_all_points(&mut self) {
        self.goal_points.write().clear();
        self.avoid_points.write().clear();
    }    
    /// Add a point to the list of points to avoid.
    pub fn add_avoid_point(&mut self, x: impl Into<f64>, y: impl Into<f64>) {
        self.avoid_points.write().push((x.into(), y.into()));
    }
    pub fn remove_avoid_point(&mut self) -> bool {
        self.avoid_points.write().pop().is_some()
    }        
}

// Define an enum to represent the model state
#[derive(Debug, Clone)]
pub enum MessageState {
    Waiting,
    ModelLoading,                      // Initial state while loading
    ModelLoaded,  // Successful load with the model
    EmbeddingsLoading,
    Ready,
}
