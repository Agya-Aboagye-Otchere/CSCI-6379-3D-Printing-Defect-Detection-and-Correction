"""
preprocessing.py
================
Image Preprocessing and Data Augmentation Pipeline.
Authored by Hena Parween (Part 1: Data and Simulation).

Specifications:
1. Target input size: 224 x 224 x 3 (aligned with MobileNetV2 and ResNet-50).
2. Normalization: ImageNet channel statistics
   - mean = [0.485, 0.456, 0.406]
   - std  = [0.229, 0.224, 0.225]
3. Augmentations (applied to Training set only):
   - Random horizontal & vertical flips
   - Random rotation (+/- 15 degrees)
   - Hue, Saturation, Brightness, and Contrast jitter
     (Directly adopting methodology from De La Rosa et al., CIRP 2024)
4. Class imbalance weighting:
   - Inverse class frequency weights for weighted cross-entropy loss.
"""

import math
import random
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = (224, 224)


class ImageTransformPipeline:
    """
    Standard preprocessing and augmentation pipeline implemented using PIL & NumPy,
    ensuring zero-dependency portability and exact reproducibility across hardware.
    """
    def __init__(self, is_training: bool = False, target_size: Tuple[int, int] = TARGET_SIZE):
        self.is_training = is_training
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> np.ndarray:
        return self.process(img)

    def process(self, img: Image.Image) -> np.ndarray:
        """
        Processes an input PIL image: applies augmentations if is_training=True,
        resizes to target_size, scales pixels to [0, 1], and applies z-score normalization.
        """
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.is_training:
            # 1. Random Horizontal Flip (50% probability)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Random Vertical Flip (50% probability)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            # 3. Random Rotation (-15 to +15 degrees)
            angle = random.uniform(-15.0, 15.0)
            img = img.rotate(angle, resample=Image.BILINEAR)

            # 4. Color Jitter: Hue, Saturation, Brightness (De La Rosa et al. 2024)
            # Brightness jitter (0.85 to 1.15)
            brightness_factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

            # Contrast jitter (0.85 to 1.15)
            contrast_factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)

            # Color / Saturation jitter (0.80 to 1.20)
            color_factor = random.uniform(0.80, 1.20)
            img = ImageEnhance.Color(img).enhance(color_factor)

        # Resize to standard input dimensions (224x224)
        img = img.resize(self.target_size, resample=Image.BILINEAR)

        # Convert to numpy array and scale to [0, 1]
        img_arr = np.asarray(img, dtype=np.float32) / 255.0

        # Channel-wise normalization (x - mean) / std
        norm_img = (img_arr - IMAGENET_MEAN) / IMAGENET_STD

        # Return in HWC format (or CHW if needed)
        return norm_img


def get_pytorch_transforms_code() -> str:
    """
    Returns code snippet for CS teammate to drop directly into PyTorch pipeline.
    """
    return '''
import torchvision.transforms as T

train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.20, hue=0.10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''


def get_tensorflow_transforms_code() -> str:
    """
    Returns code snippet for CS teammate to drop directly into TensorFlow/Keras pipeline.
    """
    return '''
import tensorflow as tf
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.Resizing(224, 224),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.04), # approx +/- 15 deg
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
])

def preprocess_input_keras(x):
    # Normalized with ImageNet statistics
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)
'''


def compute_loss_class_weights(class_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Computes balanced class weights using standard formula:
    w_c = total_samples / (num_classes * class_samples)
    """
    total = sum(class_counts.values())
    k = len(class_counts)
    return {
        cls: round(total / (k * count), 4) if count > 0 else 1.0
        for cls, count in class_counts.items()
    }
