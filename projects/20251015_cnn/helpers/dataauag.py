import cv2
import numpy as np
from PIL import Image
import os


def augment_image(image_path, output_dir, num_augmentations=1000, size=128):
    """Generate augmented images from a single image."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_augmentations):
        # Random transformations
        # angle = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.8, 1.2)
        h, w = img_rgb.shape[:2]

        # Rotation and scaling
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        augmented = cv2.warpAffine(img_rgb, M, (w, h))

        # Random brightness and contrast
        augmented = np.clip(
            augmented * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255
        ).astype(np.uint8)

        # Resize to 128x128
        augmented = cv2.resize(augmented, (size, size))

        # Save
        output_path = os.path.join(output_dir, f"ladybugbr_{i:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))


# Usage
augment_image("turtle.png", "aug", num_augmentations=1000, size=128)
