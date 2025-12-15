import cv2
import os
from pathlib import Path


def resize_images(input_folder, output_folder, size=(128, 128)):
    """Resize all images in input_folder to specified size and save to output_folder."""

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for filename in os.listdir(input_folder):
        if Path(filename).suffix.lower() in extensions:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read image
            img = cv2.imread(input_path)

            if img is not None:
                # Resize image
                resized = cv2.resize(img, size)

                # Save resized image
                cv2.imwrite(output_path, resized)
                print(f"Resized: {filename}")
            else:
                print(f"Failed to read: {filename}")


if __name__ == "__main__":
    input_dir = "C:\\Users\\angel\\Downloads\\data\\Data"
    output_dir = "C:\\Users\\angel\\Downloads\\data\\gato"

    resize_images(input_dir, output_dir)
