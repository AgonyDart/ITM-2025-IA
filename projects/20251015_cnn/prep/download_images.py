import os
from bing_image_downloader import downloader
import cv2
from PIL import Image
import imagehash
import numpy as np
from tqdm import tqdm
import shutil

CLASSES = {"tortuga": 100, "mariquita": 100}
OUTPUT_DIR = "dataset"


def download_images():
    for cls, count in CLASSES.items():
        print(f"\n---- Descargando imágenes de {cls} ----")
        downloader.download(
            query=cls,
            limit=count,
            output_dir=OUTPUT_DIR,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
        )


def is_valid_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return False
        h, w = img.shape[:2]
        return h > 20 and w > 20  # mínima calidad
    except:
        return False


def clean_corrupted():
    print("\n---- Eliminando imágenes corruptas ----")
    total = 0
    removed = 0

    for cls in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, cls, "Image")
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            total += 1

            if not is_valid_image(fpath):
                os.remove(fpath)
                removed += 1

    print(f"Total analizadas: {total} | Eliminadas: {removed}")


def remove_duplicates():
    print("\n---- Eliminando duplicados ----")

    for cls in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, cls, "Image")
        if not os.path.isdir(class_dir):
            continue

        hashes = {}
        removed = 0
        files = os.listdir(class_dir)

        for fname in tqdm(files, desc=f"Procesando {cls}"):
            fpath = os.path.join(class_dir, fname)

            try:
                img = Image.open(fpath)
                h = imagehash.phash(img)

                if h in hashes:
                    os.remove(fpath)
                    removed += 1
                else:
                    hashes[h] = fpath

            except:
                os.remove(fpath)
                removed += 1

        print(f"{cls}: {removed} duplicados eliminados")


def reorganize():
    print("\n---- Reorganizando carpetas ----")
    final_dir = os.path.join(OUTPUT_DIR, "cleaned")
    os.makedirs(final_dir, exist_ok=True)

    for cls in CLASSES:
        source = os.path.join(OUTPUT_DIR, cls, "Image")
        target = os.path.join(final_dir, cls)
        os.makedirs(target, exist_ok=True)

        if not os.path.isdir(source):
            continue

        for fname in os.listdir(source):
            shutil.move(os.path.join(source, fname), os.path.join(target, fname))

    print("Dataset organizado en: dataset/cleaned")


if __name__ == "__main__":
    download_images()
    clean_corrupted()
    remove_duplicates()
    reorganize()

    print("\n==== TODO LISTO ====")
