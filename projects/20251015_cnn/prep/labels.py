import os
import pandas as pd

DATASET_DIR = "dataset"

CLASSES = ["perro", "gato", "mariquita", "tortuga", "hormiga"]

rows = []

for cls in CLASSES:
    class_dir = os.path.join(DATASET_DIR, cls)

    if not os.path.isdir(class_dir):
        print(f"No existe la carpeta: {class_dir}")
        continue

    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(class_dir, fname)

            # Vector de 0/1
            labels = [1 if c == cls else 0 for c in CLASSES]

            rows.append([image_path] + labels)

df = pd.DataFrame(rows, columns=["image"] + CLASSES)

df.to_csv("labels.csv", index=False)

print("\nlabels.csv generado con éxito:")
print(df.head())
print(f"\nTotal de imágenes: {len(df)}")
