import os
import cv2

data_dir = os.path.join(
    os.getcwd(),
    "C:\\Users\\angel\\MyCode\\ITM-2025-IA\\projects\\20251015_cnn\\data\\gato",
)
print(f"Escaneando imágenes corruptas en: {data_dir}")

bad_files = 0
for root, dirs, files in os.walk(data_dir):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            img = cv2.imread(filepath)

            if img is None or img.size == 0:
                print(f"🗑️ Borrando archivo corrupto: {file}")
                os.remove(filepath)
                bad_files += 1
            else:
                if len(img.shape) < 3 or img.shape[2] != 3:
                    print(f"🗑️ Borrando imagen sin RGB: {file}")
                    os.remove(filepath)
                    bad_files += 1

        except Exception as e:
            print(f"💀 Error crítico en {file}: {e}")
            os.remove(filepath)
            bad_files += 1

print(f"✅ Limpieza terminada. Se eliminaron {bad_files} archivos basura.")
