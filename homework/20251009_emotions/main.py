import cv2
import os
import numpy as np

dataPath = "train"
emotionsList = os.listdir(dataPath)
print("Lista de emociones a entrenar: ", emotionsList)

facesData = []
labels = []
label = 0
MAX_PER_FOLDER = 1000  # máximo de imágenes por carpeta

for nameDir in emotionsList:
    emotionPath = os.path.join(dataPath, nameDir)
    print("Leyendo las imágenes de la emoción:", nameDir)

    count = 0
    for fileName in os.listdir(emotionPath):
        # if count >= MAX_PER_FOLDER:
        #     print(f"Se alcanzó el límite de {MAX_PER_FOLDER} imágenes para {nameDir}")
        #     break

        imagePath = os.path.join(emotionPath, fileName)
        image = cv2.imread(imagePath, 0)

        if image is None:
            print(f"Advertencia: No se pudo leer la imagen {imagePath}")
            continue

        # resized_image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)

        facesData.append(image)
        labels.append(label)
        count += 1

    label += 1

print(f"\nSe han procesado un total de {len(labels)} imágenes.")

face_recognizer = cv2.face.EigenFaceRecognizer_create()

print("Entrenando el modelo Eigen...")
face_recognizer.train(facesData, np.array(labels))

output_file = "modeloEigenFaceEmotions.xml"
face_recognizer.write(output_file)

print(f"¡Modelo entrenado y guardado exitosamente en '{output_file}'!")
