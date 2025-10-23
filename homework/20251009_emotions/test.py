import cv2 as cv

# === Cargar modelo Fisherfaces y lista de emociones entrenadas ===
faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.read("modeloFisherFaceEmotions.xml")
emociones = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

rostro = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)


cap = cv.VideoCapture(0)
# Ajusta este umbral según el comportamiento del modelo (en Fisher/LBPH, valor menor => mejor coincidencia)
THRESHOLD = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in rostros:
        face_img = gray[y : y + h, x : x + w]
        face_img = cv.resize(face_img, (48, 48), interpolation=cv.INTER_CUBIC)

        label, confidence = faceRecognizer.predict(face_img)

        if 0 <= label < len(emociones) and confidence < THRESHOLD:
            name = emociones[label]
            cv.putText(
                frame, f"{name}", (x, y - 25), 2, 1.1, (0, 255, 0), 2, cv.LINE_AA
            )
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv.putText(
                frame, "Desconocido", (x, y - 20), 2, 0.8, (0, 0, 255), 2, cv.LINE_AA
            )
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("frame", frame)
    k = cv.waitKey(1)
    if k == 27:  # ESC para salir
        break

cap.release()
cv.destroyAllWindows()
