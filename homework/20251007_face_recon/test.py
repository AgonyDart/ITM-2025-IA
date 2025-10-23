import cv2 as cv
import os
import random

# === Cargar modelo y clasificadores ===
faceRecognizer = cv.face.EigenFaceRecognizer_create()
faceRecognizer.read("Eigenface80.xml")
faces = ["_angeld", "_godoy", "_jsiel", "_margaret", "_miguel"]
# faces = ["_godoy", "_margaret", "_angeld", "_jsiel", "_miguel"]
rostro = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in rostros:
        face_img = gray[y : y + h, x : x + w]
        face_img = cv.resize(face_img, (80, 80), interpolation=cv.INTER_CUBIC)
        label, confidence = faceRecognizer.predict(face_img)

        name = faces[label]

        if confidence > 2800:
            cv.putText(frame, name, (x, y - 25), 2, 1.1, (0, 255, 0), 2, cv.LINE_AA)

            if name == "_angeld":
                #     # Glitch effect: slice-shift, color blocks and scanlines
                #     roi = frame[y : y + h, x : x + w].copy()
                #     rows, cols = roi.shape[:2]

                #     # Random horizontal slices that are shifted and tinted
                #     for i in range(random.randint(12, 18)):
                #         y1 = random.randint(0, rows - 1)
                #         y2 = y1 + random.randint(1, max(2, rows // 8))
                #         y2 = min(rows, y2)
                #         slice = roi[y1:y2].copy()
                #         dx = random.randint(-w // 6, w // 6)
                #         dst_x = max(0, min(frame.shape[1] - cols, x + dx))
                #         tint = (
                #             random.randint(-40, 40),
                #             random.randint(-40, 40),
                #             random.randint(-40, 40),
                #         )
                #         try:
                #             slice = cv.add(slice, tint)  # tint the slice (clamped)
                #         except Exception:
                #             pass
                #         frame[y + y1 : y + y2, dst_x : dst_x + cols] = slice

                #     # Colored glitch blocks
                #     for i in range(random.randint(4, 8)):
                #         rx = random.randint(0, max(1, w - 1))
                #         ry = random.randint(0, max(1, h - 1))
                #         rw = random.randint(5, max(6, w // 5))
                #         rh = random.randint(2, max(3, h // 10))
                #         color = (
                #             random.randint(0, 255),
                #             random.randint(0, 255),
                #             random.randint(0, 255),
                #         )
                #         x1 = x + rx
                #         y1 = y + ry
                #         x2 = x + min(w, rx + rw)
                #         y2 = y + min(h, ry + rh)
                #         cv.rectangle(frame, (x1, y1), (x2, y2), color, -1)

                #     # Scanlines / noise lines across the face
                #     for i in range(0, h, 3):
                #         if random.random() > 0.25:
                #             color = (
                #                 random.randint(0, 255),
                #                 random.randint(0, 255),
                #                 random.randint(0, 255),
                #             )
                #             thickness = random.choice([1, 2])
                #             cv.line(frame, (x, y + i), (x + w, y + i), color, thickness)

                #     # Outer jagged rectangles for a noisy border
                #     for _ in range(3):
                #         dx = random.randint(-4, 4)
                #         dy = random.randint(-4, 4)
                #         color = (
                #             random.randint(50, 255),
                #             random.randint(50, 255),
                #             random.randint(50, 255),
                #         )
                #         cv.rectangle(
                #             frame, (x + dx, y + dy), (x + w + dx, y + h + dy), color, 1
                #         )
                # else:
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
