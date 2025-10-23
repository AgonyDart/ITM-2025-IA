import cv2
import mediapipe as mp
import numpy as np
import math

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

def draw_rotated_square(frame, center, size, angle_deg, color=(0,255,0), thickness=-1):
  """Dibuja un cuadrado centrado en `center`, tamaño `size` y rotado `angle_deg` grados."""
  h = size / 2.0
  # puntos del cuadrado sin rotación (relativos al centro)
  pts = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float32)
  # matriz de rotación
  theta = math.radians(angle_deg)
  rot = np.array([[math.cos(theta), -math.sin(theta)],
          [math.sin(theta),  math.cos(theta)]], dtype=np.float32)
  pts_rot = pts.dot(rot.T) + np.array(center, dtype=np.float32)
  pts_int = np.int32(pts_rot)
  cv2.fillConvexPoly(frame, pts_int, color) if thickness == -1 else cv2.polylines(frame, [pts_int], True, color, thickness)

# Captura de video
cap = cv2.VideoCapture(0)

# parámetros para escalar cuadrado
MIN_SIZE = 30
MAX_SIZE = 300
# distancia máxima estimada (diagonal de la imagen) se calcula cuando tengamos frame size
max_dist_px = None

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  h, w = frame.shape[:2]
  if max_dist_px is None:
    max_dist_px = math.hypot(w, h)

  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_rgb)

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      # Obtener punta del pulgar (4) y punta del índice (8)
      lm = hand_landmarks.landmark
      thumb = lm[4]
      index = lm[8]

      tx, ty = int(thumb.x * w), int(thumb.y * h)
      ix, iy = int(index.x * w), int(index.y * h)

      # Dibujar línea entre punta del pulgar y punta del índice
      cv2.line(frame, (tx, ty), (ix, iy), (0, 200, 255), 3, cv2.LINE_AA)
      cv2.circle(frame, (tx, ty), 6, (0,255,0), -1)
      cv2.circle(frame, (ix, iy), 6, (0,255,0), -1)

      # Distancia en píxeles
      dx, dy = ix - tx, iy - ty
      dist_px = math.hypot(dx, dy)

      # Escalar tamaño del cuadrado según la distancia
      t = np.clip(dist_px / max_dist_px, 0.0, 1.0)
      size = int(MIN_SIZE + t * (MAX_SIZE - MIN_SIZE))

      # Ángulo de rotación basado en la orientación entre los puntos (además se puede añadir factor)
      angle = math.degrees(math.atan2(dy, dx))

      # Centro del cuadrado en la esquina superior izquierda (respeta margen)
      margin = 20
      center = (margin + size // 2, margin + size // 2)

      # Dibujar cuadrado rotado
      draw_rotated_square(frame, center, size, angle, color=(50, 180, 255), thickness=-1)

      # Opcional: mostrar valores
      cv2.putText(frame, f"Dist: {int(dist_px)} px", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
      cv2.putText(frame, f"Size: {size}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

  cv2.imshow("Thumb-Index Visual", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
