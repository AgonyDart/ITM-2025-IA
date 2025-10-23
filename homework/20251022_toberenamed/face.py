import cv2
import time
import numpy as np
import mediapipe as mp

LM = {
  "mouth_left": 61,
  "mouth_right": 291,
  "mouth_top": 13,
  "mouth_bottom": 14,
  "left_eye_top": 159,
  "left_eye_bottom": 145,
  "right_eye_top": 386,
  "right_eye_bottom": 374,
  "left_eyebrow_outer": 70,
  "left_eyebrow_inner": 105,
  "right_eyebrow_inner": 334,
  "right_eyebrow_outer": 300,
  "nose_tip": 1,
  "chin": 199,
}

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def dist(a, b):
  return np.linalg.norm(np.array(a) - np.array(b))


def get_pixel_landmark(landmark, frame_w, frame_h):
  return int(landmark.x * frame_w), int(landmark.y * frame_h)


def analyze(face_landmarks, w, h):
  L = {}
  for name, idx in LM.items():
    lm = face_landmarks.landmark[idx]
    L[name] = get_pixel_landmark(lm, w, h)

  mouth_width = dist(L["mouth_left"], L["mouth_right"])
  mouth_open = dist(L["mouth_top"], L["mouth_bottom"])
  face_height = dist(L["nose_tip"], L["chin"]) + 1e-6

  mouth_width_n = mouth_width / face_height
  mouth_open_n = mouth_open / face_height

  left_eye_open = dist(L["left_eye_top"], L["left_eye_bottom"])
  right_eye_open = dist(L["right_eye_top"], L["right_eye_bottom"])
  eye_open_n = ((left_eye_open + right_eye_open) / 2.0) / face_height

  left_eyebrow_y = (L["left_eyebrow_outer"][1] + L["left_eyebrow_inner"][1]) / 2.0
  right_eyebrow_y = (L["right_eyebrow_outer"][1] + L["right_eyebrow_inner"][1]) / 2.0
  eye_top_y = (L["left_eye_top"][1] + L["right_eye_top"][1]) / 2.0
  eyebrow_eye_diff_n = (
    eye_top_y - (left_eyebrow_y + right_eyebrow_y) / 2.0
  ) / face_height

  emotion = "neutral"
  if mouth_open_n > 0.1:
    emotion = "surprised"
  elif mouth_width_n > 0.80:
    emotion = "happy"
  elif mouth_open_n < 0.05 and mouth_width_n < 0.64:
    emotion = "sad"
  elif eyebrow_eye_diff_n < -0.02:
    emotion = "angry"
  else:
    emotion = "neutral"

  status = {
    "emotion": emotion,
    "mouth_width_n": mouth_width_n,
    "mouth_open_n": mouth_open_n,
    "eye_open_n": eye_open_n,
    "eyebrow_eye_diff_n": eyebrow_eye_diff_n,
  }
  return L, status


def main(source=0):
  cap = cv2.VideoCapture(source)
  prev_time = 0
  with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
  ) as face_mesh:
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      h, w = frame.shape[:2]
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(rgb)

      if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
        face_landmarks = results.multi_face_landmarks[0]
        L, status = analyze(face_landmarks, w, h)

        mp_drawing.draw_landmarks(
          image=frame,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        for name, p in L.items():
          cv2.circle(frame, p, 2, (0, 255, 0), -1)

        cv2.putText(
          frame,
          f"Emotion: {status['emotion']}",
          (10, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.9,
          (0, 255, 255),
          2,
        )
        cv2.putText(
          frame,
          f"mouth_w:{status['mouth_width_n']:.2f} open:{status['mouth_open_n']:.2f}",
          (10, 60),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (200, 200, 200),
          1,
        )
        cv2.putText(
          frame,
          f"eyes:{status['eye_open_n']:.2f} brow_diff:{status['eyebrow_eye_diff_n']:.2f}",
          (10, 80),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (200, 200, 200),
          1,
        )

      cur_time = time.time()
      fps = 1 / (cur_time - prev_time + 1e-6)
      prev_time = cur_time
      cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (w - 90, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
      )

      cv2.imshow("Emotion (heuristic) - press q to quit", frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main(0)
