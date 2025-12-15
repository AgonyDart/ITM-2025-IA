import cv2
import os
import yt_dlp


def extraer_frames(video_path, output_folder, size=(128, 128), saltar_frames=10):
    if not os.path.exists(video_path):
        print(f"❌ Error: No encuentro el video")
        return

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    print(f"📸 Extrayendo frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % saltar_frames == 0:
            try:
                frame_resized = cv2.resize(frame, size)
                nombre_archivo = f"{output_folder}/ant_yt02_{count:04d}.png"
                cv2.imwrite(nombre_archivo, frame_resized)
                saved_count += 1
            except Exception as e:
                pass
        count += 1

    cap.release()
    print(f"✅ Se guardaron {saved_count} imágenes en '{output_folder}'.")


def procesar_youtube(url, carpeta_destino, saltar_frames=30):
    """
    Descarga un video de YouTube y lo convierte en dataset.
    saltar_frames: Auméntalo para videos de YT que suelen ser largos (30 o 60).
    """
    print(f"⬇️ Descargando video de YouTube: {url}...")

    nombre_temporal = "temp_video.mp4"
    ydl_opts = {
        "format": "best[ext=mp4]",  # Forzar mp4 para asegurar compatibilidad con OpenCV
        "outtmpl": nombre_temporal,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        extraer_frames(nombre_temporal, carpeta_destino, saltar_frames=saltar_frames)

        if os.path.exists(nombre_temporal):
            os.remove(nombre_temporal)
            print("🗑️ Video temporal eliminado.")

    except Exception as e:
        print(f"❌ Error con YouTube: {e}")


url_youtube = "https://www.youtube.com/shorts/oHQfpOO7CNg"

carpeta = r"yt"

procesar_youtube(url_youtube, carpeta, saltar_frames=1)
