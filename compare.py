
import cv2
import numpy as np

# Vídeos que vamos a comparar
videos = {
    "ByteTrack": "outputs/tracking_bytetrack.mp4",
    "BoT-SORT": "outputs/tracking_botsort.mp4",
    "BoT-SORT + Re-ID": "outputs/tracking_botsort_reid.mp4",
    "OC-SORT": "outputs/tracking_ocsort.mp4"
}

caps = [cv2.VideoCapture(path) for path in videos.values()]

if not all(cap.isOpened() for cap in caps):
    raise RuntimeError("No se pudieron abrir todos los vídeos")

fps = caps[0].get(cv2.CAP_PROP_FPS)

width = 960
height = 540

writer = cv2.VideoWriter(
    "outputs/comparacion_trackers.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width * 2, height * 2)
)

if not writer.isOpened():
    raise RuntimeError("No se pudo crear el vídeo de salida")

while True:

    frames = []

    for cap, name in zip(caps, videos):

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Identificar el algoritmo
        cv2.rectangle(frame, (0, 0), (310, 38), (0, 0, 0), -1)

        cv2.putText(
            frame, name, (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        frames.append(frame)

    if len(frames) != 4:
        break

    # Combinar los cuatro vídeos
    top = np.hstack(frames[:2])
    bottom = np.hstack(frames[2:])

    combined = np.vstack((top, bottom))

    writer.write(combined)

for cap in caps:
    cap.release()

writer.release()

print("Comparación guardada correctamente")
