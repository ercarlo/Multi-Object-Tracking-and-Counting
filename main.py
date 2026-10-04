
import cv2
import math
from pathlib import Path
from ultralytics import YOLO
import csv
import argparse
import time
import json
import numpy as np
import torch

# Configuración
# Selección del algoritmo de tracking
parser = argparse.ArgumentParser()

parser.add_argument(
    "--tracker",
    default="bytetrack",
    choices=["bytetrack", "botsort", "botsort_reid", "ocsort"]
)

args = parser.parse_args()

TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
    "botsort_reid": "bot_reid.yaml",
    "ocsort": "ocsort.yaml"
}

TRACKER_FILE = TRACKERS[args.tracker]

VIDEO_PATH = "videos/video.mp4"
OUTPUT_PATH = f"outputs/tracking_{args.tracker}.mp4"
EVENTS_PATH = f"outputs/events_{args.tracker}.csv"

model = YOLO("yolo11s.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el vídeo")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

Path("outputs").mkdir(exist_ok=True)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("No se pudo crear el vídeo")

# Líneas de conteo adaptadas a la resolución
ped_line = (
    (int(0.393 * width), int(0.620 * height)),
    (int(0.803 * width), int(0.380 * height))
)

vehicle_line = (
    (int(0.607 * width), int(0.242 * height)),
    (int(0.607 * width), int(0.839 * height))
)


class LineCounter:

    def __init__(self, p1, p2, band=12, confirm=2):
        self.p1 = p1
        self.p2 = p2
        self.band = band
        self.confirm = confirm
        self.states = {}
        self.max_gap = max(1, int(fps * 0.75))
        self.cooldown = max(1, int(fps * 0.65))

    def update(self, track_id, point, frame_idx):

        ax, ay = self.p1
        bx, by = self.p2
        px, py = point

        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy

        # Comprobar que el punto está dentro del segmento
        projection = (
            (px - ax) * dx + (py - ay) * dy
        ) / length_sq


        # Distancia con signo respecto a la línea
        distance = (
            dx * (py - ay) - dy * (px - ax)
        ) / math.sqrt(length_sq)

        if abs(distance) <= self.band:
            return None

        side = 1 if distance > 0 else -1

        state = self.states.get(track_id)

        # Inicializar o reiniciar una trayectoria antigua
        if state is None or frame_idx - state["last"] > self.max_gap:
            self.states[track_id] = {
                "stable": side if self.confirm == 1 else None,
                "candidate": side,
                "hits": 1,
                "last": frame_idx,
                "last_count": -float("inf")
            }
            return None

        state["last"] = frame_idx

        # Confirmar el lado mediante observaciones sucesivas
        if state["candidate"] == side:
            state["hits"] += 1
        else:
            state["candidate"] = side
            state["hits"] = 1

        if state["hits"] < self.confirm:
            return None

        previous = state["stable"]
        state["stable"] = side

        if previous is None or previous == side:
            return None
        if not 0 <= projection <= 1:
            return None

        # Evitar eventos repetidos demasiado próximos
        if frame_idx - state["last_count"] < self.cooldown:
            return None

        state["last_count"] = frame_idx

        return 0 if previous == -1 else 1

    def prune(self, frame_idx):
        self.states = {
            key: state
            for key, state in self.states.items()
            if frame_idx - state["last"] <= self.max_gap
        }


# Contador de peatones
ped_counter = LineCounter(*ped_line)

# Contador de bicicletas (misma línea azul)
bike_counter = LineCounter(
    *ped_line,
    band=6,
    confirm=1
)

# Contador de coches (línea naranja)
vehicle_counter = LineCounter(*vehicle_line)

counts = {
    "Peatones": [0, 0],
    "Coches": [0, 0],
    "Bicicletas": [0, 0]
}

events = []

frame_idx = 0

# Configuración del rendimiento
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WARMUP_FRAMES = 10

track_times = []
total_times = []

def synchronize():
    if DEVICE != "cpu":
        torch.cuda.synchronize()

# Procesamiento del vídeo
while True:

    start_total = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break
    synchronize()

    start_track = time.perf_counter()
    result = model.track(

        source=frame,
        persist=True,
        tracker=TRACKER_FILE,
        classes=[0,1, 2],
        conf=0.10,
        imgsz=960,
        verbose=False
    )[0]

    synchronize()
    end_track = time.perf_counter()

    track_ms = (end_track - start_track) * 1000

    annotated = result.plot()
    boxes = result.boxes

    if boxes is not None and boxes.id is not None:

        for box, track_id, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.int().cpu().tolist(),
            boxes.cls.int().cpu().tolist()
        ):

            x1, y1, x2, y2 = box


            if cls in (0, 1):

                # Peatones y bicicletas: línea azul
                point = ((x1 + x2) / 2, y2)

                counter = ped_counter if cls == 0 else bike_counter

                direction = counter.update(
                    (cls, track_id),
                    point,
                    frame_idx
                )

                # Dibujar el punto de referencia de la bicicleta
                if cls == 1:
                    cv2.circle(
                        annotated,
                        (int(point[0]), int(point[1])),
                        6,
                        (0, 255, 0),
                        -1
                    )

                if direction is not None:
                    name = "Peatones" if cls == 0 else "Bicicletas"

                    counts[name][direction] += 1

                    events.append([
                        frame_idx,
                        round(frame_idx / fps, 2),
                        name,
                        track_id,
                        f"S{direction + 1}"
                    ])

            elif cls == 2:

                # Coches: línea naranja
                point = ((x1 + x2) / 2, (y1 + y2) / 2)

                direction = vehicle_counter.update(
                    (cls, track_id), point, frame_idx
                )

                if direction is not None:
                    counts["Coches"][direction] += 1

                    events.append([
                        frame_idx,
                        round(frame_idx / fps, 2),
                        "Coches",
                        track_id,
                        f"S{direction + 1}"
                    ])


    # Dibujar las líneas de conteo
    cv2.line(annotated, *ped_line, (255, 200, 0), 3)
    cv2.line(annotated, *vehicle_line, (0, 145, 255), 3)

    # Mostrar los contadores
    for i, (name, values) in enumerate(counts.items()):

        text = f"{name}: S1={values[0]} S2={values[1]}"

        cv2.putText(
            annotated, text,
            (30, 45 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2
        )

    writer.write(annotated)

    # Tiempo del ciclo completo
    end_total = time.perf_counter()

    total_ms = (end_total - start_total) * 1000

    # Excluir los primeros fotogramas
    if frame_idx >= WARMUP_FRAMES:
        track_times.append(track_ms)
        total_times.append(total_ms)


    frame_idx += 1

    if frame_idx % 30 == 0:
        ped_counter.prune(frame_idx)
        bike_counter.prune(frame_idx)
        vehicle_counter.prune(frame_idx)

cap.release()
writer.release()

print("\nRESULTADOS DEL CONTEO")

with open(
    EVENTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer_csv = csv.writer(file)

    writer_csv.writerow([
        "frame",
        "time_s",
        "class",
        "track_id",
        "direction"
    ])

    writer_csv.writerows(events)
    

for name, values in counts.items():
    print(f"{name}: {sum(values)}")


# Calcular las métricas de rendimiento
if not total_times:
    raise RuntimeError("No hay suficientes fotogramas para medir")

mean_track = float(np.mean(track_times))
mean_total = float(np.mean(total_times))
p95_total = float(np.percentile(total_times, 95))

effective_fps = 1000 / mean_total

# Mostrar resultados
print("\nRENDIMIENTO")
print(f"Tracker: {args.tracker}")
print(f"Dispositivo: {DEVICE}")
print(f"Fotogramas medidos: {len(total_times)}")
print(f"Tiempo medio tracking: {mean_track:.2f} ms")
print(f"Tiempo medio total: {mean_total:.2f} ms")
print(f"Latencia P95: {p95_total:.2f} ms")
print(f"FPS efectivos: {effective_fps:.2f}")

# Guardar resultados
summary = {
    "tracker": args.tracker,
    "model": "yolo11s.pt",
    "device": str(DEVICE),
    "imgsz": 960,
    "conf": 0.10,
    "source_fps": fps,
    "measured_frames": len(total_times),
    "mean_track_ms": mean_track,
    "mean_total_ms": mean_total,
    "p95_total_ms": p95_total,
    "effective_fps": effective_fps,
    "counts": counts
}

summary_path = f"outputs/summary_{args.tracker}.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print(f"Resultados guardados en {summary_path}")


print(f"\nVídeo guardado en {OUTPUT_PATH}")
