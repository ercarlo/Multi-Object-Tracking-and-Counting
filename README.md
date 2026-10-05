# Multi-Object Tracking & Traffic Counting

Language: [🇪🇸 Español](README_ES.md) | [🇬🇧 English](README.md)

Computer vision system for **detecting, tracking, and counting pedestrians, cars, and bicycles** in an urban video recorded with a fixed camera.

The project combines **YOLO11s** with four different tracking configurations and compares both their counting accuracy and computational performance.

The goal is not only to build an object counter, but also to study how detection losses, occlusions, trajectory fragmentation, and identity switches (*ID switches*) affect a multi-object tracking application.

## 🎬 Demo

https://github.com/user-attachments/assets/eea570a2-2624-4835-9389-3804011eca77

Comparison of **ByteTrack, BoT-SORT, BoT-SORT with Re-ID, and OC-SORT** running on the same traffic video.

## Features

- Detection of pedestrians, bicycles, and cars using a pretrained YOLO model.
- Object ID assignment and temporal tracking using ByteTrack, BoT-SORT, BoT-SORT with Re-ID, and OC-SORT.
- Independent counting by object class and crossing direction.
- Two virtual counting lines: one for pedestrians and bicycles, and another for cars.
- Export of annotated videos, crossing events to CSV, and performance metrics to JSON.
- Reproducible comparison using the same video, detector, resolution, and inference parameters for every tracker.

## Technologies and Architecture

**Python · Ultralytics/YOLO11s · OpenCV · PyTorch · NumPy**

```text
Video → Frames → YOLO11s → Tracker → ID-based trajectories
                                   ↓
                              Counting logic
                                   ↓
                 Annotated video + events.csv + summary.json
```

YOLO detects and classifies objects in each frame. The tracker associates detections between consecutive frames in order to maintain object identities over time.

Finally, an independent counting module converts trajectory side changes into crossing events.

### Common Experimental Configuration

| Parameter | Value |
|---|---|
| Detector | `yolo11s.pt`, pretrained on COCO |
| COCO classes | Person (`0`), bicycle (`1`), car (`2`) |
| Input size (`imgsz`) | `960` |
| Minimum confidence (`conf`) | `0.10` |
| Input video | Fixed camera, 30 FPS, approximately 36 s |
| Device used for performance measurements | CPU |
| Frames included in timing measurements | 1,072 (first 10 frames excluded) |

The pedestrian and bicycle counting line uses the **bottom-center point** of the detected bounding box as its reference point.

The car counting line uses the **center point** of the bounding box.

Tolerance bands, side confirmation, and minimum time intervals between events are used to reduce duplicated counts caused by small bounding-box oscillations.

Bicycles use an independent counter even though they share the pedestrian counting line.

The system counts **crossing events**, not unique real-world individuals across different sessions. Tracker IDs are local identifiers and may change when a trajectory is lost.

## Project Structure

```text
Traffic_Tracking/
├── main.py                 # Detection, tracking, counting, and metrics
├── compare.py              # Visual comparison of the four trackers
├── run_all.ps1             # Sequential execution of all experiments
├── bot_reid.yaml           # BoT-SORT configuration with Re-ID enabled
├── videos/
│   └── video.mp4           # Local test video
├── outputs/
│   ├── tracking_<tracker>.mp4
│   ├── events_<tracker>.csv
│   └── summary_<tracker>.json
├── README.md
└── README_ES.md
```

`compare.py` generates a synchronized visual comparison of the tracker outputs, while `run_all.ps1` automatically executes all tracking configurations sequentially.

Original video files should only be included in the repository if their license allows redistribution.

## Installation and Execution

On Windows, using the VS Code terminal or PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy torch
```

Place the input video at:

```text
videos/video.mp4
```

The tracking configuration files used during the experiments must also be available.

For BoT-SORT with Re-ID, `bot_reid.yaml` must have:

```yaml
with_reid: True
```

Run each tracker with:

```powershell
python main.py --tracker bytetrack
python main.py --tracker botsort
python main.py --tracker botsort_reid
python main.py --tracker ocsort
```

If the automation script is available, all experiments can be executed sequentially with:

```powershell
.\run_all.ps1
```

Each execution generates:

```text
tracking_<tracker>.mp4
events_<tracker>.csv
summary_<tracker>.json
```

The CSV file stores:

```text
frame
time_s
class
track_id
direction
```

while the JSON file contains the final counters and performance measurements.

## Counting Results

The ground-truth reference was obtained through manual inspection of the video:

- **21 pedestrians**
- **45 cars**
- **1 bicycle**

This corresponds to **67 crossing events in total**.

| Tracker | Pedestrians | Cars | Bicycles | Total | Absolute Error by Class |
|---|---:|---:|---:|---:|---:|
| **Manual reference** | **21** | **45** | **1** | **67** | — |
| ByteTrack | 22 | 45 | 0 | 67 | 2 |
| BoT-SORT | 21 | 45 | 1 | 67 | 0 |
| BoT-SORT + Re-ID | 21 | 45 | 1 | 67 | 0 |
| OC-SORT | 22 | 45 | 0 | 67 | 2 |

The absolute error by class is calculated as the sum of the absolute differences between the predicted and reference counts for the three object categories.

### Interpretation

- All four trackers correctly count **45 cars**, matching the manual reference.
- BoT-SORT, both with and without Re-ID, reproduces the correct count for every object category in this video.
- ByteTrack and OC-SORT also produce a total of **67 events**, but both register one extra pedestrian while failing to count the bicycle.
- Therefore, the correct total obtained by ByteTrack and OC-SORT hides two class-level errors.

During visual inspection, the bicycle case was related to the simultaneous detection of the cyclist as a person and the bicycle as a separate object, together with tracking and crossing-event conditions.

A single definitive cause has not yet been isolated. The error may involve ID continuity, crossing geometry, or the counting logic.

A correct final total **does not guarantee the absence of false events or missed crossings**.

To calculate event-level precision and recall, every predicted crossing would need to be matched against a manually annotated reference containing timestamp, class, and direction information.

## Performance Results

Performance measurements were carried out on **CPU**, using the same detector, video, input resolution, and detection confidence for every tracker.

The first 10 frames were excluded from the timing measurements to reduce initialization effects.

| Tracker | Detection + Tracking (ms/frame) | Full Loop (ms/frame) | Loop P95 (ms) | Effective FPS |
|---|---:|---:|---:|---:|
| **ByteTrack** | **162.53** | **184.23** | **194.30** | **5.43** |
| BoT-SORT | 191.18 | 212.85 | 224.56 | 4.70 |
| BoT-SORT + Re-ID | 205.78 | 227.62 | 239.17 | 4.39 |
| OC-SORT | 165.36 | 187.29 | 197.77 | 5.34 |

The **Detection + Tracking** measurement corresponds to the complete YOLO + tracker call. It should therefore not be interpreted as the isolated computational cost of the tracking association algorithm.

The **Full Loop** measurement additionally includes:

- Frame reading
- Detection and tracking
- Counting logic
- Drawing annotations
- Video writing

The **P95 latency** indicates the processing time below which approximately 95% of the measured frames were processed.

### Interpretation

- **ByteTrack is the fastest configuration**, achieving approximately **5.43 effective FPS**.
- **OC-SORT performs very similarly**, reaching approximately **5.34 FPS**.
- **BoT-SORT reaches 4.70 FPS**, with a higher computational cost in this configuration.
- Enabling **Re-ID** increases the average BoT-SORT full-loop processing time from **212.85 ms to 227.62 ms**, an increase of approximately **6.9%**.
- Re-ID also reduces effective performance from **4.70 FPS to 4.39 FPS**.
- In this particular video, enabling Re-ID does not change the final counting results.

None of the CPU configurations reaches the original video's 30 FPS.

Therefore, these experiments represent **offline video processing**, not a real-time deployment benchmark.

These measurements correspond to one execution per configuration. More robust performance conclusions would require repeated experiments under controlled system load and with exact software versions recorded.

## Tracking Analysis and Limitations

Visual inspection showed difficulties in maintaining stable IDs when pedestrians walk very close to each other or cross paths.

This is one of the main challenges of multi-object tracking.

The Re-ID configuration was included to evaluate whether visual appearance information could help resolve ambiguous associations.

However, the current results **do not quantitatively measure ID switches or trajectory fragmentation**.

The exported CSV files contain only **crossing events**, rather than the complete position of every tracked object in every frame.

For this reason, metrics such as:

- IDF1
- HOTA
- MOTA

cannot currently be calculated from the exported data.

Similarly, it would not be correct to claim that one tracker maintains object identities better based only on the final counting results.

Other limitations of the current prototype include:

- Evaluation using a single video.
- Fixed-camera scenario only.
- Possible dual interpretation of a cyclist as both a person and a bicycle.
- Sensitivity to counting-line geometry.
- Sensitivity to confirmation thresholds and tracker parameters.
- No global Re-ID across different cameras or different video sessions.

## Future Improvements

1. **Distinguish pedestrians from cyclists**

   Associate person and bicycle detections when they correspond to the same cyclist, while avoiding confusion with a person simply walking next to a bicycle.

2. **Improve line-crossing logic**

   Calculate the actual intersection between the object trajectory and the counting segment instead of relying only on point-side changes.

3. **Evaluate tracking quantitatively**

   Export full per-frame trajectories, manually annotate difficult occlusion cases, and count ID switches and trajectory fragmentation.

4. **Expand the evaluation dataset**

   Test the system on additional videos with different traffic densities, viewpoints, lighting conditions, and occlusion levels.

5. **Repeat performance measurements**

   Run multiple trials per tracker and analyze average execution time and variability.

6. **Evaluate the accuracy/speed trade-off**

   Compare different YOLO models, image resolutions, tracker configurations, and hardware acceleration options.

## Conclusions

In this scenario, **BoT-SORT achieves the best class-level counting results**, correctly detecting:

- 21 pedestrians
- 45 cars
- 1 bicycle

**ByteTrack is the fastest tracker**, while OC-SORT provides very similar computational performance.

Enabling Re-ID introduces additional computational cost without improving the final counting results in this specific experiment.

However, its potential benefit for maintaining object identity during difficult pedestrian interactions requires a dedicated tracking evaluation.

The main conclusion of this project is that **a correct final object count does not necessarily imply correct tracking**.

Detection quality, identity continuity, and counting logic must be evaluated separately in order to understand where errors originate and how the system can be improved.
