# Soccer — Detection · Tracking · Team Clustering · Roster IDs

Process a soccer video end-to-end and produce:
1) an annotated overlay video (team-colored open foot rings, ball pointer, team-local numbers 1–11), and  
2) a CSV with per-frame coordinates and metadata.


## Overview

- **Detection (YOLO):** player and ball classes
- **Tracking (ByteTrack/BOTSORT via Ultralytics `model.track`):** stable track IDs over time
- **Team Clustering (K=3):** jersey color features → TEAM1, TEAM2, REF
- **Roster IDs:** per-team numbering 1–11 (reuses the lowest free number on return)
- **Visualization:** open foot rings (gap on top) in team color, ball highlighted with a pointer
- **Export:** CSV with time, pixel coordinates (optionally normalized), role/team, number, motion stats


## Pipeline

Video
 └─ YOLO detection (players, ball)
    └─ Multi-object tracking (ByteTrack/BOTSORT): per-object track ID (tid)
       └─ Per-track jersey color feature update (EMA)
          └─ K=3 clustering (TEAM1 / TEAM2 / REF) with permutation protection
             └─ Roster numbering: 1–11 per team (reassign lowest free on return)
                ├─ Overlay (open rings + ball pointer + numbers)
                └─ CSV export (coordinates + metadata)


## Project Structure

├── detect.py                   # Main loop (I/O, tracking, clustering, rendering, CSV)
├── tracking/
│   ├── clustering.py           # TeamClusterer: jersey_patch, hsv_hist_feature, K=3 + prev-centers
│   ├── roster.py               # TeamRoster: stable team-local numbers, 1–11, reuse lowest free
│   ├── viz.py                  # draw_team_rings (open rings), ball pointer integration
│   └── utils.py                # tracker_cfg_path, draw_ball_pointer, mad_threshold (optional)
├── weights/
│   └── yolov8m_forzasys_soccer.pt   # Example weights (path is configurable)
├── data/
│   └── testvideo.mp4           # Input video
└── outputs/
    └── test_run.mp4            # Annotated result

## Theory

### Detection
A pretrained YOLO model detects **person** and **ball** classes on each processed frame.

### Tracking
Ultralytics `model.track` runs ByteTrack or BOTSORT to produce consistent **track IDs (tid)** over time, bridging brief misses.

### Team Clustering (Color-based)
- **Patch:** top fraction of the person box (jersey region).
- **Features:** HSV histograms with an explicit **grass mask**, L1-normalized; per-track **EMA** smoothing over time.
- **Clustering:** **K-Means (K=3)** → smallest cluster is **REF** (tie-break by mean S/V); remaining two are teams.
- **Permutation protection:** map current team clusters to TEAM1/TEAM2 by minimal distance to **previous centers**.

### Roster IDs (1–11 per team)
- Maintain `tid ↔ number` maps per team, considering **only visible** TEAM1/TEAM2 tracks each frame.
- Release numbers immediately when a track is not visible or leaves TEAM1/TEAM2.
- When a player returns, assign the **lowest free number** (1–11; then 12, 13, … if temporarily over capacity).

### Visualization
- **Open, thin foot rings** (gap on top), slightly desaturated for a clean look; placed at the foot anchor.
- **Ball pointer** (triangle above the ball, subtle circle at center); no ring for the ball.


## ⚙️ Installation

```bash
# Python 3.9–3.11 empfohlen
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy torch torchvision
# Optional
pip install scipy
```

## Quick Start

python detect.py \
  --source data/testvideo.mp4 \
  --out outputs/test_run.mp4 \
  --weights weights/yolov8m_forzasys_soccer.pt \
  --mark_ball \
  --show_ids

## Useful CLI Flags

| Flag | Description |
|---|---|
| `--imgsz 1280` | Inference resolution |
| `--conf 0.30`, `--iou 0.6` | Detection thresholds |
| `--target_fps 8.0` | Output FPS (video writer) |
| `--process_every_n 3` | Run inference every Nth frame (tracking fills in between) |
| `--cluster_every 10` | Re-cluster every N **processed** frames |
| `--detect "person,ball"` | Class filter (names or IDs) |
| `--tracker bytetrack` | Or `botsort` |
| `--tracker_cfg bytetrack.yaml` | Optional custom tracker YAML |
| `--save_csv csv/run1_tracks.csv` | Write enriched CSV |

## CSV-Output

frame,t_sec,tid,cls_id,class_name,conf,
x1,y1,x2,y2,cx,cy,foot_x,foot_y,w,h,area,aspect,
role,team_num,is_ball,
speed_px_s,dir_deg,trail_len

## Attribution / License

- Detector/Tracker via [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).  
- ByteTrack/BOTSORT as integrated in Ultralytics.
