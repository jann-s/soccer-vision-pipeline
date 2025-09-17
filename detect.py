# detect.py
import cv2
import math
import argparse
from ultralytics import YOLO
import torch
from pathlib import Path
import csv
from collections import defaultdict, deque
import numpy as np

from tracking.utils import tracker_cfg_path
from tracking.clustering import TeamClusterer
from tracking.roster import TeamRoster
from tracking.viz import draw_team_rings  # optional: draw_team_rings

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/testvideo.mp4", help="Pfad zum Eingabevideo")
    ap.add_argument("--out", default="outputs/test_run.mp4", help="Ausgabedatei")
    ap.add_argument("--weights", default="weights/yolov8m_forzasys_soccer.pt",
                    help="Ultralytics Modellname oder Pfad zu lokalen Gewichten")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inferenz Auflösung")
    ap.add_argument("--conf", type=float, default=0.30, help="Konfidenzschwelle")
    ap.add_argument("--iou", type=float, default=0.6, help="IOU Schwelle NMS")
    ap.add_argument("--target_fps", type=float, default=8.0, help="Ziel Bildrate der Ausgabe")
    ap.add_argument("--preview_sec", type=int, default=10, help="Nur die ersten N Sekunden verarbeiten, 0 für alles")
    ap.add_argument("--process_every_n", type=int, default=3, help="Nur jedes nte Frame inferieren")
    ap.add_argument("--detect", default="", help="Kommagetrennte Klassenamen oder IDs. Leer bedeutet alle Klassen")
    ap.add_argument("--show", action="store_true", help="Fensteranzeige aktivieren")
    ap.add_argument("--max_frames", type=int, default=-1, help="Maximale Anzahl Frames, -1 für alle")
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"], help="Trackerwahl")
    ap.add_argument("--tracker_cfg", default="", help="Optional eigener YAML Pfad für Tracker Tuning")
    ap.add_argument("--save_csv", default="csv/final_version_tracking_clustering.csv", help="Optionaler Pfad für CSV Export. Leer deaktiviert.")
    ap.add_argument("--show_ids", action="store_true", help="IDs am Bild einblenden")
    ap.add_argument("--traj_len", type=int, default=20, help="Trajektorien Länge pro ID")
    ap.add_argument("--mark_ball", default=True, help="Ball speziell hervorheben")
    ap.add_argument("--cluster_every", type=int, default=10, help="Alle N verarbeiteten Frames neu clustern")
    ap.add_argument("--jersey_top_frac", type=float, default=0.45, help="Oberer Anteil der Box für Trikotfarbe")
    return ap.parse_args()

def _parse_detect_arg(arg_str, name_to_id):
    if not arg_str or not arg_str.strip():
        return None  # alle Klassen
    wanted = [s.strip().lower() for s in arg_str.split(",") if s.strip()]
    classes = []
    for w in wanted:
        if w.isdigit():
            classes.append(int(w))
        elif w in name_to_id:
            classes.append(name_to_id[w])
        else:
            print(f"[Warnung] Unbekannte Klasse '{w}' ignoriert.")
    return classes if classes else None

def main():
    args = parse_args()

    model = YOLO(args.weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_id_to_name = getattr(model.model, "names", getattr(model, "names", {})) 
    name_to_id = {str(v).lower(): int(k) for k, v in class_id_to_name.items()} if isinstance(class_id_to_name, dict) else {} 
    classes_arg = [0,1]
    print(model.model.names)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Quelle nicht öffnen: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if math.isclose(src_fps, 0.0):
        src_fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    process_stride = max(1, args.process_every_n)
    processed_fps = src_fps / process_stride
    eff_fps = min(args.target_fps, processed_fps)
    out = cv2.VideoWriter(args.out, fourcc, eff_fps, (w, h))
    write_stride_processed = max(1, round(processed_fps / eff_fps))

    max_preview_frames = int(src_fps * args.preview_sec) if args.preview_sec > 0 else float("inf")
    tracker_cfg = tracker_cfg_path(args.tracker, args.tracker_cfg)

    csv_writer, csv_f = None, None
    if args.save_csv and args.save_csv.strip():
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(args.save_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow([
            "frame","t_sec",
            "tid","cls_id","class_name","conf",
            "x1","y1","x2","y2","cx","cy","foot_x","foot_y","w","h","area","aspect",
            "role","team_num","is_ball",
            "speed_px_s","dir_deg","trail_len"
        ])

    track_history = defaultdict(lambda: deque(maxlen=args.traj_len))
    speed_state = {}  # tid -> {"pos": (x,y), "frame": int}
    ball_cls_id = name_to_id.get("ball", None)
    clusterer = TeamClusterer()
    roster = TeamRoster(max_players=11)

    print("Start")
    frame_idx = 0
    processed_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (args.max_frames >= 0 and frame_idx >= args.max_frames) or frame_idx >= max_preview_frames:
            break

        if frame_idx % process_stride == 0:
            print(frame_idx)
            results = model.track(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                classes=classes_arg,
                device=device,
                half=True if device == "cuda" else False,
                agnostic_nms=True,
                verbose=False,
                persist=True,
                tracker=tracker_cfg
            )

            annotated = frame
            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    ids  = boxes.id.detach().cpu().numpy() if boxes.id is not None else None
                    cls  = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else None
                    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None

                    # valid index (filtere NaN)
                    N = len(xyxy)
                    valid = list(range(N))
                    if ids is not None:
                        valid = [i for i in valid if not np.isnan(ids[i])]
                    if cls is not None:
                        valid = [i for i in valid if not np.isnan(cls[i])]

                    if valid:
                        xyxy = xyxy[valid]
                        ids  = ids[valid]  if ids  is not None else None
                        cls  = cls[valid]  if cls  is not None else None
                        conf = conf[valid] if conf is not None else None

                        # Mapping Track ID -> Klassenname
                        id_to_name = class_id_to_name if isinstance(class_id_to_name, dict) else {}
                        tid_to_clsname = {}
                        active_tids = []
                        for i in range(len(xyxy)):
                            tid_i = int(ids[i]) if ids is not None else -1
                            cls_i = int(cls[i]) if cls is not None else -1
                            cname = id_to_name.get(cls_i, str(cls_i))
                            if tid_i >= 0:
                                tid_to_clsname[tid_i] = cname
                                active_tids.append(tid_i)

                        # Features aktualisieren und clustern
                        clusterer.update_with_frame(frame, xyxy, ids, cls, id_to_name, args.jersey_top_frac)
                        if processed_idx % args.cluster_every == 0:
                            clusterer.cluster_and_assign(active_tids, tid_to_clsname)
                        clusterer.assign_balls_and_unknowns(active_tids, tid_to_clsname)
                        roles_by_tid = clusterer.roles

                        tid_to_teamnum = roster.update(frame_idx, clusterer.roles, active_tids)

                        # Zeichnen
                        # annotated = draw_tracks(
                        #     annotated.copy(),          # aktuelles Frame
                        #     xyxy,                      # Boxes
                        #     ids,                       # Track IDs
                        #     cls,                       # Klassen IDs
                        #     class_id_to_name,          # Namen
                        #     show_ids=args.show_ids,    # Textlabel ja/nein
                        #     history=track_history,     # Trails
                        #     mark_ball=args.mark_ball,  # Ball hervorheben
                        #     ball_cls_id=ball_cls_id,   # Klassen ID für Ball
                        #     clusterer=clusterer        # liefert Farben pro Track
                        # )
                        annotated = draw_team_rings(
                            annotated.copy(),
                            xyxy, ids, cls,
                            role_colors=clusterer.role_colors,
                            tid_to_teamnum=tid_to_teamnum,
                            ball_cls_id=ball_cls_id,
                            mark_ball=args.mark_ball,
                            include_unknown=True,
                            history=track_history
                        )
                        

                        # CSV
                        if csv_writer is not None:
                            for i in range(len(xyxy)):
                                cls_id = int(cls[i]) if cls is not None else -1
                                cname  = class_id_to_name.get(cls_id, str(cls_id)) if isinstance(class_id_to_name, dict) else str(cls_id)
                                tid    = int(ids[i]) if ids is not None else -1
                                cf     = float(conf[i]) if conf is not None else -1.0

                                x1, y1, x2, y2 = map(int, xyxy[i])
                                w_box = max(1, x2 - x1)
                                h_box = max(1, y2 - y1)
                                cx = x1 + w_box // 2
                                cy = y1 + h_box // 2
                                foot_x, foot_y = cx, y2

                                area   = w_box * h_box
                                aspect = w_box / float(h_box) if h_box > 0 else 0.0

                                # Rolleninfos
                                role = clusterer.roles.get(tid, {}).get("role", "UNK") if tid >= 0 else "UNK"
                                team_num = None
                                if tid >= 0 and tid_to_teamnum is not None and tid in tid_to_teamnum:
                                    _, team_num = tid_to_teamnum[tid]
                                is_ball = 1 if (cname.lower() == "ball" or role == "BALL") else 0

                                # Zeitstempel
                                t_sec = frame_idx / float(src_fps)

                                # Geschwindigkeit am Fußpunkt
                                speed_px_s, dir_deg = 0.0, 0.0
                                if tid >= 0:
                                    prev = speed_state.get(tid)
                                    if prev is not None:
                                        dx = foot_x - prev["pos"][0]
                                        dy = foot_y - prev["pos"][1]
                                        df = max(1, frame_idx - prev["frame"])
                                        speed_px_s = ( (dx*dx + dy*dy) ** 0.5 ) * (src_fps / df)
                                        # Richtung: 0 Grad nach rechts, 90 Grad nach unten
                                        dir_deg = math.degrees(math.atan2(dy, dx))
                                    speed_state[tid] = {"pos": (foot_x, foot_y), "frame": frame_idx}

                                # Traillänge, falls vorhanden
                                trail_len = len(track_history[tid]) if (track_history is not None and tid in track_history) else 0

                                csv_writer.writerow([
                                    frame_idx, f"{t_sec:.3f}",
                                    tid, cls_id, cname, f"{cf:.4f}",
                                    x1, y1, x2, y2, cx, cy, foot_x, foot_y, w_box, h_box, area, f"{aspect:.3f}",
                                    role, (team_num if team_num is not None else ""),
                                    is_ball,
                                    f"{speed_px_s:.2f}", f"{dir_deg:.1f}", trail_len
                                ])

                    # wenn keine validen Boxen, bleibt annotated das rohe frame

            # Schreiben nur für verarbeitete Frames, auf eff_fps ausgedünnt
            if (processed_idx % write_stride_processed) == 0:
                out.write(annotated)
                written += 1
                if args.show:
                    cv2.imshow("YOLO Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            processed_idx += 1

        frame_idx += 1

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    if csv_f is not None:
        csv_f.close()

    print(f"Fertig. Frames geschrieben: {written}. Ausgabe: {args.out}")
    if csv_writer is not None:
        print(f"CSV geschrieben: {args.save_csv}")

if __name__ == "__main__":
    main()