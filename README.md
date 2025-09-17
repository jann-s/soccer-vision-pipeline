# Fußball – Detection · Tracking · Team-Clustering · Roster-IDs

Kurz: Dieses Projekt verarbeitet ein Fußball-Video und liefert
1) ein annotiertes Overlay-Video (Spieler-Ringe in Teamfarbe, Ball-Pfeil, Team-Nummern 1–11)  
2) eine CSV mit X/Y-Koordinaten und Metadaten je Frame.

---

## 🧭 Überblick

- **Detection** (YOLO): Personen + Ball
- **Tracking** (ByteTrack/BOTSORT via Ultralytics `model.track`): stabile Track-IDs über Zeit
- **Team-Clustering** (K=3): Farbfeatures aus Trikot-Patches → TEAM1, TEAM2, REF
- **Roster-IDs**: pro Team laufende Nummern 1–11 (kleinste freie Nummer wird wiederverwendet)
- **Visualisierung**: offene Fußringe (oben offen) in Teamfarbe, Ball mit Pfeil
- **Export**: CSV mit Zeit, Koordinaten (Pixel + normalisiert optional), Rolle/Team, Nummer, etc.

---

## 🔩 Pipeline (End-to-End)

Video
└─ YOLO Detection (Spieler, Ball)
└─ Multi-Object-Tracking (ByteTrack/BOTSORT): tid je Person/Ball
└─ Farbfeature-Update je Track (EMA)
└─ K=3 Clustering (Team1 / Team2 / Ref), Permutationsschutz
└─ Roster: Team-Nummern 1–11 (kleinste freie Nummer)
├─ Overlay (Ringe + Ballpfeil + Nummern)
└─ CSV Export (Koordinaten + Metadaten)


## 📁 Projektstruktur
├── detect.py # Main-Loop (I/O, Tracking, Clustering, Rendering, CSV)
├── tracking/
│ ├── clustering.py # TeamClusterer: jersey_patch, hsv_hist_feature, K=3 + prev-centers
│ ├── roster.py # TeamRoster: stabile Nummern pro Team, 1–11, kleinste freie
│ ├── viz.py # draw_team_rings (offene Ringe), draw_ball_pointer-Integration
│ └── utils.py # tracker_cfg_path, draw_ball_pointer, mad_threshold (optional)
├── weights/
│ └── yolov8m_forzasys_soccer.pt # Beispiel-Gewichte (Pfad frei wählbar)
├── data/
│ └── testvideo.mp4 # Eingabevideo
└── outputs/
└── test_run.mp4 # Annotiertes Ergebnis

---

## 🧠 Theorie (kurz & praxisnah)

### Detection
- Vortrainiertes YOLO-Modell (Fußball-tauglich): erkennt **Person** und **Ball**.

### Tracking
- Ultralytics `model.track` nutzt ByteTrack/BOTSORT → **tid** pro Objekt über Zeit (robust auch bei Lücken).

### Team-Clustering (Farbbasiert)
- Patch: oberer Anteil der Personenbox (Trikot).
- HSV-Features mit **Gras-Maske** (Grünanteile ausgeschlossen), Normierung, **EMA-Glättung** je Track.
- K=3 k-means: Drei Cluster → kleinste Gruppe = **REF** (mit Helligkeits-/Sättigungs-Tie-Break).
- **Permutationsschutz**: Zuordnung der beiden Team-Cluster zu TEAM1/TEAM2 via Distanz zu den **Zentren des letzten Laufs**.

### Roster-IDs (Teamnummern 1–11)
- Pro Team: `tid ↔ num` Maps. Nur **sichtbare** TEAM1/TEAM2-TIDs zählen.
- Nicht sichtbare/UNK/REF werden **sofort freigegeben**.
- Rückkehrer bekommen die **kleinste freie Nummer** (1–11, danach 12, 13, … bei Überbelegung).

### Visualisierung
- **Offene, dünne Fußringe** (oben Lücke), etwas entsättigt; Rahmen um die Beine.
- **Ball-Pfeil** (kein Ring beim Ball).

---

## ⚙️ Installation

```bash
# Python 3.9–3.11 empfohlen
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy torch torchvision
# Optional, falls benötigt:
# pip install scipy

python detect.py \
  --source data/testvideo.mp4 \
  --out outputs/test_run.mp4 \
  --weights weights/yolov8m_forzasys_soccer.pt \
  --mark_ball \
  --show_ids

  Nützliche Parameter:

--imgsz 1280 – Inferenzauflösung

--conf 0.30 / --iou 0.6 – Detektionsschwellen

--target_fps 8.0 – Ziel-FPS im Output (Writer)

--process_every_n 3 – nur jedes n-te Frame inferieren (Tracking füllt dazwischen)

--cluster_every 10 – alle N verarbeiteten Frames neu clustern

--detect "person,ball" – Klassenfilter (Namen oder IDs)

--tracker bytetrack – oder botsort

--tracker_cfg bytetrack.yaml – optional eigenes YAML

## CSV-Output

frame,t_sec,tid,cls_id,class_name,conf,
x1,y1,x2,y2,cx,cy,foot_x,foot_y,w,h,area,aspect,
role,team_num,is_ball,
speed_px_s,dir_deg,trail_len

Bedeutung:

    frame, t_sec – Frameindex und Timestamp [s] (aus FPS)

    tid – Tracker-ID, cls_id, class_name, conf – Detektionsinfos

    x1..y2 – BBox (px), cx,cy – Boxzentrum (px), foot_x,foot_y – Fußpunkt (px)

    w,h,area,aspect – Boxgeometrie

    role – {TEAM1, TEAM2, REF, UNK, BALL}

    team_num – 1–11 (nur TEAM1/TEAM2), sonst leer

    is_ball – 0/1

    speed_px_s – Geschwindigkeit am Fußpunkt [px/s]

    dir_deg – Richtung (0° rechts, 90° unten)

    trail_len – Länge der gezeichneten Spur