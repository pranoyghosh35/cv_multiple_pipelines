# pipelines/vehicle_classifications.py
import os
import sys
import struct
import yaml
import cv2
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from ultralytics import YOLO

# allow import of app.py one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import get_shared_data_client

# connect to the manager exported by app.py
shared_data = get_shared_data_client()

# --- CONFIGURATION ---
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

BROKER           = f"{cfg['RAW_BROKER']},{cfg['ANNOTATED_BROKER']}"
RAW_TOPIC        = cfg['RAW_TOPIC']
VIDEO_PATH       = 'sample_videos/3cars.mp4'
MODEL_NAME       = cfg['YOLO_MODEL']
YOLO_CONF_THRESH = 0.4
VEHICLE_CLASS_IDS= {2: 'Car', 5: 'Bus', 7: 'Truck'}
USE_KAFKA        = True
SHOW_WINDOW      = False
CSV_LOG          = MODEL_NAME.replace('.pt', '_detection_log.csv')
OUTPUT_FRAMES    = f"{MODEL_NAME}_output_frames"
os.makedirs(OUTPUT_FRAMES, exist_ok=True)

# --- LOAD MODEL ---
yolo = YOLO(MODEL_NAME)
yolo.to('cpu')

# --- PREPARE ---
ncsv_rows = []
frame_idx = 0

if USE_KAFKA:
    consumer = KafkaConsumer(
        RAW_TOPIC,
        bootstrap_servers=BROKER.split(','),
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda v: v
    )
    print("[KAFKA] Connected to RAW_TOPIC")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {VIDEO_PATH}")

# --- MAIN LOOP ---
source = consumer if USE_KAFKA else iter(lambda: cap.read(), (False, None))
for msg in source:
    if USE_KAFKA:
        ts_bytes = msg.value[:8]
        ts = struct.unpack('>d', ts_bytes)[0]
        jpg = msg.value[8:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype='uint8'), cv2.IMREAD_COLOR)
        if frame is None:
            continue
    else:
        ret, frame = msg
        if not ret:
            break
        ts = frame_idx

    frame_idx += 1
    annotated = frame.copy()

    results = yolo.track(frame, conf=YOLO_CONF_THRESH, tracker='bytetrack.yaml')[0]
    if hasattr(results.boxes, 'xyxy') and results.boxes.id is not None:
        boxes     = results.boxes.xyxy.cpu().numpy()
        confs     = results.boxes.conf.cpu().numpy()
        cls_ids   = results.boxes.cls.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cls_id = cls_ids[i]
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            tid   = track_ids[i]
            conf  = confs[i]
            label = results.names[cls_id]
            pt1   = (int(x1), int(y1))
            pt2   = (int(x2), int(y2))
            color = (255, 0, 0)

            cv2.rectangle(annotated, pt1, pt2, color, 2)
            cv2.putText(
                annotated,
                f"ID{tid}:{label} {conf:.2f}",
                (pt1[0], pt1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            ncsv_rows.append([frame_idx, tid, label, int(x1), int(y1), int(x2), int(y2)])

    # --- PUSH INTO SHARED MEMORY ---
    shared_data['frame']     = annotated
    shared_data['ts']        = ts
    shared_data['frame_idx'] = frame_idx
    print(f"[DEBUG] Pushed frame {frame_idx} to shared memory")

    if not USE_KAFKA:
        cv2.imwrite(os.path.join(OUTPUT_FRAMES, f"frame_{frame_idx}.jpg"), annotated)

    if SHOW_WINDOW:
        cv2.imshow('Vehicle Tracking', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- CLEANUP & LOGGING ---
if USE_KAFKA:
    consumer.close()
else:
    cap.release()
if SHOW_WINDOW:
    cv2.destroyAllWindows()

pd.DataFrame(ncsv_rows, columns=['Frame','TrackID','Class','X1','Y1','X2','Y2']) \
  .to_csv(CSV_LOG, index=False)
print(f"[CSV] Detections saved to {CSV_LOG}")
