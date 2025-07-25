import cv2
import os
import json
import struct
import numpy as np
import pandas as pd
from ultralytics import YOLO
from kafka import KafkaConsumer, KafkaProducer

# --- CONFIGURATION ---
import yaml

config_path = 'config.yaml'
cfg={}
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
BROKER = f"{cfg['RAW_BROKER']},{cfg['ANNOTATED_BROKER']}"
RAW_TOPIC = cfg['RAW_TOPIC']
ANNOTATED_TOPIC = cfg['ANNOTATED_TOPIC']
METADATA_TOPIC = 'FrameMetadataTopic'

USE_KAFKA         = True   # True = live Kafka stream, False = local video file
SHOW_WINDOW       = False   # Toggle display of annotated stream window
VIDEO_PATH        = 'sample_videos/3cars.mp4'
MODEL_NAME        = 'weights/yolo11n.pt'
YOLO_CONF_THRESH  = 0.4
VEHICLE_CLASS_IDS = {2: 'Car', 5: 'Bus', 7: 'Truck'}
OUTPUT_FRAMES     = f"{MODEL_NAME}_output_frames"
os.makedirs(OUTPUT_FRAMES, exist_ok=True)
CSV_LOG           = MODEL_NAME.replace('.pt', '_detection_log.csv')

# --- LOAD MODEL ---
yolo = YOLO(MODEL_NAME)
yolo.to('cpu')

# --- PREPARE DATA STRUCTURES ---
ncsv_rows = []
frame_idx = 0

# --- SETUP STREAM SOURCE ---
if USE_KAFKA:
    consumer = KafkaConsumer(
        RAW_TOPIC,
        bootstrap_servers=BROKER.split(','),
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda v: v
    )
    producer = KafkaProducer(bootstrap_servers=BROKER.split(','))
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {VIDEO_PATH}")

# --- PROCESS LOOP ---
source = consumer if USE_KAFKA else iter(lambda: cap.read(), (False, None))
for msg in source:
    # Acquire frame
    if USE_KAFKA:
        ts_bytes = msg.value[:8]
        ts = struct.unpack('>d', ts_bytes)[0]
        jpg = msg.value[8:]
        frame = cv2.imdecode(
            np.frombuffer(jpg, dtype='uint8'),
            cv2.IMREAD_COLOR
        )
        if frame is None:
            continue
    else:
        ret, frame = msg
        if not ret:
            break

    frame_idx += 1
    # Work on copy for annotation
    annotated = frame.copy()

    # YOLO + ByteTrack inference
    results = yolo.track(frame, conf=YOLO_CONF_THRESH, tracker='bytetrack.yaml')[0]
    if hasattr(results.boxes, 'xyxy') and results.boxes.id is not None:
        boxes     = results.boxes.xyxy.cpu().numpy()
        confs     = results.boxes.conf.cpu().numpy()
        cls_ids   = results.boxes.cls.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        # Annotate & log detections
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

            # Draw box and label
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

            ncsv_rows.append([frame_idx, tid, label,
                              int(x1), int(y1), int(x2), int(y2)])

    # Always send annotated frame when USE_KAFKA
    if USE_KAFKA:
        ok, buf = cv2.imencode('.jpg', annotated)
        if ok:
            try:
                payload = struct.pack('>d', ts) + buf.tobytes()
                # Use key only if valid, else None
                msg_key = msg.key if msg.key and isinstance(msg.key, bytes) else None
                producer.send(ANNOTATED_TOPIC, key=msg_key, value=payload)

                meta = json.dumps({'frame': frame_idx, 'unix_ts': ts}).encode()
                producer.send(METADATA_TOPIC, key=msg_key, value=meta)

                producer.flush()
            except Exception as e:
                print(f"[ERROR] Kafka send failed at frame {frame_idx}: {e}")
    else:
        # Save locally
        out_path = os.path.join(OUTPUT_FRAMES, f"frame_{frame_idx}.jpg")
        cv2.imwrite(out_path, annotated)

    # Show only annotated stream
    if SHOW_WINDOW:
        cv2.imshow('Vehicle Tracking', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- CLEANUP ---
if USE_KAFKA:
    consumer.close()
    producer.close()
else:
    cap.release()

if SHOW_WINDOW:
    cv2.destroyAllWindows()

# --- WRITE CSV LOG ---
df = pd.DataFrame(ncsv_rows,
                  columns=['Frame','TrackID','Class','X1','Y1','X2','Y2'])
df.to_csv(CSV_LOG, index=False)
print(f"Detections saved to {CSV_LOG}")