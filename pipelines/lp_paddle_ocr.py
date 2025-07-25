import cv2
import os
import json
import struct
import numpy as np
import Levenshtein
import csv
from tensorflow.lite.python.interpreter import Interpreter
from paddleocr import TextRecognition
from kafka import KafkaConsumer, KafkaProducer

# --- CONFIGURATION ---
import yaml

config_path = 'config.yaml'
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    BROKER = f"{cfg['RAW_BROKER']},{cfg['ANNOTATED_BROKER']}"
    RAW_TOPIC = cfg['RAW_TOPIC']
    ANNOTATED_TOPIC = cfg['ANNOTATED_TOPIC']
    METADATA_TOPIC = 'FrameMetadataTopic'

USE_KAFKA         = True   # True = live Kafka RAW_TOPIC, False = local video
SHOW_WINDOW       = False   # Display annotated stream window
VIDEO_SOURCE      = r'sample_videos/license_plates.mp4'
MODEL_PATH        = r'weights/detect.tflite'
COCO_NAMES        = r'weights/coco.names'
OUTPUT_VIDEO      = VIDEO_SOURCE + '_output.mp4'
CSV_OUTPUT        = VIDEO_SOURCE + '_output_data.csv'
PLATE_DICT_PATH   = r'plate_dict.txt'

MIN_CONFIDENCE       = 0.6
SIMILARITY_THRESHOLD = 0.6
FRAME_SKIP           = 4

# --- LOAD MODELS ---
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_details  = interpreter.get_input_details()
out_details  = interpreter.get_output_details()
IN_H, IN_W   = inp_details[0]['shape'][1:3]
FLOAT_INPUT  = (inp_details[0]['dtype'] == np.float32)

# COCO detection via SSD-MobileNet
net = cv2.dnn_DetectionModel(
    'weights/frozen_inference_graph.pb',
    'weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

with open(COCO_NAMES, 'rt') as f:
    classNames = f.read().splitlines()

ocr = TextRecognition(model_name="PP-OCRv5_server_rec")

# --- STATE ---
known_plates = {}
next_id      = 0

# --- HELPERS ---
def detect_objects(img, conf_t, nms_t, objects=('car',)):
    cids, confs, boxes = net.detect(img, confThreshold=conf_t, nmsThreshold=nms_t)
    if len(cids):
        return [
            (box, classNames[cid-1], float(conf))
            for cid, conf, box in zip(cids.flatten(), confs.flatten(), boxes)
            if classNames[cid-1] in objects
        ]
    return []

def paddle_read_plate(plate_roi):
    rgb = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB)
    results = ocr.predict(input=[rgb], batch_size=1)
    if not results:
        return "", 0.0
    res = results[0]
    text = getattr(res, 'rec_text', None) or res.json['res']['rec_text']
    conf = getattr(res, 'rec_score', None) or res.json['res']['rec_score']
    text = text.strip().replace(" ", "")
    return (text, conf) if conf > 0.70 else ("", 0.0)

def detect_license_plate(roi):
    img = cv2.resize(roi, (IN_W, IN_H))
    if FLOAT_INPUT:
        img = (img.astype(np.float32) - 127.5) / 127.5
    interpreter.set_tensor(inp_details[0]['index'], np.expand_dims(img, 0))
    interpreter.invoke()
    boxes  = interpreter.get_tensor(out_details[1]['index'])[0]
    scores = interpreter.get_tensor(out_details[0]['index'])[0]
    H, W   = roi.shape[:2]
    for i, score in enumerate(scores):
        if MIN_CONFIDENCE < score <= 1.0:
            y1 = int(max(0, boxes[i][0] * H)); x1 = int(max(0, boxes[i][1] * W))
            y2 = int(min(H, boxes[i][2] * H)); x2 = int(min(W, boxes[i][3] * W))
            crop = roi[y1:y2, x1:x2]
            text, _ = paddle_read_plate(crop)
            return text, (x1, y1, x2, y2)
    return "", None

# --- STREAM SETUP ---
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
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {VIDEO_SOURCE}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

# --- CSV SETUP ---
csv_file = open(CSV_OUTPUT, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Class','Frame','X1','Y1','X2','Y2','Car ID','Plate'])
frame_idx = 0

# --- MAIN LOOP ---
source = consumer if USE_KAFKA else iter(lambda: cap.read(), (False,None))
for msg in source:
    # read frame
    if USE_KAFKA:
        ts = struct.unpack('>d', msg.value[:8])[0]
        jpg = msg.value[8:]
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: continue
    else:
        ret, frame = msg
        if not ret: break
    frame_idx += 1
    annotated = frame.copy()

    if frame_idx % FRAME_SKIP == 0:
        objs = detect_objects(frame, 0.65, 0.2)
        for box, cname, _ in objs:
            x,y,fw,fh = box
            text, plate_box = detect_license_plate(frame[y:y+fh, x:x+fw])
            color=(0,255,0)
            if text:
                match_id, best = None, 1.0
                for kp, kid in known_plates.items():
                    d = Levenshtein.distance(text, kp)/max(len(text),len(kp))
                    if d<SIMILARITY_THRESHOLD and d<best:
                        best, match_id = d, kid
                if match_id is None:
                    match_id = next_id; known_plates[text]=next_id; next_id+=1;color=(255,0,0)
                cv2.rectangle(annotated,(x,y),(x+fw,y+fh),color,2)
                cv2.putText(annotated,f"Car {match_id}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,2)
                if plate_box:
                    x1,y1,x2,y2=plate_box
                    cv2.rectangle(annotated,(x+x1, y+y1),(x+x2, y+y2),(0,255,255),2)
                    cv2.putText(annotated,text,(x+x1,y+y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                csv_writer.writerow([cname,frame_idx,x,y,x+fw,y+fh,match_id,text])
            else:
                cv2.rectangle(annotated,(x,y),(x+fw,y+fh),(0,255,0),2)
                cv2.putText(annotated,'No Plate',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                csv_writer.writerow([cname,frame_idx,x,y,x+fw,y+fh,'N/A','No Plate'])

    # publish or save
    if USE_KAFKA:
        ok, buf = cv2.imencode('.jpg', annotated)
        if ok:
            payload = struct.pack('>d', ts) + buf.tobytes()
            producer.send(ANNOTATED_TOPIC, value=payload)
            meta = json.dumps({'frame': frame_idx, 'unix_ts': ts}).encode()
            producer.send(METADATA_TOPIC, value=meta)
    else:
        out.write(annotated)

    # display annotated
    if SHOW_WINDOW:
        cv2.imshow('Paddle Plate Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- CLEANUP ---
if USE_KAFKA:
    consumer.close(); producer.close()
else:
    cap.release(); out.release()
csv_file.close()
if SHOW_WINDOW:
    cv2.destroyAllWindows()
print('Done. Video and CSV output complete.')
