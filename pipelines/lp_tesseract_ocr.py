import cv2
import os
import json
import struct
import numpy as np
import pytesseract
from tensorflow.lite.python.interpreter import Interpreter
import Levenshtein
import csv
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
LABEL_PATH        = r'weights/labelmap.pbtxt'
COCO_NAMES        = r'weights/coco.names'
OUTPUT_VIDEO      = VIDEO_SOURCE + r'_output.mp4'
CSV_OUTPUT        = VIDEO_SOURCE + r'_output_data.csv'

CONFIG_PATH       = r'weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
WEIGHTS_PATH      = r'weights/frozen_inference_graph.pb'

MIN_CONFIDENCE       = 0.6
SIMILARITY_THRESHOLD = 0.6
FRAME_SKIP           = 4

# --- LOAD TFLITE & COCO DETECTION ---
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IN_H, IN_W      = input_details[0]['shape'][1:3]
FLOAT_INPUT     = (input_details[0]['dtype'] == np.float32)

net = cv2.dnn_DetectionModel(WEIGHTS_PATH, CONFIG_PATH)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

# --- LOAD LABELS ---
with open(COCO_NAMES, 'rt') as f:
    classNames = f.read().splitlines()

# --- STATE ---
known_license_plates = {}
next_id              = 0

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

def detect_license_plate(roi):
    img = cv2.resize(roi, (IN_W, IN_H))
    if FLOAT_INPUT:
        img = (img.astype(np.float32) - 127.5) / 127.5
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img,0))
    interpreter.invoke()
    boxes  = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    H, W = roi.shape[:2]
    for i, score in enumerate(scores):
        if MIN_CONFIDENCE < score <= 1.0:
            y1 = int(max(0, boxes[i][0]*H)); x1 = int(max(0, boxes[i][1]*W))
            y2 = int(min(H, boxes[i][2]*H)); x2 = int(min(W, boxes[i][3]*W))
            crop = roi[y1:y2, x1:x2]
            text = pytesseract.image_to_string(crop, config='--psm 8').strip()
            return text, (x1,y1,x2,y2)
    return '', None

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

# --- PROCESS LOOP ---
source = consumer if USE_KAFKA else iter(lambda: cap.read(), (False,None))
for msg in source:
    # get frame
    if USE_KAFKA:
        ts = struct.unpack('>d', msg.value[:8])[0]
        jpg = msg.value[8:]
        frame = cv2.imdecode(np.frombuffer(jpg,np.uint8), cv2.IMREAD_COLOR)
        if frame is None: continue
    else:
        ret, frame = msg
        if not ret: break
    frame_idx += 1
    annotated = frame.copy()
    if frame_idx % FRAME_SKIP == 0:
        objs = detect_objects(frame, 0.65, 0.2)
        for box, cname, conf in objs:
            x,y,wb,hb = box
            roi = frame[y:y+hb, x:x+wb]
            text, plate_box = detect_license_plate(roi)
            if text:
                match_id, best = None, 1.0
                for kt, kid in known_license_plates.items():
                    d = Levenshtein.distance(text, kt)/max(len(text),len(kt))
                    if d< SIMILARITY_THRESHOLD and d<best:
                        best, match_id = d, kid
                if match_id is None:
                    match_id = next_id; known_license_plates[text]=next_id; next_id+=1; color=(255,0,0)
                else:
                    color=(0,0,255)
                # draw
                cv2.rectangle(annotated,(x,y),(x+wb,y+hb),color,2)
                cv2.putText(annotated,f"Car {match_id}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,2)
                if plate_box:
                    x1,y1,x2,y2=plate_box
                    cv2.rectangle(annotated,(x+x1,y+y1),(x+x2,y+y2),(0,255,255),2)
                    cv2.putText(annotated,text,(x+x1,y+y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                csv_writer.writerow([cname,frame_idx,x,y,x+wb,y+hb,match_id,text])
            else:
                cv2.rectangle(annotated,(x,y),(x+wb,y+hb),(0,255,0),2)
                cv2.putText(annotated,'No Plate',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                csv_writer.writerow([cname,frame_idx,x,y,x+wb,y+hb,'N/A','No Plate'])
    # send or save
    if USE_KAFKA:
        ok,buf=cv2.imencode('.jpg',annotated)
        if ok:
            payload=struct.pack('>d',ts)+buf.tobytes();producer.send(ANNOTATED_TOPIC,value=payload)
            meta=json.dumps({'frame':frame_idx,'unix_ts':ts}).encode();producer.send(METADATA_TOPIC,value=meta)
    else:
        out.write(annotated)
    # display
    if SHOW_WINDOW:
        cv2.imshow('License Plate Detection',annotated)
        if cv2.waitKey(1)&0xFF==ord('q'): break

# --- CLEANUP ---
if USE_KAFKA:
    consumer.close();producer.close()
else:
    cap.release();out.release()
csv_file.close()
if SHOW_WINDOW: cv2.destroyAllWindows()
print('Done.')
