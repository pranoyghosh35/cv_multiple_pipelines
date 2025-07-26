#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, csv, struct, json, time, logging, numpy as np
from kafka import KafkaConsumer, KafkaProducer
from ultralytics import YOLO
import os, sys, yaml
from multiprocessing.managers import BaseManager

# ── SHARED DATA CLIENT IMPORT ────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sort.sort import Sort
from app import get_shared_data_client
shared_data = get_shared_data_client()

# ----------------------------------------------------------------------------
# 1.  LOAD CONFIG
# ----------------------------------------------------------------------------
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
RAW_TOPIC = cfg['RAW_TOPIC']
BROKER    = cfg['RAW_BROKER']
MODEL_PATH= cfg['MODEL_PATH']
LABEL_FILE= cfg['DIST_LABEL_FILE']

H_SCALE = cfg.get('H_SCALE', 0.009264)
W_SCALE = cfg.get('W_SCALE', 0.033218)
CONF_THRESH = cfg.get('CONF_THRESH', {
    'car':0.4,'person':0.3,'license-plate':0.6,
    'truck':0.6,'autorickshaw':0.8,'animal':1.0,'bus':1.0
})
SKIP_FRAMES = cfg.get('SKIP_FRAMES',2)
CSV_LOG     = cfg.get('CSV_LOG','object_distance_log.csv')

# ----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------
def load_labels(path):
    with open(path,'r') as f:
        return f.read().splitlines()

def compute_iou(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xB-xA+1)*max(0,yB-yA+1)
    areaA=(a[2]-a[0]+1)*(a[3]-a[1]+1)
    areaB=(b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter/(areaA+areaB-inter+1e-6)

def display_vehicle_counts(frame, counts, corner=(20,20), alpha=0.6, pad=10):
    # (identical to previous)
    pass

def corner_distance(b1,b2,h_scale,w_scale):
    c1=[(b1[0],b1[1]),(b1[2],b1[1]),(b1[0],b1[3]),(b1[2],b1[3])]
    c2=[(b2[0],b2[1]),(b2[2],b2[1]),(b2[0],b2[3]),(b2[2],b2[3])]
    best,dmin=None,float('inf')
    for p in c1:
        for q in c2:
            d=np.linalg.norm(np.subtract(p,q))
            if d<dmin: best,dmin=(p,q),d
    dx_px,dy_px=best[1][0]-best[0][0],best[1][1]-best[0][1]
    return (dx_px*w_scale,dy_px*h_scale,
            'right' if dx_px>0 else 'left' if dx_px<0 else 'aligned',
            'below' if dy_px>0 else 'above' if dy_px<0 else 'aligned',
            best)

# ----------------------------------------------------------------------------
# MAIN STREAMING PIPELINE
# ----------------------------------------------------------------------------
def stream_infer():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Initializing YOLO-SORT pipeline")

    labels = load_labels(LABEL_FILE)
    model  = YOLO(MODEL_PATH)
    logging.info(f"Loaded YOLO model '{MODEL_PATH}'")

    consumer = KafkaConsumer(
        RAW_TOPIC,
        bootstrap_servers=BROKER.split(','),
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda b: b
    )
    producer = KafkaProducer(bootstrap_servers=BROKER.split(','))

    tracker = Sort(max_age=39, min_hits=3, iou_threshold=0.1)
    vehicle_counts = {lab:0 for lab in labels}
    object_classes, class_counters = {}, {}
    frame_idx = 0

    csv_f = open(CSV_LOG,'w',newline='')
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        'Frame','Object1','Object2','Real_DX_m','Real_DY_m','H_Pos','V_Pos'
    ])
    csv_w.writeheader()

    for msg in consumer:
        frame_idx +=1
        if SKIP_FRAMES>1 and (frame_idx-1)%SKIP_FRAMES: continue

        ts = struct.unpack('>d',msg.value[:8])[0]
        frame = cv2.imdecode(np.frombuffer(msg.value[8:],np.uint8),cv2.IMREAD_COLOR)
        if frame is None: continue

        dets, det_info = [], []
        for box in model(frame)[0].boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            conf = float(box.conf[0]); cid=int(box.cls[0])
            lbl = labels[cid] if cid<len(labels) else 'Unknown'
            if conf>=CONF_THRESH.get(lbl,0.4):
                dets.append([x1,y1,x2,y2,conf])
                det_info.append({'bbox':[x1,y1,x2,y2],'class_name':lbl})
        dets_np = np.array(dets,dtype=float) if dets else np.empty((0,5))

        tracks = tracker.update(dets_np)
        for x1,y1,x2,y2,tid in tracks.astype(int):
            if tid not in object_classes:
                best_iou,best_lbl = 0,'Unknown'
                for d in det_info:
                    iou=compute_iou([x1,y1,x2,y2],d['bbox'])
                    if iou>best_iou: best_iou,best_lbl=iou,d['class_name']
                if best_iou>0.4:
                    class_counters.setdefault(best_lbl,0)
                    class_counters[best_lbl]+=1
                    object_classes[tid]={'class_name':best_lbl,'unique_id':class_counters[best_lbl]}
                    vehicle_counts[best_lbl]+=1
                else:
                    object_classes[tid]={'class_name':'Unknown','unique_id':0}
            info=object_classes[tid]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            txt=f"{info['class_name']} ID {info['unique_id']}"
            tw,th=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.9,2)[0]
            cv2.rectangle(frame,(x1,y1-th-10),(x1+tw,y1),(0,0,255),-1)
            cv2.putText(frame,txt,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        trk_list = tracks.astype(int)
        for i in range(len(trk_list)):
            for j in range(i+1,len(trk_list)):
                a,b=trk_list[i],trk_list[j]
                dx,dy,hpos,vpos,pair = corner_distance(a[:4],b[:4],H_SCALE,W_SCALE)
                csv_w.writerow({
                    'Frame':frame_idx,
                    'Object1':f"{object_classes[a[4]]['class_name']} ID {object_classes[a[4]]['unique_id']}",
                    'Object2':f"{object_classes[b[4]]['class_name']} ID {object_classes[b[4]]['unique_id']}",
                    'Real_DX_m':abs(dx),'Real_DY_m':abs(dy),'H_Pos':hpos,'V_Pos':vpos
                })
                csv_f.flush()
                cv2.line(frame,pair[0],pair[1],(255,0,0),2)
                mid=((pair[0][0]+pair[1][0])//2,(pair[0][1]+pair[1][1])//2)
                cv2.putText(frame,f"H:{abs(dx):.2f}m({hpos}) V:{abs(dy):.2f}m({vpos})",mid,cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

        display_vehicle_counts(frame,vehicle_counts)

        # ── ALWAYS PUSH TO SHARED MEMORY ────────────────────────────────────
        shared_data['frame']     = frame
        shared_data['ts']        = ts
        shared_data['frame_idx'] = frame_idx

        # ── PUBLISH ANNOTATED (if needed) ──────────────────────────────────
        # now optional: no longer requires configurations for annotated topic
        # if you wish to publish, uncomment below lines:
        # ok,buf=cv2.imencode('.jpg',frame)
        # if ok:
        #     producer.send('AnnotatedFramesTopic', value=struct.pack('>d',ts)+buf.tobytes())
        #     producer.send('FrameMetadataTopic', value=json.dumps({'frame':frame_idx,'unix_ts':ts}).encode())

        if cv2.waitKey(1)&0xFF==ord('q'): break

    consumer.close(); producer.close(); csv_f.close(); cv2.destroyAllWindows()
    logging.info("Pipeline ended")

if __name__ == '__main__':
    try:
        stream_infer()
    except KeyboardInterrupt:
        print("Interrupted, shutting down")