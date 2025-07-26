# app.py
from flask import Flask, render_template, request, jsonify, Response
import yaml, os, subprocess, signal, threading, time, sys, cv2
from kafka import KafkaConsumer
from multiprocessing.managers import BaseManager

# --- SHARED DATA DICTIONARY (server-side) ---
_data = {'frame': None, 'ts': None, 'frame_idx': None}

# --- DEFINE THE MANAGER ---
class SharedManager(BaseManager):
    pass

# Register get_data, exposing dict methods for assignment
SharedManager.register(
    'get_data', callable=lambda: _data,
    exposed=['__getitem__','__setitem__','get','keys']
)

def start_manager_server(address=('0.0.0.0', 5001), authkey=b'sharedsecret'):
    manager = SharedManager(address=address, authkey=authkey)
    manager.start()
    print(f"[SHARED MANAGER] Serving at {address}")
    return manager

# Client to connect and retrieve the proxy dict
def get_shared_data_client(address='127.0.0.1', port=5001, authkey=b'sharedsecret'):
    SharedManager.register('get_data')
    mgr = SharedManager(address=(address, port), authkey=authkey)
    mgr.connect()
    print("[CLIENT] Connected to shared manager")
    return mgr.get_data()

# Placeholder for the shared proxy
shared_data = None

# --- FLASK SETUP ---
app = Flask(__name__)

# --- LOAD CONFIG ---
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
raw_topic    = cfg['RAW_TOPIC']
raw_broker   = cfg['RAW_BROKER']
pipeline_dir = cfg['PIPELINE_DIR']

@app.route('/mjpeg/raw')
def mjpeg_raw():
    def generate():
        consumer = KafkaConsumer(
            raw_topic,
            bootstrap_servers=raw_broker.split(','),
            auto_offset_reset='latest',
            consumer_timeout_ms=1000
        )
        try:
            for msg in consumer:
                jpg = msg.value[8:]
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                )
        finally:
            consumer.close()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mjpeg/annotated')
def mjpeg_annotated():
    def generate():
        while True:
            frame = shared_data.get('frame')
            if frame is not None:
                idx = shared_data.get('frame_idx')
                ok, jpeg = cv2.imencode('.jpg', frame)
                if ok:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                    )
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scripts')
def list_scripts():
    scripts = [f[:-3] for f in os.listdir(pipeline_dir) if f.endswith('.py') and "init" not in f]
    return jsonify(sorted(scripts))

current_proc = None
proc_lock = threading.Lock()
log_lines = []
log_lock = threading.Lock()

@app.route('/run', methods=['POST'])
def run_script():
    global current_proc
    python_exec = sys.executable
    name = request.json.get('name') + ".py"
    path = os.path.join(pipeline_dir, name)

    if not os.path.isfile(path):
        return 'Script not found', 400

    with proc_lock:
        if current_proc and current_proc.poll() is not None:
            pass
        if current_proc and current_proc.poll() is None:
            return 'Already running', 400
        with log_lock:
            log_lines.clear()
        try:
            current_proc = subprocess.Popen(
                [python_exec, "-u", path],
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy()
            )
        except Exception as e:
            return f'Failed to start script: {e}', 500
        def collect_logs():
            try:
                for line in iter(current_proc.stdout.readline, ''):
                    with log_lock:
                        log_lines.append(line.strip())
                    time.sleep(0.01)
                current_proc.stdout.close()
                exit_code = current_proc.wait()
                with log_lock:
                    log_lines.append(f"[EXIT] Script exited with code {exit_code}")
            except Exception as e:
                with log_lock:
                    log_lines.append(f"[LOG ERROR] {e}")
        threading.Thread(target=collect_logs, daemon=True).start()
    return 'Started', 200

@app.route('/stop', methods=['POST'])
def stop_script():
    global current_proc
    with proc_lock:
        if not current_proc or current_proc.poll() is not None:
            return 'No process', 400
        os.killpg(os.getpgid(current_proc.pid), signal.SIGTERM)
        current_proc = None
    return 'Stopped', 200

@app.route('/logs')
def stream_logs():
    def generate():
        last_idx = 0
        while True:
            time.sleep(0.5)
            with log_lock:
                if last_idx < len(log_lines):
                    for line in log_lines[last_idx:]:
                        yield f"data: {line}\n\n"
                    last_idx = len(log_lines)
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    mgr = start_manager_server()
    shared_data = mgr.get_data()
    app.run(host='0.0.0.0', port=5000, threaded=True)
