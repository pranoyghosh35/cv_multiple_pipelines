from flask import Flask, render_template, request, jsonify, Response
import yaml, os, subprocess, signal, threading, time, sys
from kafka import KafkaConsumer

app = Flask(__name__)

# Load configuration
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# Kafka Config
raw_topic = cfg['RAW_TOPIC']
annotated_topic = cfg['ANNOTATED_TOPIC']
raw_broker = cfg['RAW_BROKER']
annotated_broker = cfg['ANNOTATED_BROKER']
pipeline_dir = cfg['PIPELINE_DIR']

# Process tracking
current_proc = None
proc_lock = threading.Lock()
log_lines = []
log_lock = threading.Lock()

def mjpeg_stream(topic, brokers):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=brokers.split(','),
        auto_offset_reset='latest',
        consumer_timeout_ms=1000
    )
    frame_count = 0
    try:
        for msg in consumer:
            frame_count += 1
            frame_size = len(msg.value)
            print(f"[KAFKA] Received frame #{frame_count} from topic '{topic}', size={frame_size} bytes")
            jpg = msg.value[8:]  # skip timestamp
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                jpg +
                b'\r\n'
            )
    finally:
        consumer.close()


@app.route('/mjpeg/raw')
def mjpeg_raw():
    return Response(mjpeg_stream(raw_topic, raw_broker), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mjpeg/annotated')
def mjpeg_annotated():
    return Response(mjpeg_stream(annotated_topic, annotated_broker), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scripts')
def list_scripts():
    scripts = [f for f in os.listdir(pipeline_dir) if f.endswith('.py')]
    return jsonify(sorted(scripts))

@app.route('/run', methods=['POST'])
def run_script():
    global current_proc
    import sys
    python_exec = sys.executable  # Ensure subprocess uses same venv

    name = request.json.get('name')
    path = os.path.join(pipeline_dir, name)

    print(f"[RUN] Attempting to launch: {python_exec} {path}")

    if not os.path.isfile(path):
        print(f"[ERROR] Script not found at path: {path}")
        return 'Script not found', 400

    with proc_lock:
        if current_proc and current_proc.poll() is None:
            print("[WARN] A script is already running.")
            return 'Already running', 400

        # Clear old logs
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
            print(f"[RUN] Subprocess started with PID {current_proc.pid}")

        except Exception as e:
            print(f"[ERROR] Failed to start subprocess: {e}")
            return 'Failed to start script', 500

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
                print(f"[RUN] Script exited with code {exit_code}")
            except Exception as e:
                print(f"[ERROR] While collecting logs: {e}")
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
        print(f"[STOP] Terminated script with PID {current_proc.pid}")
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
    print(f"[INIT] Flask running with Python: {sys.executable}")
    app.run(host='0.0.0.0', port=5000, threaded=True)
