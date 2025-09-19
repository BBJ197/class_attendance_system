from flask import Flask, render_template, Response, request, jsonify
from pymongo import MongoClient
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# ---------------------------
# Config
# ---------------------------
MONGO_URI = "mongodb+srv://beamlakbekele197_db_user:m5pG8eSXLFxM0Y9e@cluster0.mrqsfdq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "attendance_system"
THRESHOLD = 0.55
DEVICE_NAME = "Device-01"

# ---------------------------
# MongoDB
# ---------------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
students_col = db["students"]
attendance_col = db["attendance"]

# ---------------------------
# FaceAnalysis
# ---------------------------
app_face = FaceAnalysis(providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load known embeddings
# ---------------------------
def load_known_faces():
    known_embeddings = {}
    for doc in students_col.find():
        known_embeddings[doc["student_id"]] = {
            "name": doc["name"],
            "embedding": np.array(doc["embedding"])
        }
    print(f"[INFO] Loaded {len(known_embeddings)} users from MongoDB.")
    return known_embeddings

known_embeddings = load_known_faces()
current_name = "No Face"  # shared variable

def log_attendance(student_id, name, device=DEVICE_NAME):
    now = datetime.datetime.now().isoformat()
    attendance_col.insert_one({
        "student_id": student_id,
        "name": name,
        "timestamp": now,
        "device": device
    })
    print(f"[INFO] Attendance logged for {name} ({student_id}) at {now}")

# ---------------------------
# Video Stream
# ---------------------------
cap = cv2.VideoCapture(0)

def gen_frames():
    global current_name, known_embeddings
    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = app_face.get(frame)
        current_name = "Unknown"

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding.reshape(1, -1)

            for sid, data in known_embeddings.items():
                sim = cosine_similarity(emb, data["embedding"].reshape(1, -1))[0][0]
                if sim > THRESHOLD:
                    current_name = data["name"]
                    log_attendance(sid, data["name"], DEVICE_NAME)
                    break

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, current_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # send frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------
# Web Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize_status")
def recognize_status():
    """Send current recognition status"""
    return jsonify({"name": current_name})

@app.route("/register_unknown", methods=["POST"])
def register_unknown():
    global known_embeddings
    data = request.json
    student_id = data["student_id"]
    name = data["name"]

    # Capture one frame
    ret, frame = cap.read()
    faces = app_face.get(frame)
    if len(faces) == 0:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    embedding = faces[0].embedding.tolist()

    students_col.insert_one({
        "student_id": student_id,
        "name": name,
        "embedding": embedding
    })

    known_embeddings = load_known_faces()
    return jsonify({"status": "success", "message": f"{name} registered successfully!"})

# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
