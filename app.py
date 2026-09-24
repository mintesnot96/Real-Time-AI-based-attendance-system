import cv2
import os
import pickle
import warnings
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
import pandas as pd

# Suppress harmless scikit-image / InsightFace FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

from insightface.app import FaceAnalysis

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing InsightFace Engine (buffalo_sc: fast, mobile/edge optimized)
MODEL_DIR = 'static/insightface_models'
face_app = FaceAnalysis(name='buffalo_sc', root=MODEL_DIR)
face_app.prepare(ctx_id=-1, det_thresh=0.4, det_size=(320, 320))

EMBEDDINGS_FILE = 'static/face_embeddings.pkl'

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')


def load_embeddings():
    """Loads all registered 512-D face embeddings from disk."""
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def save_embeddings(db):
    """Saves the embeddings dictionary to disk."""
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(db, f)


def match_face(live_embedding, db, threshold=0.45):
    """Computes cosine similarity between live face vector and all registered vectors."""
    if not db or live_embedding is None:
        return None, 0.0
    norm_live = live_embedding / np.linalg.norm(live_embedding)
    scores = {}
    for user, master_emb in db.items():
        score = float(np.dot(norm_live, master_emb))
        scores[user] = score
    if not scores:
        return None, 0.0
    best_user = max(scores, key=scores.get)
    best_score = scores[best_user]
    if best_score >= threshold:
        return best_user, best_score
    return None, best_score


def train_model():
    """Builds/Updates master embeddings for all registered users in static/faces."""
    user_embeddings = {}
    if not os.path.isdir('static/faces'):
        return
    userlist = os.listdir('static/faces')
    for user in userlist:
        userpath = os.path.join('static/faces', user)
        if not os.path.isdir(userpath):
            continue
        embs = []
        for imgname in os.listdir(userpath)[:15]:
            img = cv2.imread(os.path.join(userpath, imgname))
            if img is None:
                continue
            # Pad image in case it's a tight crop from older Haar sessions
            img_padded = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            faces = face_app.get(img_padded)
            if len(faces) > 0:
                emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
                embs.append(emb)
        if embs:
            master = np.mean(embs, axis=0)
            master = master / np.linalg.norm(master)
            user_embeddings[user] = master
    save_embeddings(user_embeddings)
    # Maintain legacy pkl indicator for backward compatibility
    with open('static/face_recognition_model.pkl', 'w') as f:
        f.write('InsightFace Model Active\n')


# Auto-initialize embeddings if file is missing
if not os.path.exists(EMBEDDINGS_FILE):
    train_model()


#### Get number of total registered users
def totalreg():
    if not os.path.isdir('static/faces'):
        return 0
    return len([d for d in os.listdir('static/faces') if os.path.isdir(os.path.join('static/faces', d))])


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    filepath = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('Name,Roll,Time\n')
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return [], [], [], 0
    if len(df) == 0 or not all(col in df.columns for col in ['Name', 'Roll', 'Time']):
        return [], [], [], 0
    names = df['Name'].tolist()
    rolls = df['Roll'].tolist()
    times = df['Time'].tolist()
    l = len(df)
    return names, rolls, times, l


#### Add Attendance of a specific user
def add_attendance(name):
    if '_' not in name:
        return
    username, userid = name.rsplit('_', 1)
    current_time = datetime.now().strftime("%H:%M:%S")

    filepath = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            f.write('Name,Roll,Time\n')
    try:
        df = pd.read_csv(filepath)
        if 'Roll' in df.columns and str(userid) in [str(r) for r in df['Roll']]:
            return
    except Exception:
        pass

    need_newline = False
    try:
        with open(filepath, 'r') as f:
            text = f.read()
            if text and not text.endswith('\n'):
                need_newline = True
    except Exception:
        pass

    with open(filepath, 'a') as f:
        if need_newline:
            f.write('\n')
        f.write(f'{username},{userid},{current_time}\n')


def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()
    db = load_embeddings()
    if not db:
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There are no registered faces in the database. Please add a new face first.')

    cap = get_camera()
    if not cap.isOpened():
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Error: Could not access webcam. Please check your camera permissions or connection.')

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        faces = face_app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            live_emb = face.embedding
            identified_person, score = match_face(live_emb, db, threshold=0.45)

            if identified_person:
                # Green bounding box for recognized student
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                confidence = int(score * 100)
                display_name = identified_person.split('_')[0]
                label_text = f"{display_name} ({confidence}%)"
                cv2.putText(frame, label_text, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                add_attendance(identified_person)
            else:
                # Red bounding box for unknown/unregistered person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "InsightFace AI Engine - Press 'q' or ESC to finish", (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Attendance - InsightFace Engine', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if cv2.getWindowProperty('Attendance - InsightFace Engine', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername'].strip()
    newuserid = request.form['newuserid'].strip()
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = get_camera()
    if not cap.isOpened():
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Error: Could not access webcam. Please check your camera permissions or connection.')

    i = 0
    frame_count = 0
    collected_embs = []
    TOTAL_SAMPLES = 15

    while i < TOTAL_SAMPLES:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1
        faces = face_app.get(frame)

        if len(faces) > 0:
            primary_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            bbox = primary_face.bbox.astype(int)
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(frame, f"Capturing InsightFace Embeddings: {i+1}/{TOTAL_SAMPLES}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2, cv2.LINE_AA)

            if frame_count % 2 == 0 and (x2 > x1) and (y2 > y1):
                face_crop = frame[y1:y2, x1:x2]
                cv2.imwrite(f"{userimagefolder}/{newusername}_{i}.jpg", face_crop)
                emb = primary_face.embedding / np.linalg.norm(primary_face.embedding)
                collected_embs.append(emb)
                i += 1
        else:
            cv2.putText(frame, f"Capturing: {i}/{TOTAL_SAMPLES}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "No face detected - look directly at camera", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Press 'q' or ESC to cancel", (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow('Enroll Student - InsightFace', frame)

        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if cv2.getWindowProperty('Enroll Student - InsightFace', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    if len(collected_embs) == 0:
        if os.path.exists(userimagefolder):
            os.rmdir(userimagefolder)
        mess = 'User registration was cancelled.'
    else:
        # Calculate normalized master embedding and persist
        master_emb = np.mean(collected_embs, axis=0)
        master_emb = master_emb / np.linalg.norm(master_emb)
        db = load_embeddings()
        db[f"{newusername}_{newuserid}"] = master_emb
        save_embeddings(db)
        mess = f"Student {newusername} (ID: {newuserid}) enrolled successfully with InsightFace!"

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=mess)


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)