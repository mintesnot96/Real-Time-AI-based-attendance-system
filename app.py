import cv2
import os
import shutil
import pickle
import warnings
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# Suppress harmless scikit-image / InsightFace FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

from insightface.app import FaceAnalysis

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### 1. Initializing Classic Haar Cascade Face Detector
CASCADE_PATH = 'static/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

#### 2. Initializing Modern InsightFace Deep AI Engine
MODEL_DIR = 'static/insightface_models'
face_app = FaceAnalysis(name='buffalo_sc', root=MODEL_DIR)
face_app.prepare(ctx_id=-1, det_thresh=0.4, det_size=(320, 320))

KNN_MODEL_FILE = 'static/face_recognition_model.pkl'
EMBEDDINGS_FILE = 'static/face_embeddings.pkl'

#### Ensure required directories and attendance CSV exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')


def load_embeddings():
    """Loads all registered 512-D InsightFace embeddings from disk."""
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def save_embeddings(db):
    """Saves the InsightFace embeddings dictionary to disk."""
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(db, f)


def match_face_insight(live_embedding, db, threshold=0.45):
    """Computes cosine similarity between live face vector and registered vectors."""
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


def extract_faces_cascade(img):
    """Detects faces using classic Haar Cascade."""
    if img is None:
        return ()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.3, 5)


def train_both_models():
    """
    Synchronously trains/updates BOTH recognition models:
    1. Classic KNN Classifier on 50x50 raw pixel vectors (KNN_MODEL_FILE)
    2. Modern InsightFace 512-D Normalized Master Embeddings (EMBEDDINGS_FILE)
    """
    if not os.path.isdir('static/faces'):
        return

    knn_faces = []
    knn_labels = []
    insight_embeddings = {}

    userlist = os.listdir('static/faces')
    for user in userlist:
        userpath = os.path.join('static/faces', user)
        if not os.path.isdir(userpath):
            continue

        user_insight_embs = []
        for imgname in os.listdir(userpath):
            img = cv2.imread(os.path.join(userpath, imgname))
            if img is None:
                continue

            # 1. Feature extraction for Classic KNN (50x50 flattened vector)
            resized_50 = cv2.resize(img, (50, 50))
            knn_faces.append(resized_50.ravel())
            knn_labels.append(user)

            # 2. Feature extraction for InsightFace (512-D deep embedding)
            img_padded = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            faces = face_app.get(img_padded)
            if len(faces) > 0:
                emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
                user_insight_embs.append(emb)

        if user_insight_embs:
            master = np.mean(user_insight_embs, axis=0)
            master = master / np.linalg.norm(master)
            insight_embeddings[user] = master

    # Save InsightFace embeddings
    save_embeddings(insight_embeddings)

    # Train and save Classic KNN Classifier
    if len(knn_faces) > 0:
        knn = KNeighborsClassifier(n_neighbors=min(5, len(knn_faces)))
        knn.fit(np.array(knn_faces), knn_labels)
        joblib.dump(knn, KNN_MODEL_FILE)


# Ensure models are initialized if faces exist
if os.path.isdir('static/faces') and len(os.listdir('static/faces')) > 0:
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(KNN_MODEL_FILE):
        train_both_models()


#### Get number of total registered users
def totalreg():
    if not os.path.isdir('static/faces'):
        return 0
    return len([d for d in os.listdir('static/faces') if os.path.isdir(os.path.join('static/faces', d)) and len(os.listdir(os.path.join('static/faces', d))) > 0])


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


#### This function handles attendance with either InsightFace or Classic Haar+KNN
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()
    engine = request.args.get('engine', 'insightface').lower()

    cap = get_camera()
    if not cap.isOpened():
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Error: Could not access webcam. Please check your camera permissions or connection.')

    # -------------------------------------------------------------
    # OPTION A: CLASSIC HAAR CASCADE + KNN ENGINE
    # -------------------------------------------------------------
    if engine == 'cascade':
        if not os.path.exists(KNN_MODEL_FILE):
            cap.release()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Classic model not trained yet. Please add a student first.')

        try:
            knn_model = joblib.load(KNN_MODEL_FILE)
        except Exception as e:
            cap.release()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=f'Error loading KNN model: {str(e)}')

        win_title = 'Attendance [CLASSIC ENGINE: Haar Cascade + KNN]'
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            faces = extract_faces_cascade(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 2)
                try:
                    face_crop = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                    identified_person = knn_model.predict(face_crop.reshape(1, -1))[0]
                    add_attendance(identified_person)
                    display_name = identified_person.split('_')[0]
                    cv2.putText(frame, f"{display_name}", (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 128, 0), 2, cv2.LINE_AA)
                except Exception:
                    pass

            cv2.putText(frame, "Engine: Classic Haar + KNN | Press 'q' or ESC to exit", (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow(win_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                break

    # -------------------------------------------------------------
    # OPTION B: MODERN SOTA INSIGHTFACE AI ENGINE
    # -------------------------------------------------------------
    else:
        db = load_embeddings()
        if not db:
            cap.release()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='InsightFace embeddings empty. Please add a student first.')

        win_title = 'Attendance [MODERN SOTA: InsightFace ArcFace AI]'
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            faces = face_app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                live_emb = face.embedding
                identified_person, score = match_face_insight(live_emb, db, threshold=0.45)

                if identified_person:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    confidence = int(score * 100)
                    display_name = identified_person.split('_')[0]
                    label_text = f"{display_name} ({confidence}%)"
                    cv2.putText(frame, label_text, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    add_attendance(identified_person)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, "Engine: InsightFace AI (ArcFace) | Press 'q' or ESC to exit", (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(win_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
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
    TOTAL_SAMPLES = 20
    win_title = 'Enroll Student [Dual Engine Capture]'

    while i < TOTAL_SAMPLES:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1

        # Use InsightFace detector for high-precision face localization
        faces = face_app.get(frame)

        if len(faces) > 0:
            primary_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            bbox = primary_face.bbox.astype(int)
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(frame, f"Capturing Samples: {i+1}/{TOTAL_SAMPLES}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2, cv2.LINE_AA)

            if frame_count % 2 == 0 and (x2 > x1) and (y2 > y1):
                face_crop = frame[y1:y2, x1:x2]
                cv2.imwrite(f"{userimagefolder}/{newusername}_{i}.jpg", face_crop)
                i += 1
        else:
            # Fallback to Haar Cascade if SCRFD has high threshold
            cascade_faces = extract_faces_cascade(frame)
            if len(cascade_faces) > 0:
                (x, y, w, h) = cascade_faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, f"Capturing Samples: {i+1}/{TOTAL_SAMPLES}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                if frame_count % 2 == 0:
                    face_crop = frame[y:y+h, x:x+w]
                    cv2.imwrite(f"{userimagefolder}/{newusername}_{i}.jpg", face_crop)
                    i += 1
            else:
                cv2.putText(frame, f"Capturing: {i}/{TOTAL_SAMPLES}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "No face detected - look directly at camera", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Press 'q' or ESC to cancel", (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow(win_title, frame)

        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    saved_images = os.listdir(userimagefolder) if os.path.exists(userimagefolder) else []
    if len(saved_images) == 0:
        if os.path.exists(userimagefolder):
            os.rmdir(userimagefolder)
        mess = 'Student enrollment was cancelled.'
    else:
        # Synchronously train BOTH engines
        print('Training Both InsightFace and Haar+KNN Models...')
        train_both_models()
        mess = f"Student {newusername} (ID: {newuserid}) enrolled successfully in BOTH engines (InsightFace & Classic Haar)!"

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=mess)


#### This function will reset all face records and attendance to zero
@app.route('/clear', methods=['POST'])
def clear():
    # Delete all face image folders
    if os.path.isdir('static/faces'):
        for d in os.listdir('static/faces'):
            d_path = os.path.join('static/faces', d)
            if os.path.isdir(d_path):
                shutil.rmtree(d_path)

    # Delete embeddings and model files
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            os.remove(EMBEDDINGS_FILE)
        except Exception:
            pass

    if os.path.exists(KNN_MODEL_FILE):
        try:
            os.remove(KNN_MODEL_FILE)
        except Exception:
            pass

    # Reset today's attendance CSV with clean header
    filepath = f'Attendance/Attendance-{datetoday}.csv'
    with open(filepath, 'w') as f:
        f.write('Name,Roll,Time\n')

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='All face records and attendance logs have been completely cleared to zero for both engines!')


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)