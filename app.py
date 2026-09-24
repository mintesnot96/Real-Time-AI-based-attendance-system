import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing face detector
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time\n')


#### get a number of total registered users
def totalreg():
    if not os.path.isdir('static/faces'):
        return 0
    return len([d for d in os.listdir('static/faces') if os.path.isdir(os.path.join('static/faces', d))])


#### extract the face from an image
def extract_faces(img):
    if img is None:
        return ()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        userpath = os.path.join('static/faces', user)
        if not os.path.isdir(userpath):
            continue
        for imgname in os.listdir(userpath):
            img = cv2.imread(os.path.join(userpath, imgname))
            if img is not None:
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
    if len(faces) > 0:
        faces = np.array(faces)
        n_neighbors = min(5, len(faces))
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')


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

    # Ensure newline before appending if file didn't end with one
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
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    names,rolls,times,l = extract_attendance()
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = get_camera()
    if not cap.isOpened():
        return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2,mess='Error: Could not access webcam. Please check your camera permissions or connection.')

    ret = True
    while ret:
        ret,frame = cap.read()
        if not ret or frame is None:
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x,y,w,h) = faces[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            add_attendance(identified_person)
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        
        cv2.putText(frame,"Press 'q' or ESC to finish",(30,frame.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow('Attendance',frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if cv2.getWindowProperty('Attendance', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = get_camera()
    if not cap.isOpened():
        names,rolls,times,l = extract_attendance()
        return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2,mess='Error: Could not access webcam. Please check your camera permissions or connection.')

    i = 0
    frame_count = 0
    while i < 50:
        ret,frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x,y,w,h) = max(faces, key=lambda b: b[2]*b[3])
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if frame_count % 2 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i += 1
        else:
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            cv2.putText(frame,'No face detected - look at the camera',(30,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame,"Press 'q' or ESC to cancel",(30,frame.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow('Adding new User',frame)
        
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if cv2.getWindowProperty('Adding new User', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    saved_images = os.listdir(userimagefolder) if os.path.exists(userimagefolder) else []
    if len(saved_images) == 0:
        if os.path.exists(userimagefolder):
            os.rmdir(userimagefolder)
        mess = 'User registration was cancelled.'
    else:
        print('Training Model')
        train_model()
        mess = f'Student {newusername} (ID: {newuserid}) added successfully!'

    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2,mess=mess) 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)