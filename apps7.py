from flask import Flask, render_template, Response, request, jsonify
import cv2
import cvzone
from ultralytics import YOLO   
import math
import mysql.connector
from mysql.connector import Error
import threading
from enum import Enum

app = Flask(__name__)
# Load YOLO model
model = YOLO(r"C:\Users\ASUS\Intern - Copy\yolov8s-ppe.pt")
TELE_TOKEN = 'rep'
TELE_CHATID = 'rep'


# Initialize face detector and recognizer
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Dataset/training.xml')
modelhelm = YOLO(r"C:\Users\ASUS\Intern - Copy\helm.pt")

# Class names
classNames = ['Hardhat', 'NO-Hardhat', 'NO-Safety Vest', 'Safety Vest']
classNamesHelm = ['With Helmet', 'Without Helmet']

# Initialize video capture
cap = cv2.VideoCapture(0)
cap1= cv2.VideoCapture(1)
detector_active = False
detector_activeh = False

# Database connection
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='testai'
        )
        print("Koneksi ke MySQL berhasil")
    except Error as e:
        print(f"Terjadi kesalahan: {e}")
    return connection

db_connection = create_connection()
db_connection_chart = create_connection()
db_connection_helm = create_connection()
db_connection_chart_helm = create_connection()
def insert_data(connection, id, name, conf, class_name):
    try:
        cursor1 = connection.cursor()
        query = "INSERT INTO detection_face (id, name, confidence, timestamp, classes) VALUES (%s, %s, %s, NOW(), %s)"
        values = (id, name, conf, class_name)
        cursor1.execute(query, values)
        connection.commit()
        print("Data berhasil disimpan")
    except Error as e:
        print(f"Terjadi kesalahan saat menyimpan data: {e}")

def insert_data_helmet(connection, id, name, conf, class_name):
    try:
        cursor1 = connection.cursor()
        query = "INSERT INTO detection_helmet (id, name, confidence, timestamp, classes) VALUES (%s, %s, %s, NOW(), %s)"
        values = (id, name, conf, class_name)
        cursor1.execute(query, values)
        connection.commit()
        print("Data berhasil disimpan")
    except Error as e:
        print(f"Terjadi kesalahan saat menyimpan data: {e}")

def get_data_from_db():
    if db_connection:
        try:
            cursor = db_connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM detection_face ORDER BY timestamp DESC LIMIT 5")
            rows = cursor.fetchall()
            return rows
        except Error as e:
            print(f"Terjadi kesalahan saat mengambil data table: {e}")
    return []

def get_data_helm_from_db():
    if db_connection_helm:
        try:
            cursor = db_connection_helm.cursor(dictionary=True)
            cursor.execute("SELECT * FROM detection_helmet ORDER BY timestamp DESC LIMIT 5")
            rows = cursor.fetchall()
            return rows
        except Error as e:
            print(f"Terjadi kesalahan saat mengambil data table: {e}")
    return []

def get_data_chart():
   
    if db_connection_chart:
        try:
            cursor = db_connection_chart.cursor(dictionary=True)
            cursor.execute("SELECT * FROM detection_face ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return rows
        except Error as e:
            print(f"Terjadi kesalahan saat mengambil data chart: {e}")
    return []

def get_data_chart_helm():
   
    if db_connection_chart_helm:
        try:
            cursor = db_connection_chart_helm.cursor(dictionary=True)
            cursor.execute("SELECT * FROM detection_helmet ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return rows
        except Error as e:
            print(f"Terjadi kesalahan saat mengambil data chart: {e}")
    return []

import requests

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{'7162932731:AAGE0bfd0aQEs92Dwzh7P-XotDoX8IzZuzk'}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    response = requests.post(url, data=payload)
    return response.json()


def generate_frames():
    global detector_active
    while True:
        success, img = cap.read()
        if not success:
            print('failed to encode frame')
            continue

        if detector_active:
            abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
            id=None
            name='Unknown'
            for (x, y, w, h) in wajah:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, conf = recognizer.predict(abu[y:y + h, x:x + w])
                if id == 1:
                    name = 'Angga'
                elif id == 2:
                    name = 'Putri'
                elif id == 3:
                    name = 'Billy'
                elif id == 4:
                    name = 'Dimas'
                elif id == 5:
                    name = 'Angga H'
                else:
                    name = 'Unknown'
                cv2.putText(img, str(name), (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

            results = model(img, stream=True)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    classColorMap = {
                        'Hardhat' : (0,255,0),
                        'NO-Hardhat': (0,0,255),
                        'NO-Safety Vest': (0,0,255),
                        'Safety Vest': (0,255,0), 
                      
                
                    }

                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest'] :
                        # Set color based on class
                        cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=(0,0,255), colorT=(0, 0, 0), colorR=(0,0,255), offset=5)
                        cv2.rectangle(img, (x1, y1), (x2, y2), classColorMap[currentClass], 3)
                        
                        # Save detected data to the database if class is NO-Hardhat or NO-Safety Vest
                        if db_connection and id and currentClass in ['NO-Hardhat', 'NO-Safety Vest']:
                            insert_data(db_connection, id, name, conf, currentClass)
                            messageapd = f'Karyawan atas nama {name} tidak menggunakan apd'
                            send_telegram_message(TELE_TOKEN, TELE_CHATID, messageapd)              
                        else:
                            print("Error to insert Data")
                    elif currentClass in ['Hardhat', 'Safety Vest']:
                        cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=(0,255,0), colorT=(0, 0, 0), colorR=(0,255,0), offset=5)
                        cv2.rectangle(img, (x1, y1), (x2, y2), classColorMap[currentClass], 3)
                    else:
                        cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=(255,255,255), colorT=(0, 0, 0), colorR=(255,255,255), offset=5)
                        cv2.rectangle(img, (x1, y1), (x2, y2), classColorMap[currentClass], 3)
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            print('failed to encode frame from generate frame2')
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

def generate_frame2():
    global detector_activeh
    while True:
        success, img = cap1.read()
        if not success:
            continue

        if detector_activeh:
            abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
            id=None
            name='Unknown'
            for (x, y, w, h) in wajah:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, conf = recognizer.predict(abu[y:y + h, x:x + w])
                if id == 1:
                    name = 'Angga'
                elif id == 2:
                    name = 'Putri'
                elif id == 3:
                    name = 'Billy'
                elif id == 4:
                    name = 'Dimas'
                elif id == 5:
                    name = 'Angga H'
                else:
                    name = 'Tidak dikenal'
                cv2.putText(img, str(name), (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

            result1 = modelhelm(img, stream=True)
            for h in result1:
                boxesh = h.boxes
                for boxs in boxesh:
                    x3, y3, x4, y4 = boxs.xyxy[0]
                    x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
                    confh = math.ceil((boxs.conf[0] * 100)) / 100
                    clsh = int(boxs.cls[0])
                    currentClass = classNamesHelm[clsh]

                    classColorMaph = {
                       'With Helmet' : (0,255,0),
                       'Without Helmet' : (0,0,255)
                
                    }

                    if currentClass == 'Without Helmet' :
                        # Set color based on class
                        cvzone.putTextRect(img, f'{currentClass} {confh}', (max(0, x3), max(35, y3)), scale=1, thickness=1, colorB=(0,0,255), colorT=(0, 0, 0), colorR=(0,0,255), offset=5)
                        cv2.rectangle(img, (x3, y3), (x4, y4), classColorMaph[currentClass], 3)
                        
                        # Save detected data to the database if class is NO-Hardhat or NO-Safety Vest
                        if db_connection and currentClass == 'Without Helmet':
                            insert_data_helmet(db_connection, id, name, confh, currentClass)
                            send_telegram_message(TELE_TOKEN, TELE_CHATID, 'Terdapat Karyawan yang tidak mengenakan Helm')              
                        else:
                            print("Error to insert Data")
                    else:
                         cvzone.putTextRect(img, f'{currentClass} {confh}', (max(0, x3), max(35, y3)), scale=1, thickness=1, colorB=(0,255,0), colorT=(0, 0, 0), colorR=(0,255,0), offset=5)
                         cv2.rectangle(img, (x3, y3), (x4, y4), classColorMaph[currentClass], 3)
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            print('failed to encode frame from generate frame2')
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/camera_preview')
def camera_preview():
    return render_template('camera_preview.html')

@app.route('/helm_preview')
def helm_preview():
    return render_template('helm_preview.html')
# Lainnya tetap seperti sebelumnya


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frame2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detector_active
    detector_active = True
    return jsonify({'status': 'Detection started'})

@app.route('/start_detection1', methods=['POST'])
def start_detection1():
    global detector_activeh
    detector_activeh = True
    return jsonify({'status': 'Detection started'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detector_active
    detector_active = False
    return jsonify({'status': 'Detection stopped'})

@app.route('/stop_detection1', methods=['POST'])
def stop_detection1():
    global detector_activeh
    detector_activeh = False
    return jsonify({'status': 'Detection stopped'})

@app.route('/data')
def data():
    rows = get_data_from_db()
    return jsonify(rows)

@app.route('/datahelm')
def datahelm():
    rows = get_data_helm_from_db()
    return jsonify(rows)

@app.route('/datachart')
def datachart():
    rows1 = get_data_chart()
    return jsonify(rows1)

@app.route('/datacharthelm')
def datacharthelm():
    rows1 = get_data_chart_helm()
    return jsonify(rows1)

def start_video_thread():
    video_thread = threading.Thread(target=generate_frames)
    video_thread.daemon = True
    video_thread.start()

def start_video_thread2():
    video_thread = threading.Thread(target=generate_frame2)
    video_thread.daemon = True
    video_thread.start()

if __name__ == '__main__':
    start_video_thread()
    start_video_thread2()
    app.run(debug=True, use_reloader=False)  # Disable reloader to avoid issues with threading
