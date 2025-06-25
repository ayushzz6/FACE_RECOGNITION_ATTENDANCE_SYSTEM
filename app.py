from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import face_recognition
import pickle
import csv
import datetime
import time
import os
from flask import send_file 

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load encodings
with open('encodings.pickle', 'rb') as f:
    data = pickle.load(f)

camera = cv2.VideoCapture(0)
last_mark_time = {}

def mark_attendance(name, cooldown=60):
    now = time.time()
    if name in last_mark_time and (now - last_mark_time[name]) < cooldown:
        return

    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.datetime.now().strftime('%H:%M:%S')
    filename = 'attendance.csv'

    try:
        with open(filename, 'r') as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        rows = []

    found = False
    for row in rows:
        if len(row) >= 4 and row[0] == name and row[1] == date_str:
            found = True
            if row[3] == '':
                row[3] = time_str
                with open(filename, 'w', newline='') as f_write:
                    writer = csv.writer(f_write)
                    writer.writerows(rows)
                last_mark_time[name] = now
                return

    if not found:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str, ''])
        last_mark_time[name] = now

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if password == '1234':
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('index.html', error="Invalid password")
    return render_template('index.html')


@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    attendance = []
    if request.args.get('show_csv') == 'true':
        try:
            with open('attendance.csv', 'r') as f:
                reader = csv.reader(f)
                attendance = list(reader)
        except FileNotFoundError:
            attendance = []

    return render_template('home.html', username=session['username'], attendance=attendance)


def gen_frames(username):
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data['encodings'], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    matched_name = data['names'][i]
                    counts[matched_name] = counts.get(matched_name, 0) + 1

                name = max(counts, key=counts.get)

                if name == username:
                    mark_attendance(name)
                else:
                    name = "Not Authorized"

            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            color = (0, 255, 0) if name == username else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    return Response(gen_frames(username), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))



@app.route('/view_attendance')
def view_attendance():
    if 'username' not in session:
        return redirect(url_for('login'))
    return send_file('attendance.csv', mimetype='text/csv', as_attachment=False, download_name='attendance.csv')



if __name__ == '__main__':
    app.run(debug=True)

