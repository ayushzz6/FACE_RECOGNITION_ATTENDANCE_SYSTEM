from flask import Flask, render_template, Response, request, redirect, url_for, session, send_file
import cv2
import face_recognition
import pickle
import csv
import datetime
import time
import os
import geocoder

app = Flask(__name__)                      # it is used to create a Flask application
app.secret_key = 'your_secret_key'         # it is used to encrypt the session data #session is used to store the user's login status 

# Load face encodings
with open('encodings.pickle', 'rb') as f:
    data = pickle.load(f)

camera = cv2.VideoCapture(0)
last_mark_time = {}

# Get approximate geolocation based on IP
def get_location():
    try:
        g = geocoder.ip('me')   # it is used to get the geolocation of the user using their IP address
        if g.ok:              # it is used to check if the geolocation was successful
            return f"{g.latlng[0]}, {g.latlng[1]}"   # it is used to return the latitude and longitude
    except:
        pass
    return "Unknown"

def mark_attendance(name, cooldown=60):
    now = time.time()         # it is used to get the current time 
    if name in last_mark_time and (now - last_mark_time[name]) < cooldown:
        return

    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.datetime.now().strftime('%H:%M:%S')
    filename = 'attendance.csv'
    location = get_location()

    try:
        with open(filename, 'r') as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        rows = []

    found = False        # it is used to check if the face has been marked
    for row in rows:      # it is used to iterate over the rows of the attendance data
        if len(row) >= 4 and row[0] == name and row[1] == date_str:     # it is used to check if the face has been marked
            found = True        # it is used to indicate that the face has been marked
            if row[3] == '':
                row[3] = time_str  # Out time
                row.append(location)
                with open(filename, 'w', newline='') as f_write:
                    writer = csv.writer(f_write)
                    writer.writerows(rows)
                last_mark_time[name] = now
                return

    if not found:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str, '', location])
        last_mark_time[name] = now

@app.route('/', methods=['GET', 'POST'])       # it is used to handle HTTP requests
def login():                                   # it is used to handle the login page
    if request.method == 'POST':             
        username = request.form['username']
        password = request.form['password']
        if password == '1234':
            session['username'] = username            # it is used to store the username in the session #a session is a dictionary that stores data that is specific to a single user
            return redirect(url_for('home'))           # it is used to redirect the user to the home page
        else:
            return render_template('index.html', error="Invalid password")         # it is used to display an error message
    return render_template('index.html')           # it is used to display the login page


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
    while True:          # it is used to capture video from the camera until the user closes the window
        success, frame = camera.read()       # it is used to capture a frame from the camera
        if not success:
            break
        frame = cv2.flip(frame, 1)        # it is used to flip the frame horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # it is used to convert the frame from BGR to RGB
        boxes = face_recognition.face_locations(rgb_frame)      # it is used to detect faces in the frame and return the bounding boxes
        encodings = face_recognition.face_encodings(rgb_frame, boxes)        # it is used to compute face encodings
        names = []

        for encoding in encodings:       # it is used to loop over the encodings and compare them with the encodings in the database
            matches = face_recognition.compare_faces(data['encodings'], encoding)     # it is used to compare the encodings with the encodings in the database
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]      # it is used to get the indices of the matches
                counts = {}     # it is used to store the counts of the names

                for i in matchedIdxs:      # it is used to loop over the matched indices
                    matched_name = data['names'][i]     # it is used to get the name of the person
                    counts[matched_name] = counts.get(matched_name, 0) + 1       # it is used to increment the count of the name

                name = max(counts, key=counts.get)       # it is used to get the name with the highest count

                if name == username:
                    mark_attendance(name)
                else:
                    name = "Not Authorized"

            names.append(name)


        for ((top, right, bottom, left), name) in zip(boxes, names):        # it is used to loop over the bounding boxes and names
            color = (0, 255, 0) if name == username else (0, 0, 255)         # it is used to set the color of the rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)      # it is used to draw the rectangle around the face
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)  # it is used to add the name to the rectangle

        ret, buffer = cv2.imencode('.jpg', frame)     # it is used to encode the frame as a JPEG to display in the browser 
        frame = buffer.tobytes()        # it is used to convert the frame to bytes #it is neceessary because the frame is a numpy array and it is used to display in thw browser


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')       # it is used to send the frame to the browser
        

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

if __name__ == '__main__':      # it is used to run the app if the file is being run as the main program
    app.run(debug=True)         # it is used to run the app in debug mode
