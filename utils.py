import pickle
from datetime import datetime

def load_encodings(path="encodings.pickle"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def mark_attendance(name, attendance_log, file_path="attendance.csv"):
    if name not in attendance_log:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(file_path, "a") as f:
            f.write(f"{name},{dt_string}\n")
        attendance_log.add(name)
        print(f"Attendance marked for {name} at {dt_string}")

def preprocess_frame(frame, scale=0.25):
    import cv2
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB
    return rgb_small_frame


import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime

# Load the known faces and embeddings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Initialize some variables
attendance_log = set()  # To keep track of already marked faces

def mark_attendance(name):
    with open("attendance.csv", "a") as f:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        if name not in attendance_log:
            f.write(f"{name},{dt_string}\n")
            attendance_log.add(name)
            print(f"Attendance marked for {name} at {dt_string}")

# Start video capture from webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert from BGR(OpenCV) to RGB(face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encoding with known encodings
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Unknown"

        # Use the closest known encoding if any match found
        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = data["names"][best_match_index]

        # Mark attendance
        mark_attendance(name)

        # Draw rectangle and label around face
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the resulting frame
    cv2.imshow("Face Recognition Attendance", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
