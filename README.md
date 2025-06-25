face-attendance-project/
│
├── backend/
│   ├── app.py                # Backend server (Flask)
│   ├── encodings.pickle      # Generated face encodings file
│   ├── attendance.csv        # Attendance logs (runtime created)
│   └── utils.py              # Helper functions (face encoding, attendance etc.)
│
├── frontend/
│   ├── index.html            # Login page
│   ├── dashboard.html        # Webcam + attendance page
│   ├── styles.css            # Styling
│   └── script.js             # Frontend JS for webcam and API calls
│
├── model-training/
│   ├── encode_faces.py       # Face encoding generation script
│   ├── dataset/              # Folders per person containing images
│   │    ├── Aayush/
│   │    ├── Dipayan/
│   │    └── Prateek/
│   └── requirements.txt      # Dependencies for training
│
└── README.md                 # Project overview and instructions




# 📷 Face Recognition Attendance System 🧑‍💻

This is a **Full-Stack Face Recognition-based Attendance System** built using:

-  Python + Flask (Back-end)
-  OpenCV + face_recognition (Face detection & recognition)
-  Pre-trained face encodings (using `pickle`)
-  HTML/CSS (Front-end)
-  CSV-based storage for attendance logging

---

##  Features

✅ **Login System**  
Users must login with a username and a default password (`1234`).

✅ **Live Video Face Recognition**  
Once logged in, webcam feed starts and matches user’s face with the pre-trained encodings.

✅ **Attendance Marking**  
- First detection = **In Time**
- Second detection (same day) = **Out Time**
- Names not matching logged-in user are shown as "Not Authorized"

✅ **Attendance Log Viewer**  
Button to view `attendance.csv` in a clean table format directly from the home page.

✅ **Logout Functionality**  
Secure logout option for each user session.

---




