Face Recognition Attendance System
-----------------------------------
  A real-time AI - based face recognition attendance system built using Python,OpenCV, and face_recognition(dlib).The system detects faces from a live camera feed, recognizes known people, and automatically logs their attendance into a CSV file
  Feauture:
  _________
  >>Real-time face detection usng webcam
  >>Face recognition with pre-trained encodings
  >>Automatic attendance logging into CSV
  >>Supports multiple users
  >>Fast and lightweight(HOG model)
  >>Ignores private data using .gitignore
Project Struture:
_________________
face-recognition-attendance/
│
├── src/
│ ├── collect_faces.py # Capture face images
│ ├── train_faces.py # Generate face encodings
│ ├── recognize_and_attend.py # Real-time recognition & attendance
│ └── test_camera.py # Camera test
│
├── data/
│ └── known_faces/ # Stored face images (ignored in GitHub)
│
├── results/
│ └── attendance.csv # Output file (ignored in GitHub)
│
├── requirements.txt
├── .gitignore
├── README.md
└── venv/ # Virtual environment (ignored)
Requirements:
_____________
Python 3.10
Webcam
Windows(tested)
Python Libraries:
_________________
opencv-python
face-recognition
numpy
pandas

How to Run(step by step)
________________________
1) Create virtual environment
2) Install dependencies
3) Collect face images
4) Train the model
5) Run face recognition

Privacy and Security
_____________________

The following are ignored:
venv/
data/known_faces/
results/attendance.csv

Future Improvements
____________________
Add date to attendance
Save face snapshots
Upload attendance on to google sheets
Use CNN model for higher accuracy
Build a GUI with Tkinter
 
----------------------
Author
Muhammed.T
Github:https://github.com/electronicsking
_______
