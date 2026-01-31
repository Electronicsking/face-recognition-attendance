import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os

# Load encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

attendance = []

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize for stability
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert to GRAYSCALE for detection (KEY FIX)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Face detection on GRAY image (very stable)
    locations = face_recognition.face_locations(gray, model="hog")

    # Convert to RGB ONLY for encoding
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb_small, locations)

    for encoding, loc in zip(encodings, locations):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            name = data["names"][matches.index(True)]
            time = datetime.now().strftime("%H:%M:%S")
            attendance.append([name, time])

        # Scale coordinates back
        top, right, bottom, left = loc
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

os.makedirs("../results", exist_ok=True)
df = pd.DataFrame(attendance, columns=["Name", "Time"])
df.to_csv("../results/attendance.csv", index=False)

print("Attendance saved successfully")
