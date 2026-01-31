import cv2
import os

name = input("Enter person name: ")
folder = f"../data/known_faces/{name}"

os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

print("Press 's' to save image | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f"{folder}/{count}.jpg", frame)
        print(f"Saved image {count}")
        count += 1

    if key == ord('q') or count == 20:
        break

cap.release()
cv2.destroyAllWindows()