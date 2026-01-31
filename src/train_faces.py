import face_recognition
import os
import pickle
import numpy as np
from PIL import Image

DATA_DIR = "../data/known_faces"
ENCODING_FILE = "encodings.pkl"

known_encodings = []
known_names = []

print("🔹 Starting face encoding process...")

for person in os.listdir(DATA_DIR):
    person_path = os.path.join(DATA_DIR, person)

    if not os.path.isdir(person_path):
        continue

    print(f"➡️ Processing person: {person}")

    for image_name in os.listdir(person_path):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(person_path, image_name)

        try:
            # 🔹 Load image using PIL (MOST STABLE on Windows)
            pil_image = Image.open(image_path).convert("RGB")

            # 🔹 Convert to numpy uint8 array
            rgb = np.array(pil_image, dtype=np.uint8)

            # 🔹 Force contiguous memory (CRITICAL)
            rgb = np.ascontiguousarray(rgb)

            # 🔹 Detect faces
            locations = face_recognition.face_locations(rgb, model="hog")

            if len(locations) == 0:
                print(f"⚠️ No face found in {image_name}, skipping")
                continue

            # 🔹 Encode face
            encodings = face_recognition.face_encodings(rgb, locations)

            if len(encodings) == 0:
                print(f"⚠️ Encoding failed for {image_name}, skipping")
                continue

            known_encodings.append(encodings[0])
            known_names.append(person)

            print(f"✅ Encoded: {person}/{image_name}")

        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")

# Final validation
if len(known_encodings) == 0:
    print("❌ ERROR: No faces encoded. Check your images.")
    exit()

# Save encodings
with open(ENCODING_FILE, "wb") as f:
    pickle.dump(
        {"encodings": known_encodings, "names": known_names},
        f
    )

print(f"🎉 Training complete. Total faces encoded: {len(known_encodings)}")
print(f"📁 Encodings saved to {ENCODING_FILE}")
