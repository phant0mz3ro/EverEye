"""
Step 6a: enroll known people.

Expected folder structure:
    known_faces/
        daniel/
            photo1.jpg
            photo2.jpg
        jane/
            photo1.jpg

One clean, front-facing photo per person is enough — more photos per
person (different angles/lighting) improves robustness but isn't required.

Run this once whenever you add/change known people, then re-run
stream_client.py to pick up the new encodings.
"""

import os
import pickle

import face_recognition

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"


def main():
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"'{KNOWN_FACES_DIR}/' not found. Create it with one subfolder per person, "
              f"each containing 1+ photos of that person's face.")
        return

    known_encodings = []
    known_names = []

    for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in sorted(os.listdir(person_dir)):
            filepath = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(filepath)

            # This enrollment step runs once and photos are curated (one clear
            # face expected), so the extra detection cost here is fine — unlike
            # the live pipeline where we reuse MediaPipe's box to avoid it.
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                print(f"  No face found in {filepath}, skipping")
                continue
            if len(encodings) > 1:
                print(f"  Multiple faces found in {filepath}, using the first one")

            known_encodings.append(encodings[0])
            known_names.append(person_name)
            print(f"Encoded: {person_name} <- {filename}")

    if not known_encodings:
        print("No faces encoded. Check your known_faces/ folder structure.")
        return

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"\nSaved {len(known_encodings)} encodings for "
          f"{len(set(known_names))} people to {ENCODINGS_FILE}")


if __name__ == "__main__":
    main()
