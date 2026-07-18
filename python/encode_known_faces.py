"""
Step 6a: bulk-enroll known people from a folder of reference photos.

Expected folder structure:
    known_faces/
        <any_folder_name>/
            photo1.jpg
            photo2.jpg

The folder name is just a convenient label while you're organizing photos —
it is NOT the person's permanent identity. Each folder gets a unique ID
generated at enrollment time, so two folders named "daniel" (two different
people) get two different IDs and are never confused with each other.

Run this once for bulk import, then use the live GUI's enroll popup for
anyone added afterward — both write to the same encodings.pkl/profiles.json.
"""

import json
import os
import pickle
import uuid

import face_recognition

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"
PROFILES_FILE = "profiles.json"


def generate_id():
    return uuid.uuid4().hex[:8]


def load_profiles():
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r") as f:
        return json.load(f)


def main():
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"'{KNOWN_FACES_DIR}/' not found. Create it with one subfolder per person, "
              f"each containing 1+ photos of that person's face.")
        return

    known_encodings = []
    known_ids = []
    profiles = load_profiles()

    for folder_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, folder_name)
        if not os.path.isdir(person_dir):
            continue

        person_id = generate_id()
        enrolled_any = False

        for filename in sorted(os.listdir(person_dir)):
            filepath = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                print(f"  No face found in {filepath}, skipping")
                continue
            if len(encodings) > 1:
                print(f"  Multiple faces found in {filepath}, using the first one")

            known_encodings.append(encodings[0])
            known_ids.append(person_id)
            enrolled_any = True
            print(f"Encoded: {folder_name} (id={person_id}) <- {filename}")

        if enrolled_any:
            profiles[person_id] = {"name": folder_name, "age": ""}

    if not known_encodings:
        print("No faces encoded. Check your known_faces/ folder structure.")
        return

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "ids": known_ids}, f)

    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\nSaved {len(known_encodings)} encodings for "
          f"{len(set(known_ids))} people to {ENCODINGS_FILE} and {PROFILES_FILE}")


if __name__ == "__main__":
    main()
