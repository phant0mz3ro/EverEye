"""
Search the tag database built by tag_recordings.py.

Usage:
    python search_tags.py car
    python search_tags.py person --camera front_door
"""

import argparse
import sqlite3

DB_FILE = "detections.db"


def format_timestamp(offset_sec):
    minutes, seconds = divmod(int(offset_sec), 60)
    return f"{minutes:02d}:{seconds:02d}"


def main():
    parser = argparse.ArgumentParser(description="Search tagged recordings")
    parser.add_argument("label", help="Object to search for, e.g. car, person, dog")
    parser.add_argument("--camera", help="Filter to a specific camera")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_FILE)
    query = "SELECT camera, video_file, timestamp_offset_sec, confidence FROM detections WHERE label = ?"
    params = [args.label]
    if args.camera:
        query += " AND camera = ?"
        params.append(args.camera)
    query += " ORDER BY video_file, timestamp_offset_sec"

    rows = conn.execute(query, params).fetchall()
    if not rows:
        print(f"No '{args.label}' detections found.")
        return

    print(f"Found {len(rows)} matches for '{args.label}':\n")
    for camera, video_file, offset, confidence in rows:
        print(f"  [{camera}] {video_file} @ {format_timestamp(offset)}  (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()
