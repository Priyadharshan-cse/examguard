import cv2
import os
import csv
import time
import math
import urllib.request
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


# ============================================================
# EXAMGUARD AI - PSI07
# BEHAVIOUR UNDERSTANDING ENGINE
# ============================================================

print("=" * 70)
print("                    EXAMGUARD AI")
print("             PSI07 BEHAVIOUR ENGINE")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "face_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

YOLO_MODEL = "yolo11n.pt"

ABSENCE_TIME = 5.0
MULTIPLE_PERSON_TIME = 1.0
PHONE_TIME = 1.5
EYE_EVENT_TIME = 2.0
MOVEMENT_TIME = 2.0

CSV_FILE = "examguard_events.csv"


# ============================================================
# DOWNLOAD MEDIAPIPE MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("\nDownloading MediaPipe Face Landmarker...")

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("MediaPipe model downloaded.")

    except Exception as e:

        print("Could not download model.")
        print(e)
        exit()


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded successfully.")


# ============================================================
# LOAD MEDIAPIPE
# ============================================================

print("Loading MediaPipe...")

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=2,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    options
)

print("MediaPipe loaded successfully.")


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nERROR: Webcam could not be opened.")
    exit()


# ============================================================
# DATA STRUCTURES
# ============================================================

tracked_people = {}

previous_people = set()

current_people = set()

event_log = []


# ============================================================
# EVENT TIMERS
# ============================================================

absence_start = None
absence_logged = False

multiple_start = None
multiple_logged = False

phone_start = {}
phone_logged = {}

eye_start = None
eye_logged = False

movement_start = {}
movement_logged = {}


# ============================================================
# EVENT LOGGER
# ============================================================

def log_event(
    event_type,
    track_id,
    details,
    timestamp
):

    event = {
        "time": timestamp,
        "track_id": track_id,
        "event": event_type,
        "details": details
    }

    event_log.append(event)

    print("\n" + "=" * 60)
    print("EVENT DETECTED")
    print("=" * 60)

    print("Time     :", timestamp)
    print("Track ID :", track_id)
    print("Event    :", event_type)
    print("Details  :", details)

    print("=" * 60)


# ============================================================
# CSV LOGGER
# ============================================================

def save_events():

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Time",
            "Track ID",
            "Event",
            "Details"
        ])

        for event in event_log:

            writer.writerow([
                event["time"],
                event["track_id"],
                event["event"],
                event["details"]
            ])


# ============================================================
# EYE DIRECTION
# ============================================================

def get_eye_direction(landmarks):

    if len(landmarks) < 478:
        return "CENTER"

    try:

        # ----------------------------------------------------
        # LEFT EYE
        # ----------------------------------------------------

        left_c1 = landmarks[33]
        left_c2 = landmarks[133]

        left_top = landmarks[159]
        left_bottom = landmarks[145]

        left_iris = [
            landmarks[468],
            landmarks[469],
            landmarks[470],
            landmarks[471],
            landmarks[472]
        ]

        left_iris_x = sum(
            p.x for p in left_iris
        ) / len(left_iris)

        left_iris_y = sum(
            p.y for p in left_iris
        ) / len(left_iris)

        left_min_x = min(
            left_c1.x,
            left_c2.x
        )

        left_max_x = max(
            left_c1.x,
            left_c2.x
        )

        left_min_y = min(
            left_top.y,
            left_bottom.y
        )

        left_max_y = max(
            left_top.y,
            left_bottom.y
        )

        left_width = left_max_x - left_min_x
        left_height = left_max_y - left_min_y

        if left_width == 0 or left_height == 0:
            return "CENTER"

        left_x = (
            left_iris_x - left_min_x
        ) / left_width

        left_y = (
            left_iris_y - left_min_y
        ) / left_height


        # ----------------------------------------------------
        # RIGHT EYE
        # ----------------------------------------------------

        right_c1 = landmarks[362]
        right_c2 = landmarks[263]

        right_top = landmarks[386]
        right_bottom = landmarks[374]

        right_iris = [
            landmarks[473],
            landmarks[474],
            landmarks[475],
            landmarks[476],
            landmarks[477]
        ]

        right_iris_x = sum(
            p.x for p in right_iris
        ) / len(right_iris)

        right_iris_y = sum(
            p.y for p in right_iris
        ) / len(right_iris)

        right_min_x = min(
            right_c1.x,
            right_c2.x
        )

        right_max_x = max(
            right_c1.x,
            right_c2.x
        )

        right_min_y = min(
            right_top.y,
            right_bottom.y
        )

        right_max_y = max(
            right_top.y,
            right_bottom.y
        )

        right_width = right_max_x - right_min_x
        right_height = right_max_y - right_min_y

        if right_width == 0 or right_height == 0:
            return "CENTER"

        right_x = (
            right_iris_x - right_min_x
        ) / right_width

        right_y = (
            right_iris_y - right_min_y
        ) / right_height


        # ----------------------------------------------------
        # AVERAGE
        # ----------------------------------------------------

        avg_x = (
            left_x +
            right_x
        ) / 2

        avg_y = (
            left_y +
            right_y
        ) / 2


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if avg_x < 0.35:
            return "LEFT"

        if avg_x > 0.65:
            return "RIGHT"

        if avg_y < 0.25:
            return "UP"

        if avg_y > 0.75:
            return "DOWN"

        return "CENTER"

    except Exception:

        return "CENTER"


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_number += 1

    height, width = frame.shape[:2]


    # ========================================================
    # TIMESTAMP
    # ========================================================

    video_time = frame_number / fps

    timestamp = (
        f"{int(video_time // 60):02d}:"
        f"{int(video_time % 60):02d}"
    )

    timestamp_ms = int(
        video_time * 1000
    )


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )


    people = []
    phones = []


    # ========================================================
    # PROCESS YOLO RESULTS
    # ========================================================

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            class_name = yolo.names[
                class_id
            ]

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            if box.id is not None:

                track_id = int(
                    box.id[0].item()
                )

            else:

                track_id = -1


            # =================================================
            # PERSON
            # =================================================

            if class_name == "person":

                people.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence
                })


            # =================================================
            # PHONE
            # =================================================

            elif class_name == "cell phone":

                phones.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence
                })


    # ========================================================
    # CURRENT TRACK IDs
    # ========================================================

    current_people = {
        person["id"]
        for person in people
        if person["id"] != -1
    }


    # ========================================================
    # PERSON ENTERED
    # ========================================================

    entered_people = (
        current_people -
        previous_people
    )

    for person_id in entered_people:

        log_event(
            "PERSON ENTERED",
            f"Person-{person_id}",
            "New person detected",
            timestamp
        )


    # ========================================================
    # PERSON LEFT
    # ========================================================

    left_people = (
        previous_people -
        current_people
    )

    for person_id in left_people:

        log_event(
            "PERSON LEFT",
            f"Person-{person_id}",
            "Previously tracked person disappeared",
            timestamp
        )


    previous_people = current_people.copy()


    # ========================================================
    # ABSENCE
    # ========================================================

    if len(people) == 0:

        if absence_start is None:

            absence_start = time.time()
            absence_logged = False


        duration = (
            time.time() -
            absence_start
        )


        if (
            duration >= ABSENCE_TIME
            and not absence_logged
        ):

            log_event(
                "STUDENT ABSENT",
                "N/A",
                f"No person detected for {duration:.1f} seconds",
                timestamp
            )

            absence_logged = True

    else:

        absence_start = None
        absence_logged = False


    # ========================================================
    # MULTIPLE PEOPLE
    # ========================================================

    if len(people) > 1:

        if multiple_start is None:

            multiple_start = time.time()
            multiple_logged = False


        duration = (
            time.time() -
            multiple_start
        )


        if (
            duration >= MULTIPLE_PERSON_TIME
            and not multiple_logged
        ):

            log_event(
                "MULTIPLE PEOPLE",
                "Multiple",
                f"{len(people)} people detected",
                timestamp
            )

            multiple_logged = True

    else:

        multiple_start = None
        multiple_logged = False


    # ========================================================
    # DRAW PEOPLE
    # ========================================================

    for person in people:

        person_id = person["id"]

        x1, y1, x2, y2 = person["box"]

        confidence = person["confidence"]


        # Green = normal
        box_color = (0, 255, 0)


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )


        label = (
            f"Person-{person_id} "
            f"{confidence * 100:.0f}%"
        )


        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )


    # ========================================================
    # PHONE DETECTION
    # ========================================================

    for phone in phones:

        phone_id = phone["id"]

        x1, y1, x2, y2 = phone["box"]

        confidence = phone["confidence"]


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )


        cv2.putText(
            frame,
            f"PHONE {confidence * 100:.0f}%",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # PHONE TIMER
        # ----------------------------------------------------

        if phone_id not in phone_start:

            phone_start[phone_id] = time.time()
            phone_logged[phone_id] = False


        phone_duration = (
            time.time() -
            phone_start[phone_id]
        )


        if (
            phone_duration >= PHONE_TIME
            and not phone_logged[phone_id]
        ):

            log_event(
                "PHONE DETECTED",
                f"Phone-{phone_id}",
                f"Phone visible for {phone_duration:.1f} seconds",
                timestamp
            )

            phone_logged[phone_id] = True


    # ========================================================
    # MEDIAPIPE EYE DETECTION
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    face_result = face_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    eye_direction = "CENTER"


    if face_result.face_landmarks:

        landmarks = face_result.face_landmarks[0]

        eye_direction = get_eye_direction(
            landmarks
        )


    # ========================================================
    # EYE BEHAVIOUR
    # ========================================================

    if len(people) >= 1:

        if eye_direction != "CENTER":

            if eye_start is None:

                eye_start = time.time()
                eye_logged = False


            eye_duration = (
                time.time() -
                eye_start
            )


            if (
                eye_duration >= EYE_EVENT_TIME
                and not eye_logged
            ):

                log_event(
                    "UNUSUAL EYE DIRECTION",
                    f"Person-{people[0]['id']}",
                    f"Eyes looking {eye_direction} "
                    f"for {eye_duration:.1f} seconds",
                    timestamp
                )

                eye_logged = True

        else:

            eye_start = None
            eye_logged = False

    else:

        eye_start = None
        eye_logged = False


    # ========================================================
    # MOVEMENT ANALYSIS
    # ========================================================

    for person in people:

        person_id = person["id"]

        x1, y1, x2, y2 = person["box"]

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2


        if person_id in tracked_people:

            old_x, old_y = tracked_people[
                person_id
            ]["center"]


            movement = math.sqrt(
                (center_x - old_x) ** 2 +
                (center_y - old_y) ** 2
            )


            # Significant frame-to-frame movement
            if movement > 20:

                if person_id not in movement_start:

                    movement_start[
                        person_id
                    ] = time.time()

                    movement_logged[
                        person_id
                    ] = False


                movement_duration = (
                    time.time()
                    -
                    movement_start[person_id]
                )


                if (
                    movement_duration >= MOVEMENT_TIME
                    and not movement_logged[person_id]
                ):

                    log_event(
                        "EXCESSIVE MOVEMENT",
                        f"Person-{person_id}",
                        f"Sustained movement detected",
                        timestamp
                    )

                    movement_logged[
                        person_id
                    ] = True

            else:

                movement_start.pop(
                    person_id,
                    None
                )

                movement_logged.pop(
                    person_id,
                    None
                )


        tracked_people[person_id] = {
            "center": (
                center_x,
                center_y
            )
        }


    # ========================================================
    # DETERMINE SYSTEM STATUS
    # ========================================================

    status = "NORMAL"
    status_color = (0, 255, 0)


    if len(people) == 0:

        status = "STUDENT ABSENT"
        status_color = (0, 0, 255)


    elif len(people) > 1:

        status = "MULTIPLE PEOPLE"
        status_color = (0, 0, 255)


    elif len(phones) > 0:

        status = "PHONE DETECTED"
        status_color = (0, 0, 255)


    elif eye_direction != "CENTER":

        status = "EYE MOVEMENT - REVIEW"
        status_color = (0, 165, 255)


    # ========================================================
    # HEADER
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 75),
        (25, 25, 25),
        -1
    )


    cv2.putText(
        frame,
        f"EXAMGUARD AI | {status}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.putText(
        frame,
        f"People: {len(people)}",
        (20, height - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Phones: {len(phones)}",
        (20, height - 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Eyes: {eye_direction}",
        (20, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Events: {len(event_log)}",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # TIMESTAMP
    # ========================================================

    cv2.putText(
        frame,
        f"TIME {timestamp}",
        (width - 180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # QUIT
    # ========================================================

    cv2.putText(
        frame,
        "Q = Quit",
        (width - 130, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "ExamGuard AI - Behaviour Engine",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

face_landmarker.close()


# ============================================================
# SAVE EVENTS
# ============================================================

save_events()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("                  EXAMGUARD FINAL REPORT")
print("=" * 70)

print(
    f"Total events detected: {len(event_log)}"
)

print(
    f"Event file saved to: {CSV_FILE}"
)

print("=" * 70)

for event in event_log:

    print(
        f"{event['time']} | "
        f"{event['track_id']} | "
        f"{event['event']} | "
        f"{event['details']}"
    )

print("=" * 70)