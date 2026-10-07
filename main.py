import cv2
import os
import time
import math
import urllib.request
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# EXAMGUARD AI - PSI07
# AI-Based Examination Centre Behaviour Monitoring
# ============================================================


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

MODEL_PATH = "face_landmarker.task"

ABSENCE_LIMIT = 5.0
EYE_EVENT_LIMIT = 2.0
MULTIPLE_PERSON_ALERT_DELAY = 1.0

WINDOW_NAME = "ExamGuard AI - PSI07"


# ------------------------------------------------------------
# DOWNLOAD MEDIAPIPE MODEL IF NEEDED
# ------------------------------------------------------------

def download_model():
    if os.path.exists(MODEL_PATH):
        return

    print("\nDownloading MediaPipe Face Landmarker model...")
    print("Please wait...\n")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.\n")
    except Exception as e:
        print("ERROR: Could not download MediaPipe model.")
        print("Reason:", e)
        exit()


download_model()


# ------------------------------------------------------------
# MEDIAPIPE FACE LANDMARKER
# ------------------------------------------------------------

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

face_landmarker = vision.FaceLandmarker.create_from_options(options)


# ------------------------------------------------------------
# OPENCV FACE DETECTOR
# ------------------------------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Haar Cascade could not be loaded.")
    exit()


# ------------------------------------------------------------
# EVENT LOG
# ------------------------------------------------------------

event_log = []


def log_event(event_type, student, details):
    current_time = datetime.now().strftime("%H:%M:%S")

    event = {
        "time": current_time,
        "student": student,
        "event": event_type,
        "details": details
    }

    event_log.append(event)

    print("\n----------------------------------------")
    print("EVENT DETECTED")
    print("Time     :", current_time)
    print("Student  :", student)
    print("Event    :", event_type)
    print("Details  :", details)
    print("----------------------------------------")


# ------------------------------------------------------------
# INPUT SELECTION
# ------------------------------------------------------------

print("\n========================================")
print("        EXAMGUARD AI - PSI07")
print("========================================")
print()
print("1. Laptop Webcam")
print("2. Downloaded Video")
print("========================================")

choice = input("Select input (1/2): ").strip()


if choice == "1":

    cap = cv2.VideoCapture(0)
    input_type = "WEBCAM"

    if not cap.isOpened():
        print("\nERROR: Could not open laptop webcam.")
        print("Check whether another application is using the camera.")
        exit()

elif choice == "2":

    video_path = input("Enter video path: ").strip()

    # Remove quotes if user pasted a quoted Windows path
    video_path = video_path.strip('"')

    if not os.path.exists(video_path):
        print("\nERROR: Video file not found.")
        print("Path:", video_path)
        exit()

    cap = cv2.VideoCapture(video_path)
    input_type = "VIDEO"

    if not cap.isOpened():
        print("\nERROR: Could not open video.")
        exit()

else:

    print("\nInvalid choice.")
    exit()


# ------------------------------------------------------------
# VIDEO INFORMATION
# ------------------------------------------------------------

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or math.isnan(fps):
    fps = 30.0

frame_number = 0

frame_timestamp_ms = 0


# ------------------------------------------------------------
# STUDENT TRACKING VARIABLES
# ------------------------------------------------------------

student_id = "Student-1"

previous_center = None

tracking_distance_limit = 150


# ------------------------------------------------------------
# ABSENCE VARIABLES
# ------------------------------------------------------------

absence_start_time = None
absence_event_logged = False


# ------------------------------------------------------------
# MULTIPLE PERSON VARIABLES
# ------------------------------------------------------------

multiple_person_start = None
multiple_person_event_logged = False


# ------------------------------------------------------------
# EYE DIRECTION VARIABLES
# ------------------------------------------------------------

current_eye_direction = "CENTER"

eye_direction_start = None

eye_event_logged = False


# ------------------------------------------------------------
# FPS DISPLAY
# ------------------------------------------------------------

previous_frame_time = time.time()
display_fps = 0


# ------------------------------------------------------------
# EYE DIRECTION FUNCTION
# ------------------------------------------------------------

def calculate_eye_direction(landmarks):
    """
    Estimate eye/iris direction using MediaPipe iris landmarks.

    Returns:
        LEFT
        RIGHT
        UP
        DOWN
        CENTER
    """

    # MediaPipe Face Landmarker normally returns 478 landmarks.
    if len(landmarks) < 478:
        return "CENTER"

    try:

        # ----------------------------------------------------
        # LEFT EYE
        # ----------------------------------------------------

        left_corner_1 = landmarks[33]
        left_corner_2 = landmarks[133]

        left_top = landmarks[159]
        left_bottom = landmarks[145]

        # Left iris
        left_iris_points = [
            landmarks[468],
            landmarks[469],
            landmarks[470],
            landmarks[471],
            landmarks[472]
        ]

        left_iris_x = sum(p.x for p in left_iris_points) / 5
        left_iris_y = sum(p.y for p in left_iris_points) / 5

        left_min_x = min(left_corner_1.x, left_corner_2.x)
        left_max_x = max(left_corner_1.x, left_corner_2.x)

        left_min_y = min(left_top.y, left_bottom.y)
        left_max_y = max(left_top.y, left_bottom.y)

        left_width = left_max_x - left_min_x
        left_height = left_max_y - left_min_y

        if left_width == 0 or left_height == 0:
            return "CENTER"

        left_relative_x = (
            left_iris_x - left_min_x
        ) / left_width

        left_relative_y = (
            left_iris_y - left_min_y
        ) / left_height

        # ----------------------------------------------------
        # RIGHT EYE
        # ----------------------------------------------------

        right_corner_1 = landmarks[362]
        right_corner_2 = landmarks[263]

        right_top = landmarks[386]
        right_bottom = landmarks[374]

        # Right iris
        right_iris_points = [
            landmarks[473],
            landmarks[474],
            landmarks[475],
            landmarks[476],
            landmarks[477]
        ]

        right_iris_x = sum(p.x for p in right_iris_points) / 5
        right_iris_y = sum(p.y for p in right_iris_points) / 5

        right_min_x = min(right_corner_1.x, right_corner_2.x)
        right_max_x = max(right_corner_1.x, right_corner_2.x)

        right_min_y = min(right_top.y, right_bottom.y)
        right_max_y = max(right_top.y, right_bottom.y)

        right_width = right_max_x - right_min_x
        right_height = right_max_y - right_min_y

        if right_width == 0 or right_height == 0:
            return "CENTER"

        right_relative_x = (
            right_iris_x - right_min_x
        ) / right_width

        right_relative_y = (
            right_iris_y - right_min_y
        ) / right_height

        # ----------------------------------------------------
        # AVERAGE BOTH EYES
        # ----------------------------------------------------

        avg_x = (
            left_relative_x +
            right_relative_x
        ) / 2

        avg_y = (
            left_relative_y +
            right_relative_y
        ) / 2

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if avg_x < 0.35:
            return "LEFT"

        elif avg_x > 0.65:
            return "RIGHT"

        elif avg_y < 0.25:
            return "UP"

        elif avg_y > 0.75:
            return "DOWN"

        else:
            return "CENTER"

    except Exception:
        return "CENTER"


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # VIDEO TIMESTAMP
    # --------------------------------------------------------

    if input_type == "VIDEO":

        current_video_time = cap.get(
            cv2.CAP_PROP_POS_MSEC
        ) / 1000.0

    else:

        current_video_time = frame_number / fps

    frame_timestamp_ms = int(
        current_video_time * 1000
    )

    # Make sure timestamp always increases
    if frame_timestamp_ms <= 0:
        frame_timestamp_ms = int(
            frame_number * (1000 / fps)
        )

    # --------------------------------------------------------
    # FRAME INFORMATION
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # OPENCV FACE DETECTION
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    face_count = len(faces)

    # --------------------------------------------------------
    # MEDIAPIPE FACE LANDMARK DETECTION
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # IMPORTANT:
    # This is the corrected MediaPipe code.
    # Do NOT use python.core.image.Image.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = face_landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    # --------------------------------------------------------
    # DEFAULT STATUS
    # --------------------------------------------------------

    status_text = "NO STUDENT DETECTED"
    status_color = (0, 0, 255)

    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    if face_count == 0:

        status_text = "NO STUDENT DETECTED"
        status_color = (0, 0, 255)

        # Start absence timer
        if absence_start_time is None:
            absence_start_time = time.time()
            absence_event_logged = False

        absence_duration = (
            time.time() - absence_start_time
        )

        cv2.putText(
            frame,
            f"Absent for: {absence_duration:.1f}s",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # Log absence after 5 seconds
        if (
            absence_duration >= ABSENCE_LIMIT
            and not absence_event_logged
        ):

            log_event(
                "STUDENT ABSENT",
                student_id,
                f"No face detected for {absence_duration:.1f} seconds"
            )

            absence_event_logged = True

    # --------------------------------------------------------
    # ONE FACE
    # --------------------------------------------------------

    elif face_count == 1:

        # Reset absence
        absence_start_time = None
        absence_event_logged = False

        status_text = f"{student_id} PRESENT - NORMAL"
        status_color = (0, 255, 0)

        # ----------------------------------------------------
        # SIMPLE FACE TRACKING
        # ----------------------------------------------------

        x, y, w, h = faces[0]

        center_x = x + w // 2
        center_y = y + h // 2

        if previous_center is not None:

            distance = math.sqrt(
                (center_x - previous_center[0]) ** 2 +
                (center_y - previous_center[1]) ** 2
            )

            if distance > tracking_distance_limit:

                # Large movement
                cv2.putText(
                    frame,
                    "MOVEMENT DETECTED",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

        previous_center = (
            center_x,
            center_y
        )

        # ----------------------------------------------------
        # FACE BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            student_id,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # EYE DIRECTION
        # ----------------------------------------------------

        if result.face_landmarks:

            # Use first detected face
            landmarks = result.face_landmarks[0]

            eye_direction = calculate_eye_direction(
                landmarks
            )

            current_eye_direction = eye_direction

            cv2.putText(
                frame,
                f"EYES: {eye_direction}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # ------------------------------------------------
            # EYE DIRECTION TIMER
            # ------------------------------------------------

            if eye_direction != "CENTER":

                if eye_direction_start is None:

                    eye_direction_start = time.time()
                    eye_event_logged = False

                eye_duration = (
                    time.time() -
                    eye_direction_start
                )

                cv2.putText(
                    frame,
                    f"Eye direction duration: {eye_duration:.1f}s",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )

                # Log only after sustained movement
                if (
                    eye_duration >= EYE_EVENT_LIMIT
                    and not eye_event_logged
                ):

                    log_event(
                        "UNUSUAL EYE DIRECTION",
                        student_id,
                        f"Eyes looking {eye_direction} for "
                        f"{eye_duration:.1f} seconds"
                    )

                    eye_event_logged = True

            else:

                eye_direction_start = None
                eye_event_logged = False

    # --------------------------------------------------------
    # MULTIPLE PEOPLE
    # --------------------------------------------------------

    elif face_count > 1:

        status_text = "MULTIPLE PEOPLE"
        status_color = (0, 0, 255)

        # Reset absence
        absence_start_time = None
        absence_event_logged = False

        # ----------------------------------------------------
        # MULTIPLE PERSON TIMER
        # ----------------------------------------------------

        if multiple_person_start is None:

            multiple_person_start = time.time()
            multiple_person_event_logged = False

        multiple_duration = (
            time.time() -
            multiple_person_start
        )

        # ----------------------------------------------------
        # RED BOX AROUND ALL FACES
        # ----------------------------------------------------

        for i, (x, y, w, h) in enumerate(faces):

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"Person {i + 1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # ALERT
        # ----------------------------------------------------

        if (
            multiple_duration >=
            MULTIPLE_PERSON_ALERT_DELAY
        ):

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 55),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "WARNING! MULTIPLE PEOPLE DETECTED",
                (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            if not multiple_person_event_logged:

                log_event(
                    "MULTIPLE PEOPLE DETECTED",
                    student_id,
                    f"{face_count} people detected"
                )

                multiple_person_event_logged = True

    # --------------------------------------------------------
    # RESET MULTIPLE PERSON STATE
    # --------------------------------------------------------

    if face_count <= 1:

        multiple_person_start = None
        multiple_person_event_logged = False

    # --------------------------------------------------------
    # TOP STATUS BAR
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 55),
        (width, 100),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        status_text,
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2
    )

    # --------------------------------------------------------
    # INFORMATION PANEL
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Input: {input_type}",
        (20, height - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Faces Detected: {face_count}",
        (20, height - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Events: {len(event_log)}",
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (width - 180, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # VIDEO TIME
    # --------------------------------------------------------

    minutes = int(current_video_time // 60)
    seconds = int(current_video_time % 60)

    time_text = (
        f"Time: {minutes:02d}:{seconds:02d}"
    )

    cv2.putText(
        frame,
        time_text,
        (width - 180, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_frame_time = time.time()

    elapsed = (
        current_frame_time -
        previous_frame_time
    )

    if elapsed > 0:
        display_fps = 1 / elapsed

    previous_frame_time = current_frame_time

    cv2.putText(
        frame,
        f"FPS: {display_fps:.1f}",
        (width - 180, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

cap.release()

cv2.destroyAllWindows()

face_landmarker.close()


# ------------------------------------------------------------
# PRINT FINAL EVENT REPORT
# ------------------------------------------------------------

print("\n")
print("========================================")
print("       EXAMGUARD AI - FINAL REPORT")
print("========================================")

print(f"Input Type       : {input_type}")
print(f"Total Events     : {len(event_log)}")

print("----------------------------------------")

if len(event_log) == 0:

    print("No unusual events detected.")

else:

    for event in event_log:

        print(
            f"[{event['time']}] "
            f"{event['student']} | "
            f"{event['event']} | "
            f"{event['details']}"
        )

print("========================================")
print("Program finished.")