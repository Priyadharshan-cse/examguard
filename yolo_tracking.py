import cv2
from ultralytics import YOLO


# ============================================================
# EXAMGUARD AI - YOLO OBJECT DETECTION + TRACKING
# PSI07
# ============================================================

print("=" * 60)
print("             EXAMGUARD AI - PSI07")
print("       YOLO OBJECT DETECTION + TRACKING")
print("=" * 60)

print("\nStarting YOLO model...")

# YOLO small model - suitable for laptop testing
model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully.\n")


# ============================================================
# OPEN LAPTOP WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # --------------------------------------------------------
    # YOLO DETECTION + BYTE TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Class ID
            class_id = int(
                box.cls[0].item()
            )

            # Object name
            object_name = model.names[class_id]

            # Confidence
            confidence = float(
                box.conf[0].item()
            )

            # Tracking ID
            if box.id is not None:
                track_id = int(
                    box.id[0].item()
                )
            else:
                track_id = -1

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"{object_name} "
                f"ID:{track_id} "
                f"{confidence * 100:.0f}%"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 55),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        "EXAMGUARD AI - OBJECT DETECTION + TRACKING",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        "ExamGuard AI - YOLO Tracking",
        frame
    )

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nExamGuard YOLO tracking stopped.")