# ============================================================
# EXAMGUARD AI - Autonomous Vision & Behaviour Understanding
# HackNex 2026 - PSI07
#
# Detects potentially unusual behaviour for invigilator review.
# It does NOT prove cheating or identify a person's identity.
# ============================================================

import os
import csv
import time
import uuid
import math
import threading
import urllib.request
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="ExamGuard AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
CSV_FILE = "examguard_events_v3.csv"
FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
YOLO_MODEL_NAME = "yolo11n.pt"

ABSENT_SECONDS = 5.0
MULTIPLE_PERSON_SECONDS = 1.0
PHONE_SECONDS = 1.5
EYE_SECONDS = 2.0
MOVEMENT_SECONDS = 2.0
EYE_LEFT_LIMIT, EYE_RIGHT_LIMIT = 0.40, 0.60
EYE_UP_LIMIT, EYE_DOWN_LIMIT = 0.35, 0.65
MOVEMENT_THRESHOLD = 0.055
EVENT_COOLDOWN = 5.0
MAX_EVENT_HISTORY = 100

CSV_COLUMNS = ["Session ID", "Date", "Time", "Event", "Severity",
               "Entity", "Details", "Confidence"]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def ensure_face_model():
    if os.path.exists(FACE_MODEL_PATH):
        return True
    try:
        with st.spinner("Downloading face landmark model..."):
            urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        return True
    except Exception as e:
        st.error(f"Couldn't download the face landmark model: {e}")
        return False


def create_face_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5)
    return vision.FaceLandmarker.create_from_options(opts)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def mean_point(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


ensure_face_model()

# ============================================================
# VISION PROCESSOR (detection pipeline)
# ============================================================


class ExamGuardProcessor(VideoProcessorBase):

    def __init__(self):
        self.yolo = YOLO(YOLO_MODEL_NAME)
        self.face_landmarker = create_face_landmarker()
        self.session_id = (datetime.now().strftime("%Y%m%d_%H%M%S")
                           + "_" + uuid.uuid4().hex[:6])
        self.lock = threading.Lock()
        self.cfg = {k: globals()[k] for k in (
            'ABSENT_SECONDS', 'MULTIPLE_PERSON_SECONDS', 'PHONE_SECONDS',
            'EYE_SECONDS', 'MOVEMENT_SECONDS', 'EYE_LEFT_LIMIT',
            'EYE_RIGHT_LIMIT', 'EYE_UP_LIMIT', 'EYE_DOWN_LIMIT',
            'MOVEMENT_THRESHOLD')}
        self.start_time = time.time()
        self.risk_history = deque(maxlen=90)

        self.people_count = 0
        self.object_count = 0
        self.total_events = 0
        self.status = "NORMAL"
        self.risk_score = 0
        self.eye_direction = "CENTER"
        self.eye_x = 0.5
        self.eye_y = 0.5
        self.phone_detected = False
        self.movement_detected = False
        self.face_detected = False
        self.fps = 0
        self.detected_objects = []
        self.event_history = []

        self.previous_person_ids = set()
        self.previous_centers = {}
        self.absent_start = None
        self.multiple_start = None
        self.phone_start = None
        self.eye_start = None
        self.movement_start = None
        self.eye_x_history = deque(maxlen=7)
        self.eye_y_history = deque(maxlen=7)
        self.last_event_time = {}
        self.frame_counter = 0
        self.fps_start = time.time()

    # ---------------- events ----------------
    def add_event(self, event, severity="MEDIUM", entity="Student",
                  details="", confidence=""):
        now = time.time()
        if now - self.last_event_time.get(event, 0) < EVENT_COOLDOWN:
            return
        self.last_event_time[event] = now
        ts = datetime.now()
        row = {"Session ID": self.session_id, "Date": ts.strftime("%Y-%m-%d"),
               "Time": ts.strftime("%H:%M:%S"), "Event": event,
               "Severity": severity, "Entity": entity,
               "Details": details, "Confidence": confidence}
        with self.lock:
            self.event_history.append(row)
            if len(self.event_history) > MAX_EVENT_HISTORY:
                self.event_history.pop(0)
            self.total_events += 1
        try:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
        except Exception:
            pass

    # ---------------- eye direction ----------------
    def calculate_eye_direction(self, lm):
        try:
            def pt(i):
                return (lm[i].x, lm[i].y)

            l_iris = mean_point([pt(i) for i in range(468, 473)])
            r_iris = mean_point([pt(i) for i in range(473, 478)])

            l_w = max(abs(lm[133].x - lm[33].x), 0.001)
            r_w = max(abs(lm[263].x - lm[362].x), 0.001)
            l_h = max(abs(lm[145].y - lm[159].y), 0.001)
            r_h = max(abs(lm[374].y - lm[386].y), 0.001)

            lx = clamp((l_iris[0] - lm[33].x) / l_w, 0, 1)
            rx = clamp((r_iris[0] - lm[362].x) / r_w, 0, 1)
            ly = clamp((l_iris[1] - lm[159].y) / l_h, 0, 1)
            ry = clamp((r_iris[1] - lm[386].y) / r_h, 0, 1)

            self.eye_x_history.append((lx + rx) / 2)
            self.eye_y_history.append((ly + ry) / 2)
            sx = float(np.median(list(self.eye_x_history)))
            sy = float(np.median(list(self.eye_y_history)))

            if sy < self.cfg['EYE_UP_LIMIT']:
                d = "UP"
            elif sy > self.cfg['EYE_DOWN_LIMIT']:
                d = "DOWN"
            elif sx < self.cfg['EYE_LEFT_LIMIT']:
                d = "LEFT"
            elif sx > self.cfg['EYE_RIGHT_LIMIT']:
                d = "RIGHT"
            else:
                d = "CENTER"
            return d, sx, sy
        except Exception:
            return "CENTER", 0.5, 0.5

    def draw_iris(self, frame, lm):
        h, w = frame.shape[:2]
        for i in range(468, 478):
            if i < len(lm):
                cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)),
                           3, (0, 255, 255), -1)

    def calculate_risk_score(self):
        s = 0
        if self.people_count > 1:
            s += 35
        if self.phone_detected:
            s += 30
        if self.eye_direction not in ("CENTER",):
            s += 15
        if self.movement_detected:
            s += 15
        if self.people_count == 0:
            s += 20
        return int(clamp(s, 0, 100))

    def determine_status(self):
        if self.people_count == 0:
            return "STUDENT ABSENT"
        if self.people_count > 1:
            return "MULTIPLE PEOPLE"
        if self.phone_detected:
            return "PHONE DETECTED"
        if self.eye_direction != "CENTER":
            return "EYE DEVIATION"
        if self.movement_detected:
            return "UNUSUAL MOVEMENT"
        return "NORMAL"

    # ---------------- per frame ----------------
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        display = img.copy()

        try:
            result = self.yolo.track(img, persist=True, tracker="bytetrack.yaml",
                                     verbose=False, conf=0.35)[0]
            current_ids, centers, names = set(), {}, []
            people = phones = 0

            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    cname = self.yolo.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                    if cname == "person":
                        people += 1
                        tid = None
                        if boxes.id is not None:
                            try:
                                tid = int(boxes.id[i].item())
                            except Exception:
                                tid = None
                        if tid is not None:
                            current_ids.add(tid)
                            centers[tid] = ((x1 + x2) / 2, (y1 + y2) / 2)
                            label = f"Person ID {tid} {conf:.0%}"
                        else:
                            label = f"Person {conf:.0%}"
                        cv2.rectangle(display, (x1, y1), (x2, y2), (94, 212, 45), 2)
                        cv2.putText(display, label, (x1, max(25, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (94, 212, 45), 2)
                    elif cname == "cell phone":
                        phones += 1
                        names.append("cell phone")
                        cv2.rectangle(display, (x1, y1), (x2, y2), (60, 60, 240), 2)
                        cv2.putText(display, f"PHONE {conf:.0%}", (x1, max(25, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 240), 2)
                    else:
                        names.append(cname)

            self.people_count = people
            self.object_count = len(names)
            self.detected_objects = sorted(set(names))
            self.phone_detected = phones > 0

            # entry / exit
            entered = current_ids - self.previous_person_ids
            exited = self.previous_person_ids - current_ids
            if entered:
                self.add_event("PERSON ENTERED", "LOW", "Student",
                               f"Tracking IDs: {sorted(entered)}")
            if exited:
                self.add_event("PERSON LEFT", "MEDIUM", "Student",
                               f"Tracking IDs: {sorted(exited)}")
            self.previous_person_ids = current_ids

            # movement
            fh, fw = img.shape[:2]
            moving = False
            for tid, c in centers.items():
                if tid in self.previous_centers:
                    p = self.previous_centers[tid]
                    if math.hypot(c[0] - p[0], c[1] - p[1]) / max(fw, fh) > self.cfg['MOVEMENT_THRESHOLD']:
                        moving = True
            self.previous_centers = centers.copy()

            now = time.time()
            if moving:
                if self.movement_start is None:
                    self.movement_start = now
                if now - self.movement_start >= self.cfg['MOVEMENT_SECONDS']:
                    self.movement_detected = True
                    self.add_event("UNUSUAL MOVEMENT", "MEDIUM", "Student",
                                   "Sustained body movement detected.")
            else:
                self.movement_start = None
                self.movement_detected = False

            # absence
            if people == 0:
                if self.absent_start is None:
                    self.absent_start = now
                if now - self.absent_start >= self.cfg['ABSENT_SECONDS']:
                    self.add_event("STUDENT ABSENT", "HIGH", "Student",
                                   f"No person detected for more than {self.cfg['ABSENT_SECONDS']:.0f} seconds.")
            else:
                self.absent_start = None

            # multiple people
            if people > 1:
                if self.multiple_start is None:
                    self.multiple_start = now
                if now - self.multiple_start >= self.cfg['MULTIPLE_PERSON_SECONDS']:
                    self.add_event("MULTIPLE PEOPLE", "HIGH", "Exam Area",
                                   f"{people} people detected.")
            else:
                self.multiple_start = None

            # phone
            if self.phone_detected:
                if self.phone_start is None:
                    self.phone_start = now
                if now - self.phone_start >= self.cfg['PHONE_SECONDS']:
                    self.add_event("PHONE DETECTED", "HIGH", "Student",
                                   "Mobile phone detected in camera view.")
            else:
                self.phone_start = None

            # face / eye
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = self.face_landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            self.face_detected = False
            if face.face_landmarks:
                self.face_detected = True
                lm = face.face_landmarks[0]
                d, ex, ey = self.calculate_eye_direction(lm)
                self.eye_direction, self.eye_x, self.eye_y = d, ex, ey
                self.draw_iris(display, lm)
                if d != "CENTER":
                    if self.eye_start is None:
                        self.eye_start = now
                    if now - self.eye_start >= self.cfg['EYE_SECONDS']:
                        self.add_event("PROLONGED EYE DEVIATION", "MEDIUM", "Student",
                                       f"Eye direction: {d}. Requires invigilator review.")
                else:
                    self.eye_start = None
            else:
                self.eye_direction = "NO FACE"
                self.eye_x = self.eye_y = 0.5

            self.status = self.determine_status()
            self.risk_score = self.calculate_risk_score()

            self.frame_counter += 1
            el = time.time() - self.fps_start
            if el >= 1:
                self.fps = self.frame_counter / el
                self.frame_counter = 0
                self.fps_start = time.time()
                self.risk_history.append(self.risk_score)

        except Exception:
            cv2.putText(display, "VISION ERROR", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # slim in-video HUD
        h, w = display.shape[:2]
        bar = display.copy()
        cv2.rectangle(bar, (0, 0), (w, 34), (22, 18, 14), -1)
        display = cv2.addWeighted(bar, 0.78, display, 0.22, 0)
        ok = self.status == "NORMAL"
        cv2.putText(display, f"{self.status}   People {self.people_count}   "
                    f"Eye {self.eye_direction}   Risk {self.risk_score}%",
                    (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 220, 120) if ok else (90, 140, 250), 1)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

    def get_state(self):
        with self.lock:
            return {
                "session_id": self.session_id, "people": self.people_count,
                "objects": self.object_count, "events": self.total_events,
                "status": self.status, "risk": self.risk_score,
                "eye": self.eye_direction, "eye_x": self.eye_x,
                "eye_y": self.eye_y, "fps": self.fps,
                "phone": self.phone_detected, "movement": self.movement_detected,
                "face": self.face_detected,
                "object_names": self.detected_objects.copy(),
                "event_history": self.event_history.copy(),
                "risk_history": list(self.risk_history),
                "started": self.start_time,
            }


# ============================================================
# DESIGN SYSTEM
# Palette: ink #0F1419, panel #161C24, rule #26303C,
#          paper #E8EDF2, muted #8895A5
# Severity is the only colour: calm #4ADE80, review #FBBF24, alert #F87171
# Type: Manrope throughout, tabular figures for numbers
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#0F1419;--panel:#161C24;--rule:#26303C;--paper:#E8EDF2;--muted:#8895A5;
--calm:#4ADE80;--review:#FBBF24;--alert:#F87171;}
html,body,[class*="css"],.stApp{font-family:'Manrope',system-ui,sans-serif!important;}
.stApp{background:var(--ink);color:var(--paper);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding:1.4rem 2.2rem 3rem;max-width:1500px;}
.num{font-variant-numeric:tabular-nums;}

.topbar{display:flex;align-items:center;justify-content:space-between;
padding-bottom:1.1rem;border-bottom:1px solid var(--rule);margin-bottom:1.3rem;}
.brand{font-size:1.55rem;font-weight:800;letter-spacing:-.02em;}
.brand small{display:block;font-size:.82rem;font-weight:500;color:var(--muted);letter-spacing:0;margin-top:2px;}
.session{font-size:.82rem;color:var(--muted);text-align:right;}
.session b{color:var(--paper);font-weight:600;}

.panel{background:var(--panel);border:1px solid var(--rule);border-radius:14px;padding:1.1rem 1.2rem;}
.panel h4{margin:0 0 .8rem;font-size:.95rem;font-weight:700;color:var(--paper);}

.banner{border-radius:14px;padding:1.1rem 1.3rem;border:1px solid var(--rule);
border-left-width:6px;background:var(--panel);}
.banner .t{font-size:1.35rem;font-weight:800;letter-spacing:-.01em;}
.banner .d{color:var(--muted);font-size:.9rem;margin-top:.25rem;}
.banner.calm{border-left-color:var(--calm);} .banner.calm .t{color:var(--calm);}
.banner.review{border-left-color:var(--review);} .banner.review .t{color:var(--review);}
.banner.alert{border-left-color:var(--alert);} .banner.alert .t{color:var(--alert);}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(248,113,113,.45)}100%{box-shadow:0 0 0 14px rgba(248,113,113,0)}}
.banner.alert{animation:pulse 1.6s ease-out infinite;}
@media (prefers-reduced-motion:reduce){.banner.alert{animation:none}}
.spark{width:100%;height:46px;margin-top:.9rem;display:block;}
.sparkcap{display:flex;justify-content:space-between;color:var(--muted);font-size:.75rem;}
.banner.wait{border-left-color:var(--muted);}

.ring-wrap{display:flex;align-items:center;gap:1.2rem;}
.ring{--p:0;--c:var(--calm);width:112px;height:112px;border-radius:50%;flex:none;
background:conic-gradient(var(--c) calc(var(--p)*1%),var(--rule) 0);
display:grid;place-items:center;}
.ring::before{content:"";width:86px;height:86px;border-radius:50%;background:var(--panel);grid-area:1/1;}
.ring span{grid-area:1/1;font-size:1.7rem;font-weight:800;z-index:1;font-variant-numeric:tabular-nums;}
.ring-text{font-size:.88rem;color:var(--muted);line-height:1.5;}
.ring-text b{color:var(--paper);font-size:1rem;display:block;margin-bottom:.15rem;}

.signal{display:flex;justify-content:space-between;align-items:center;
padding:.55rem 0;border-bottom:1px solid var(--rule);font-size:.9rem;}
.signal:last-child{border-bottom:none;}
.signal .k{color:var(--muted);}
.chip{padding:.15rem .6rem;border-radius:999px;font-size:.8rem;font-weight:600;
border:1px solid var(--rule);}
.chip.ok{color:var(--calm);border-color:rgba(74,222,128,.4);background:rgba(74,222,128,.08);}
.chip.warn{color:var(--review);border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.08);}
.chip.bad{color:var(--alert);border-color:rgba(248,113,113,.4);background:rgba(248,113,113,.08);}
.chip.idle{color:var(--muted);}

.stat{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:.8rem 1rem;}
.stat .k{font-size:.78rem;color:var(--muted);font-weight:500;}
.stat .v{font-size:1.6rem;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.2;}

.ev{display:grid;grid-template-columns:5px 1fr auto;gap:.8rem;padding:.7rem .2rem;
border-bottom:1px solid var(--rule);}
.ev:last-child{border-bottom:none;}
.ev .bar{border-radius:3px;}
.ev .bar.HIGH{background:var(--alert);} .ev .bar.MEDIUM{background:var(--review);} .ev .bar.LOW{background:var(--calm);}
.ev .n{font-weight:700;font-size:.92rem;}
.ev .m{color:var(--muted);font-size:.82rem;margin-top:2px;}
.ev .ts{color:var(--muted);font-size:.8rem;font-variant-numeric:tabular-nums;white-space:nowrap;}
.empty{color:var(--muted);font-size:.92rem;padding:1.2rem .2rem;}

.notice{color:var(--muted);font-size:.8rem;line-height:1.5;margin-top:1.6rem;
padding-top:1rem;border-top:1px solid var(--rule);}

[data-testid="stTabs"] button{font-weight:600;}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--paper);}
[data-testid="stDataFrame"]{border:1px solid var(--rule);border-radius:12px;}
div[data-testid="stVideo"], video{border-radius:14px;}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:700;border:1px solid var(--rule);}
:focus-visible{outline:2px solid var(--review)!important;outline-offset:2px;}
@media (max-width:900px){.block-container{padding:1rem;}.ring{width:92px;height:92px;}.ring::before{width:70px;height:70px;}}
</style>
""", unsafe_allow_html=True)


def H(s):
    """Strip indentation so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines())


with st.sidebar:
    st.subheader("Calibration")
    st.caption("Changes apply to the live session. Defaults suit a student seated about 60 cm from the camera.")
    gaze = st.slider("Gaze tolerance", 0.05, 0.30, 0.10, 0.01,
                     help="Higher means the eyes can move further before it counts as deviation.")
    eye_s = st.slider("Seconds before gaze alert", 1.0, 6.0, EYE_SECONDS, 0.5)
    abs_s = st.slider("Seconds before absence alert", 2.0, 15.0, ABSENT_SECONDS, 1.0)
    mov = st.slider("Movement sensitivity", 0.02, 0.15, MOVEMENT_THRESHOLD, 0.005,
                    help="Lower is more sensitive.")
    st.caption("Decision support only. An invigilator makes every call.")

# ------------------------------------------------------------
# CAMERA (started once, state read by fragments)
# ------------------------------------------------------------
ctx = webrtc_streamer(
    key="examguard-camera-v3",
    video_processor_factory=ExamGuardProcessor,
    media_stream_constraints={"video": {"width": {"ideal": 640},
                                        "height": {"ideal": 480},
                                        "frameRate": {"ideal": 24}},
                              "audio": False},
    async_processing=True,
)

def apply_calibration():
    if ctx is not None and ctx.video_processor:
        c = ctx.video_processor.cfg
        c.update(EYE_LEFT_LIMIT=0.5 - gaze,
                 EYE_RIGHT_LIMIT=0.5 + gaze, EYE_SECONDS=eye_s,
                 ABSENT_SECONDS=abs_s, MOVEMENT_THRESHOLD=mov)


apply_calibration()

EMPTY_STATE = {"session_id": "-", "people": 0, "objects": 0, "events": 0,
               "status": "WAITING", "risk": 0, "eye": "CENTER", "eye_x": 0.5,
               "eye_y": 0.5, "fps": 0, "phone": False, "movement": False,
               "face": False, "object_names": [], "event_history": [],
               "risk_history": [], "started": 0}


def get_current_state():
    if ctx is not None and ctx.video_processor:
        try:
            return ctx.video_processor.get_state()
        except Exception:
            pass
    return EMPTY_STATE


STATUS_COPY = {
    "WAITING": ("wait", "Waiting for camera", "Press Start on the video panel to begin monitoring."),
    "NORMAL": ("calm", "Normal behaviour", "Presence and behaviour are within the expected baseline."),
    "STUDENT ABSENT": ("alert", "Student not in view", "No person detected. An invigilator should check the seat."),
    "MULTIPLE PEOPLE": ("alert", "More than one person in view", "Review the camera feed to see who is present."),
    "PHONE DETECTED": ("alert", "Phone in view", "A mobile phone is visible. Review the feed."),
    "EYE DEVIATION": ("review", "Eyes off the paper", "Gaze is away from centre. This can be normal; review if it persists."),
    "UNUSUAL MOVEMENT": ("review", "Unusual movement", "Sustained movement differs from the baseline."),
}


def spark(hist, color):
    if len(hist) < 2:
        return '<div class="sparkcap" style="margin-top:.9rem">Risk trend appears after a few seconds.</div>'
    n = len(hist)
    pts = " ".join(f"{i * 300 / (n - 1):.1f},{44 - clamp(v, 0, 100) * 0.42:.1f}" for i, v in enumerate(hist))
    return (f'<svg class="spark" viewBox="0 0 300 46" preserveAspectRatio="none" role="img" '
            f'aria-label="Risk over the last {n} seconds"><polyline points="{pts}" fill="none" '
            f'stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>'
            f'<div class="sparkcap"><span>{n}s ago</span><span>now</span></div>')


def chip(label, level):
    return f'<span class="chip {level}">{label}</span>'


# ============================================================
# HEADER
# ============================================================
@st.fragment(run_every=2)
def header():
    s = get_current_state()
    live = ctx is not None and ctx.state.playing
    sid = s["session_id"] if s["session_id"] != "-" else "not started"
    mins, secs = divmod(int(time.time() - s["started"]), 60) if s["started"] else (0, 0)
    dot = "var(--calm)" if live else "var(--muted)"
    st.markdown(H(f"""
    <div class="topbar">
      <div class="brand">🛡️ ExamGuard AI
        <small>Live exam room monitoring for invigilators</small></div>
      <div class="session"><span style="color:{dot}">●</span>
        <b>{'Monitoring' if live else 'Idle'}</b> <span class="num">{mins:02d}:{secs:02d}</span><br>Session {sid}</div>
    </div>"""), unsafe_allow_html=True)


header()

# ============================================================
# MAIN: camera left, decision panel right
# ============================================================
left, right = st.columns([1.5, 1], gap="large")

with left:
    st.caption("Camera feed. Boxes show tracked people; yellow dots mark iris position.")
    # webrtc widget is rendered above; the columns below hold live stats
    @st.fragment(run_every=1)
    def stats_row():
        s = get_current_state()
        cols = st.columns(4)
        items = [("People in view", s["people"]), ("Other objects", s["objects"]),
                 ("Events logged", s["events"]), ("Frames per second", f'{s["fps"]:.0f}')]
        for c, (k, v) in zip(cols, items):
            c.markdown(H(f'<div class="stat"><div class="k">{k}</div>'
                         f'<div class="v">{v}</div></div>'), unsafe_allow_html=True)
    stats_row()

with right:
    @st.fragment(run_every=1)
    def decision_panel():
        s = get_current_state()
        level, title, desc = STATUS_COPY.get(s["status"], STATUS_COPY["WAITING"])
        st.markdown(H(f"""
        <div class="banner {level}"><div class="t">{title}</div>
        <div class="d">{desc}</div></div>"""), unsafe_allow_html=True)

        risk = s["risk"]
        rc = "var(--calm)" if risk < 30 else "var(--review)" if risk < 60 else "var(--alert)"
        rlabel = "Low" if risk < 30 else "Elevated" if risk < 60 else "High"
        st.markdown(H(f"""
        <div class="panel" style="margin-top:1rem">
          <div class="ring-wrap">
            <div class="ring" style="--p:{risk};--c:{rc}"><span>{risk}</span></div>
            <div class="ring-text"><b>{rlabel} review priority</b>
            Combines people count, phone, gaze and movement signals.
            It guides attention. It is not proof of cheating.</div>
          </div>
          {spark(s['risk_history'], rc)}
        </div>"""), unsafe_allow_html=True)

        eye = s["eye"]
        eye_chip = (chip("No face", "idle") if eye == "NO FACE"
                    else chip("Centre", "ok") if eye == "CENTER"
                    else chip(eye.title(), "warn"))
        rows = [
            ("Face detected", chip("Yes", "ok") if s["face"] else chip("No", "idle")),
            ("Gaze direction", eye_chip),
            ("Phone", chip("In view", "bad") if s["phone"] else chip("Not seen", "ok")),
            ("Movement", chip("Unusual", "warn") if s["movement"] else chip("Normal", "ok")),
            ("Iris position", f'<span class="num">{s["eye_x"]:.2f}, {s["eye_y"]:.2f}</span>'),
        ]
        body = "".join(f'<div class="signal"><span class="k">{k}</span>{v}</div>' for k, v in rows)
        objs = s["object_names"]
        objline = ("".join(chip(o, "idle") + " " for o in objs)
                   if objs else '<span class="k" style="color:var(--muted)">None</span>')
        st.markdown(H(f"""
        <div class="panel" style="margin-top:1rem"><h4>Live signals</h4>{body}
        <div class="signal" style="display:block"><div class="k" style="margin-bottom:.4rem">Other objects</div>{objline}</div>
        </div>"""), unsafe_allow_html=True)
    decision_panel()

# ============================================================
# BELOW: timeline / analytics / log
# ============================================================
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
tab_timeline, tab_analytics, tab_log, tab_about = st.tabs(
    ["Event timeline", "Analytics", "Full log and report", "About"])

with tab_timeline:
    @st.fragment(run_every=2)
    def timeline():
        events = get_current_state()["event_history"]
        if not events:
            st.markdown('<div class="empty">No events yet. Events appear here as soon as '
                        'something unusual is seen.</div>', unsafe_allow_html=True)
            return
        html = ""
        for e in reversed(events[-15:]):
            html += (f'<div class="ev"><div class="bar {e["Severity"]}"></div>'
                     f'<div><div class="n">{e["Event"].title()}</div>'
                     f'<div class="m">{e["Entity"]}: {e["Details"]}</div></div>'
                     f'<div class="ts">{e["Time"]}</div></div>')
        st.markdown(f'<div class="panel">{html}</div>', unsafe_allow_html=True)
    timeline()

with tab_analytics:
    @st.fragment(run_every=3)
    def analytics():
        events = get_current_state()["event_history"]
        if not events:
            st.markdown('<div class="empty">Charts appear after the first event.</div>',
                        unsafe_allow_html=True)
            return
        df = pd.DataFrame(events)
        sev = df["Severity"].value_counts()
        c = st.columns(4)
        for col, (k, v) in zip(c, [("Total events", len(df)),
                                   ("High", int(sev.get("HIGH", 0))),
                                   ("Medium", int(sev.get("MEDIUM", 0))),
                                   ("Low", int(sev.get("LOW", 0)))]):
            col.markdown(H(f'<div class="stat"><div class="k">{k}</div>'
                           f'<div class="v">{v}</div></div>'), unsafe_allow_html=True)
        a, b = st.columns([1.3, 1])
        a.markdown("**Events by type**")
        a.bar_chart(df["Event"].value_counts())
        b.markdown("**Events by severity**")
        b.bar_chart(sev)
    analytics()

with tab_log:
    @st.fragment(run_every=3)
    def full_log():
        try:
            df = pd.read_csv(CSV_FILE)
            if df.empty:
                st.markdown('<div class="empty">Nothing recorded yet.</div>',
                            unsafe_allow_html=True)
                return
            st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Couldn't read the event log: {e}")
    full_log()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            st.download_button("Download event report (CSV)", f.read(),
                               file_name="ExamGuard_Examination_Report.csv",
                               mime="text/csv", use_container_width=True)

with tab_about:
    st.markdown("""
**How it works.** Each frame is checked for people and objects (YOLO), tracked with
ByteTrack so every person keeps an ID, then analysed for face and iris position
(MediaPipe). Rules turn those signals into timestamped events and a review priority score.

| Requirement | How ExamGuard AI covers it |
|---|---|
| Recognise people and objects | YOLO person and object detection |
| Track the same person | ByteTrack IDs |
| Understand behaviour | Gaze, movement and presence analysis |
| Detect meaningful events | Timestamped event engine |
| Tell normal from unusual | Thresholds plus review priority score |
| Name who and when | Entity, time and event in the log |
""")

st.markdown(H("""
<div class="notice"><b>Decision support only.</b> Events mark potentially unusual behaviour
for an authorised invigilator to review. ExamGuard AI does not establish cheating, identity or intent.
HackNex 2026, PSI07.</div>"""), unsafe_allow_html=True)