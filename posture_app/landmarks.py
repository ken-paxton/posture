"""On-device landmark detection using MediaPipe Tasks (Pose + Face Landmarker).

Everything runs locally on the CPU. Model files are loaded from the local
``models/`` directory and verified against pinned SHA-256 hashes, so nothing is
fetched at runtime and no imagery leaves the machine.

Two models are used because each covers the other's blind spot with a frontal
webcam:

* Pose Landmarker (BlazePose GHUM, 33 landmarks) gives shoulders, ears and,
  when visible, hips. It also produces metric "world" landmarks with a learned
  depth estimate.
* Face Landmarker (478 landmarks + facial transformation matrix) gives a
  well-conditioned 3-D head pose (pitch/yaw/roll) and head-to-camera distance,
  which a head-on camera cannot otherwise recover from 2-D keypoints.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILES = {
    "pose_lite": "pose_landmarker_lite.task",
    "pose_full": "pose_landmarker_full.task",
    "pose_heavy": "pose_landmarker_heavy.task",
    "face": "face_landmarker.task",
}

# Pinned hashes of the float16 "latest" models published by Google as of 2026-10.
MODEL_SHA256 = {
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "pose_landmarker_full.task": "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    "pose_landmarker_heavy.task": "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
    "pose_landmarker_lite.task": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
}

MODEL_URLS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}

# BlazePose landmark indices.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

# Face mesh indices: outer eye corners, nose tip, chin, forehead.
FACE_L_EYE_OUTER, FACE_R_EYE_OUTER = 263, 33
FACE_NOSE_TIP, FACE_CHIN, FACE_FOREHEAD = 1, 152, 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(name: str, verify: bool = True) -> Path:
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Model {name} not found in {MODELS_DIR}. Run `python -m posture_app fetch-models` once."
        )
    if verify:
        digest = sha256(path)
        if digest != MODEL_SHA256[name]:
            raise RuntimeError(f"Model {name} failed SHA-256 verification ({digest}).")
    return path


@dataclass
class Point:
    """A landmark in image pixels (x, y), plus model depth and visibility."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class HeadPose:
    pitch: float  # degrees, positive = chin down (neck flexion)
    yaw: float  # degrees, positive = turned to the image's right
    roll: float  # degrees, positive = head tilted clockwise in the image
    distance: float  # head-to-camera distance in the face model's units (~cm)


@dataclass
class Observation:
    """Landmarks for one frame, in pixel coordinates of the analysed image."""

    timestamp: float  # seconds
    width: int
    height: int
    pose: dict[int, Point] = field(default_factory=dict)
    pose_world: dict[int, Point] = field(default_factory=dict)  # metres, hip-centred
    face: dict[int, Point] = field(default_factory=dict)
    head: HeadPose | None = None

    @property
    def has_person(self) -> bool:
        return bool(self.pose) or bool(self.face)


def head_pose_from_matrix(matrix: np.ndarray) -> HeadPose:
    """Decompose MediaPipe's facial transformation matrix (canonical face -> camera).

    The canonical face model looks down -z in an OpenGL-style camera frame
    (x right, y up). Angles are reported in conventional anatomical signs.
    """
    m = np.asarray(matrix, dtype=float).reshape(4, 4)
    r = m[:3, :3]
    # Remove any scale the solver folded into the rotation.
    r = r / np.cbrt(max(np.linalg.det(r), 1e-9))
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    distance = float(-m[2, 3])
    # Signs verified empirically on video: pitch rises as the nose tip drops toward
    # the chin in the image (flexion); roll rises as the eye line turns clockwise.
    return HeadPose(pitch=pitch, yaw=yaw, roll=-roll, distance=distance)


class LandmarkEngine:
    """Runs Pose + Face Landmarker on frames in VIDEO mode (synchronous, monotonic)."""

    def __init__(self, pose_model: str = "pose_full", max_width: int = 640, use_face: bool = True):
        from mediapipe.tasks.python import BaseOptions, vision

        self._vision = vision
        self.max_width = max_width
        self._last_ts_ms = -1
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path(MODEL_FILES[pose_model]))),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.face = None
        if use_face:
            self.face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path(MODEL_FILES["face"]))),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_facial_transformation_matrixes=True,
                )
            )

    def close(self) -> None:
        self.pose.close()
        if self.face is not None:
            self.face.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Downscale to the analysis resolution (keeps CPU use low)."""
        h, w = frame_bgr.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            frame_bgr = cv2.resize(frame_bgr, (self.max_width, round(h * scale)), interpolation=cv2.INTER_AREA)
        return frame_bgr

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Observation:
        import mediapipe as mp

        frame_bgr = self.prepare(frame_bgr)
        h, w = frame_bgr.shape[:2]
        ts_ms = max(int(round(timestamp * 1000)), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        obs = Observation(timestamp=timestamp, width=w, height=h)
        pr = self.pose.detect_for_video(image, ts_ms)
        if pr.pose_landmarks:
            for i, lm in enumerate(pr.pose_landmarks[0]):
                vis = min(lm.visibility or 0.0, lm.presence or 1.0) if lm.presence is not None else (lm.visibility or 0.0)
                obs.pose[i] = Point(lm.x * w, lm.y * h, lm.z * w, vis)
            if pr.pose_world_landmarks:
                for i, lm in enumerate(pr.pose_world_landmarks[0]):
                    obs.pose_world[i] = Point(lm.x, lm.y, lm.z, obs.pose[i].visibility)

        if self.face is not None:
            fr = self.face.detect_for_video(image, ts_ms)
            if fr.face_landmarks:
                for i in (FACE_L_EYE_OUTER, FACE_R_EYE_OUTER, FACE_NOSE_TIP, FACE_CHIN, FACE_FOREHEAD):
                    lm = fr.face_landmarks[0][i]
                    obs.face[i] = Point(lm.x * w, lm.y * h, lm.z * w, 1.0)
                if fr.facial_transformation_matrixes:
                    obs.head = head_pose_from_matrix(fr.facial_transformation_matrixes[0])
        return obs
