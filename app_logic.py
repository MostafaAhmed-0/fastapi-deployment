import os
import struct
import cv2
import numpy as np
import urllib.request
from collections import deque
from pushup_validator import PushUpValidator, ValidatorState, FrameResult

# TFLite interpreter
try:
    import tflite_runtime.interpreter as tflite
    TFLiteInterpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    TFLiteInterpreter = tf.lite.Interpreter

# MediaPipe-compatible constants
NUM_KEYPOINTS = 33
FEATURES_PER_KEYPOINT = 4
LANDMARK_FEATURES = NUM_KEYPOINTS * FEATURES_PER_KEYPOINT
NUM_ANGLES = 6
TOTAL_FEATURES = LANDMARK_FEATURES + NUM_ANGLES
SEQUENCE_LENGTH = 30

MOVENET_MODEL_URL = (
    "https://tfhub.dev/google/lite-model/"
    "movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
)

MOVENET_TO_MEDIAPIPE = {
    0:  0, 1:  2, 2:  5, 3:  7, 4:  8, 5:  11, 6:  12, 7:  13, 8:  14,
    9:  15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
}

def download_movenet(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    print(f"[INFO] Downloading MoveNet Thunder model to {dest} ...")
    urllib.request.urlretrieve(url, dest)

class MoveNetDetector:
    def __init__(self, model_path: str, num_threads: int = 4):
        self._interpreter = TFLiteInterpreter(
            model_path=model_path, num_threads=num_threads
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._input_shape = self._input_details[0]["shape"]
        self._input_h, self._input_w = self._input_shape[1], self._input_shape[2]
        self._input_dtype = self._input_details[0]["dtype"]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_w, self._input_h))
        input_tensor = np.expand_dims(resized, axis=0).astype(self._input_dtype)
        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])
        return np.squeeze(output)

def detect_pose(frame: np.ndarray, detector: MoveNetDetector):
    raw_kp = detector.detect(frame)
    core_joints = [5, 6, 7, 8, 9, 10, 11, 12]
    core_confidences = [raw_kp[i, 2] for i in core_joints]
    if np.mean(core_confidences) < 0.3:
        return np.empty(0, dtype=np.float32), raw_kp

    coords = np.zeros(LANDMARK_FEATURES, dtype=np.float32)
    for mn_idx, mp_idx in MOVENET_TO_MEDIAPIPE.items():
        y_norm, x_norm, conf = raw_kp[mn_idx]
        offset = mp_idx * FEATURES_PER_KEYPOINT
        coords[offset]     = x_norm
        coords[offset + 1] = y_norm
        coords[offset + 2] = 0.0
        coords[offset + 3] = conf
    return coords, raw_kp

def normalize_frame(row: np.ndarray) -> np.ndarray:
    row = row.copy()
    lh, rh = 23 * 4, 24 * 4
    hip_centre = np.array([(row[lh] + row[rh]) / 2, (row[lh+1] + row[rh+1]) / 2, (row[lh+2] + row[rh+2]) / 2])
    ls, rs = 11 * 4, 12 * 4
    shoulder_centre = np.array([(row[ls] + row[rs]) / 2, (row[ls+1] + row[rs+1]) / 2, (row[ls+2] + row[rs+2]) / 2])
    torso_length = np.linalg.norm(shoulder_centre - hip_centre)
    if torso_length < 1e-6: torso_length = 1.0
    for j in range(NUM_KEYPOINTS):
        idx = j * 4
        row[idx]     = (row[idx]     - hip_centre[0]) / torso_length
        row[idx + 1] = (row[idx + 1] - hip_centre[1]) / torso_length
        row[idx + 2] = (row[idx + 2] - hip_centre[2]) / torso_length
    return row

def _angle_3d(a, b, c):
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def _get_xyz(row, landmark_idx):
    i = landmark_idx * 4
    return np.array([row[i], row[i+1], row[i+2]])

def compute_joint_angles(row):
    angles = np.array([
        _angle_3d(_get_xyz(row, 11), _get_xyz(row, 13), _get_xyz(row, 15)),
        _angle_3d(_get_xyz(row, 12), _get_xyz(row, 14), _get_xyz(row, 16)),
        _angle_3d(_get_xyz(row, 13), _get_xyz(row, 11), _get_xyz(row, 23)),
        _angle_3d(_get_xyz(row, 14), _get_xyz(row, 12), _get_xyz(row, 24)),
        _angle_3d(_get_xyz(row, 23), _get_xyz(row, 25), _get_xyz(row, 27)),
        _angle_3d(_get_xyz(row, 24), _get_xyz(row, 26), _get_xyz(row, 28)),
    ], dtype=np.float32) / 180.0
    return angles

class PushUpSession:
    def __init__(self, lstm_model, detector):
        self.model = lstm_model
        self.detector = detector
        self.validator = PushUpValidator()
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0
        self.real_rep_count = 0
        self.prediction_text = ""
        self.confidence = 0.0
        self.predict_every_n = 5 # Faster for web

    def process_frame(self, frame):
        keypoints_flat, raw_kp = detect_pose(frame, self.detector)
        person_detected = keypoints_flat.shape[0] == LANDMARK_FEATURES

        if person_detected:
            vr = self.validator.process_frame(keypoints_flat, model_confidence=self.confidence)
            normalized = normalize_frame(keypoints_flat)
            angles = compute_joint_angles(normalized)
            augmented = np.concatenate([normalized, angles])
            self.frame_buffer.append(augmented)

            self.frame_count += 1
            gate_is_open = vr.state in (ValidatorState.UP, ValidatorState.DOWN)
            if (gate_is_open and len(self.frame_buffer) == SEQUENCE_LENGTH and self.frame_count % self.predict_every_n == 0):
                seq = np.array(list(self.frame_buffer), dtype=np.float32)
                seq = np.expand_dims(seq, axis=0)
                pred = self.model.predict(seq, verbose=0)[0][0]
                self.confidence = float(pred if pred > 0.5 else 1 - pred)
                if pred <= 0.5:
                    self.prediction_text = "Correct"
                else:
                    self.prediction_text = "Wrong"

            if vr.rep_completed:
                if self.prediction_text == "Correct" and self.confidence > 0.7:
                    self.real_rep_count += 1

            return {
                "person_detected": True,
                "rep_count": self.real_rep_count,
                "prediction": self.prediction_text,
                "confidence": self.confidence,
                "state": vr.state.name,
                "is_valid_posture": vr.is_valid_posture,
                "feedback": vr.feedback,
                "gate_progress": vr.gate_progress,
                "gate_required": vr.gate_required,
                "keypoints": raw_kp.tolist() # For drawing on client side
            }
        
        return {
            "person_detected": False,
            "rep_count": self.real_rep_count,
            "prediction": "",
            "confidence": 0.0,
            "state": "NOT_DETECTED",
            "is_valid_posture": False,
            "feedback": ["Person not detected"],
            "gate_progress": 0,
            "gate_required": 10,
            "keypoints": []
        }


# ──────────────────────────────────────────────
# YUV420 Binary Frame Reconstruction (Android)
# ──────────────────────────────────────────────

HEADER_SIZE = 24  # 6 × int32 (little-endian)


def parse_yuv420_header(data: bytes):
    """
    Parse the 24-byte binary header from an Android YUV420 frame.

    Returns
    -------
    meta : dict — width, height, y_row_stride, uv_row_stride,
                  uv_pixel_stride, sensor_orientation
    payload : bytes — Y + U + V plane bytes
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Frame too small: {len(data)} bytes (need >= {HEADER_SIZE})")

    header = struct.unpack('<6i', data[:HEADER_SIZE])
    meta = {
        'width':              header[0],
        'height':             header[1],
        'y_row_stride':       header[2],
        'uv_row_stride':      header[3],
        'uv_pixel_stride':    header[4],
        'sensor_orientation': header[5],
    }
    return meta, data[HEADER_SIZE:]


def reconstruct_yuv420_to_bgr(data: bytes) -> np.ndarray:
    """
    Reconstruct a BGR image from a raw binary YUV420 frame.

    Binary format: [24-byte header] [Y plane] [U plane] [V plane]

    Handles:
      • Stride padding (row stride != width)
      • Interleaved UV (pixel stride == 2, NV21-style) → planar I420
      • Sensor rotation (0, 90, 180, 270)

    Returns
    -------
    bgr : np.ndarray (H, W, 3) — BGR image for cv2 / MoveNet
    """
    meta, payload = parse_yuv420_header(data)

    w  = meta['width']
    h  = meta['height']
    y_stride   = meta['y_row_stride']
    uv_stride  = meta['uv_row_stride']
    uv_pxstride = meta['uv_pixel_stride']
    orientation = meta['sensor_orientation']

    # ── Split payload into raw plane buffers ──────────
    y_plane_size  = y_stride * h
    uv_h = h // 2
    uv_plane_size = uv_stride * uv_h

    y_raw = np.frombuffer(payload[:y_plane_size], dtype=np.uint8)
    remaining = payload[y_plane_size:]
    u_raw = np.frombuffer(remaining[:uv_plane_size], dtype=np.uint8)
    v_raw = np.frombuffer(remaining[uv_plane_size:uv_plane_size * 2], dtype=np.uint8)

    # ── Extract Y plane (handle stride padding) ──────
    if y_stride == w:
        y_plane = y_raw[:w * h].reshape(h, w)
    else:
        y_plane = np.zeros((h, w), dtype=np.uint8)
        for row in range(h):
            start = row * y_stride
            y_plane[row, :] = y_raw[start:start + w]

    # ── Extract U and V planes ────────────────────────
    uv_w = w // 2

    if uv_pxstride == 1:
        # Already planar
        if uv_stride == uv_w:
            u_plane = u_raw[:uv_w * uv_h].reshape(uv_h, uv_w)
            v_plane = v_raw[:uv_w * uv_h].reshape(uv_h, uv_w)
        else:
            u_plane = np.zeros((uv_h, uv_w), dtype=np.uint8)
            v_plane = np.zeros((uv_h, uv_w), dtype=np.uint8)
            for row in range(uv_h):
                start = row * uv_stride
                u_plane[row, :] = u_raw[start:start + uv_w]
                v_plane[row, :] = v_raw[start:start + uv_w]
    else:
        # Interleaved (NV21-style, pixel stride == 2) — de-interleave
        u_plane = np.zeros((uv_h, uv_w), dtype=np.uint8)
        v_plane = np.zeros((uv_h, uv_w), dtype=np.uint8)
        for row in range(uv_h):
            for col in range(uv_w):
                idx = row * uv_stride + col * uv_pxstride
                if idx < len(u_raw):
                    u_plane[row, col] = u_raw[idx]
                if idx < len(v_raw):
                    v_plane[row, col] = v_raw[idx]

    # ── Build I420 layout and convert ─────────────────
    # I420 for cv2: (H*3/2, W) — U and V must be reshaped from (H/2, W/2) to (H/4, W)
    u_flat = u_plane.reshape(-1, w)  # (H/4, W)
    v_flat = v_plane.reshape(-1, w)  # (H/4, W)
    yuv_i420 = np.vstack([y_plane, u_flat, v_flat]).astype(np.uint8)
    bgr = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)

    # ── Apply sensor rotation ─────────────────────────
    if orientation == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif orientation == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return bgr
