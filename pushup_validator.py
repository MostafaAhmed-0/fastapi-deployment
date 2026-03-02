"""
pushup_validator.py
===================
Modular push-up position validation, gating, and robust rep-counting
state machine. Designed to plug into realtime_detection.py.

Prevents false rep counts from non-exercise poses (standing, walking,
random movement) by requiring:
    1. Valid push-up posture (horizontal body, straight back, extended knees)
    2. Sustained posture for N consecutive frames (gating)
    3. Full DOWN → UP elbow transition (state machine)
    4. Minimum model confidence score
    5. Anti-jitter cooldown between reps
"""

from __future__ import annotations

import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple


# ──────────────────────────────────────────────
# Enums & Data Classes
# ──────────────────────────────────────────────

class ValidatorState(Enum):
    """High-level state of the push-up validator."""
    NOT_READY = auto()   # person not in push-up posture
    GATING    = auto()   # valid posture detected, accumulating frames
    UP        = auto()   # push-up ready / top position
    DOWN      = auto()   # push-up bottom position


@dataclass
class FrameResult:
    """Per-frame output returned by PushUpValidator.process_frame()."""
    state: ValidatorState = ValidatorState.NOT_READY
    is_valid_posture: bool = False
    gate_progress: int = 0          # frames of valid posture accumulated
    gate_required: int = 5          # frames needed to pass gate
    rep_count: int = 0              # kept for backward compat (no longer incremented)
    rep_just_counted: bool = False  # kept for backward compat
    rep_completed: bool = False     # NEW: True when a full DOWN→UP transition occurs
    angles: Dict[str, float] = field(default_factory=dict)
    posture_details: Dict[str, bool] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)  # NEW: Actionable coaching tips


# ──────────────────────────────────────────────
# 3D Angle Helpers
# ──────────────────────────────────────────────

def _get_xyz(row: np.ndarray, landmark_idx: int) -> np.ndarray:
    """Extract (x, y, z) for a MediaPipe landmark from stride-4 flat array."""
    i = landmark_idx * 4
    return np.array([row[i], row[i + 1], row[i + 2]], dtype=np.float64)


def _get_visibility(row: np.ndarray, landmark_idx: int) -> float:
    """Get the visibility score for a landmark."""
    return float(row[landmark_idx * 4 + 3])


def _angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at point b formed by segments a→b and c→b."""
    ba = a.astype(np.float64) - b.astype(np.float64)
    bc = c.astype(np.float64) - b.astype(np.float64)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_a = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _joints_visible(row: np.ndarray, indices: list[int],
                    threshold: float) -> bool:
    """Return True if ALL listed landmarks have visibility ≥ threshold."""
    return all(_get_visibility(row, idx) >= threshold for idx in indices)


def _best_side_visible(row: np.ndarray, threshold: float) -> str:
    """
    Determine which body side (left/right) has better landmark visibility.
    Returns 'left', 'right', or 'none' if neither side is sufficiently visible.
    At least shoulder + elbow + wrist + hip must be visible on one side.
    """
    left_indices  = [11, 13, 15, 23, 25, 27]  # L shoulder, elbow, wrist, hip, knee, ankle
    right_indices = [12, 14, 16, 24, 26, 28]  # R shoulder, elbow, wrist, hip, knee, ankle

    left_vis  = np.mean([_get_visibility(row, i) for i in left_indices])
    right_vis = np.mean([_get_visibility(row, i) for i in right_indices])

    # Require at least shoulder+elbow+wrist+hip (4 core joints) visible
    left_core  = all(_get_visibility(row, i) >= threshold for i in [11, 13, 15, 23])
    right_core = all(_get_visibility(row, i) >= threshold for i in [12, 14, 16, 24])

    if left_core and right_core:
        return 'left' if left_vis >= right_vis else 'right'
    elif left_core:
        return 'left'
    elif right_core:
        return 'right'
    return 'none'


# ──────────────────────────────────────────────
# PushUpValidator
# ──────────────────────────────────────────────

class PushUpValidator:
    """
    Gate-guarded push-up rep counter.

    Workflow per frame:
        1. Compute 3D joint angles (with visibility check).
        2. Validate push-up posture.
        3. Update gating counter.
        4. Run DOWN/UP state machine.
        5. Check model-confidence gate.
        6. Return FrameResult.

    Parameters
    ----------
    gate_frames : int
        Consecutive valid-posture frames required before counting starts.
    elbow_down_angle : float
        Elbow angle (degrees) below which the person is in DOWN position.
    elbow_up_angle : float
        Elbow angle (degrees) above which the person is in UP position.
    hip_angle_range : tuple[float, float]
        Acceptable hip-angle range for a straight back.
    knee_min_angle : float
        Minimum knee angle to consider legs straight.
    elbow_ready_min : float
        Minimum elbow angle for the "ready" (extended arms) posture.
    confidence_threshold : float
        Minimum LSTM-model confidence to allow a rep to count.
    visibility_threshold : float
        Minimum MediaPipe landmark visibility to trust a joint.
    cooldown_frames : int
        Minimum frames between two rep counts (anti-jitter).
    posture_loss_tolerance : int
        Frames of invalid posture tolerated before resetting state.
    smoothing_window : int
        Number of past frames used for angle smoothing.
    horizontal_y_threshold : float
        Maximum normalised Y-difference between shoulder-centre and
        hip-centre to consider the body horizontal.
    """

    # MediaPipe landmark indices used for push-up analysis
    _L_SHOULDER = 11
    _R_SHOULDER = 12
    _L_ELBOW    = 13
    _R_ELBOW    = 14
    _L_WRIST    = 15
    _R_WRIST    = 16
    _L_HIP      = 23
    _R_HIP      = 24
    _L_KNEE     = 25
    _R_KNEE     = 26
    _L_ANKLE    = 27
    _R_ANKLE    = 28

    # Per-side landmark groups for visibility fallback
    _LEFT_LANDMARKS  = [_L_SHOULDER, _L_ELBOW, _L_WRIST, _L_HIP, _L_KNEE, _L_ANKLE]
    _RIGHT_LANDMARKS = [_R_SHOULDER, _R_ELBOW, _R_WRIST, _R_HIP, _R_KNEE, _R_ANKLE]

    def __init__(
        self,
        gate_frames: int = 5,
        elbow_down_angle: float = 90.0,
        elbow_up_angle: float = 150.0,
        hip_angle_range: Tuple[float, float] = (130.0, 180.0),
        knee_min_angle: float = 130.0,
        elbow_ready_min: float = 140.0,
        confidence_threshold: float = 0.7,
        visibility_threshold: float = 0.3,
        cooldown_frames: int = 5,
        posture_loss_tolerance: int = 10,
        smoothing_window: int = 5,
        horizontal_y_threshold: float = 0.30,
    ):
        # Configuration
        self.gate_frames = gate_frames
        self.elbow_down_angle = elbow_down_angle
        self.elbow_up_angle = elbow_up_angle
        self.hip_angle_range = hip_angle_range
        self.knee_min_angle = knee_min_angle
        self.elbow_ready_min = elbow_ready_min
        self.confidence_threshold = confidence_threshold
        self.visibility_threshold = visibility_threshold
        self.cooldown_frames = cooldown_frames
        self.posture_loss_tolerance = posture_loss_tolerance
        self.smoothing_window = smoothing_window
        self.horizontal_y_threshold = horizontal_y_threshold

        # Internal state
        self._state = ValidatorState.NOT_READY
        self._gate_counter = 0
        self._rep_count = 0
        self._frames_since_last_rep = self.cooldown_frames  # allow first rep immediately
        self._bad_posture_streak = 0
        self._deepest_elbow_this_rep = 180.0
        self._feedback_memory: Dict[str, float] = {}
        self.feedback_ttl = 2.0  # seconds to keep feedback visible

        # Smoothing buffers (one deque per angle key)
        self._angle_buffers: Dict[str, deque] = {
            "l_elbow": deque(maxlen=smoothing_window),
            "r_elbow": deque(maxlen=smoothing_window),
            "l_shoulder": deque(maxlen=smoothing_window),
            "r_shoulder": deque(maxlen=smoothing_window),
            "l_hip": deque(maxlen=smoothing_window),
            "r_hip": deque(maxlen=smoothing_window),
            "l_knee": deque(maxlen=smoothing_window),
            "r_knee": deque(maxlen=smoothing_window),
        }

    # ── public properties ──────────────────────

    @property
    def state(self) -> ValidatorState:
        return self._state

    @property
    def rep_count(self) -> int:
        return self._rep_count

    def reset(self) -> None:
        """Reset all internal state (new session)."""
        self._state = ValidatorState.NOT_READY
        self._gate_counter = 0
        self._rep_count = 0
        self._frames_since_last_rep = self.cooldown_frames
        self._bad_posture_streak = 0
        self._deepest_elbow_this_rep = 180.0
        self._feedback_memory.clear()
        for buf in self._angle_buffers.values():
            buf.clear()

    # ── angle computation ──────────────────────

    def compute_angles(self, row: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Compute 8 joint angles from a single frame's raw/normalised landmarks.
        Uses per-side visibility fallback: if only one side is visible,
        mirrors that side's angles to the other. Returns None only if
        NEITHER side has enough visible joints.
        """
        best_side = _best_side_visible(row, self.visibility_threshold)
        if best_side == 'none':
            return None

        angles: Dict[str, float] = {}

        # Compute left side if visible
        left_ok = _joints_visible(row, [self._L_SHOULDER, self._L_ELBOW,
                                         self._L_WRIST, self._L_HIP],
                                   self.visibility_threshold)
        if left_ok:
            angles["l_elbow"] = _angle_3d(
                _get_xyz(row, self._L_SHOULDER),
                _get_xyz(row, self._L_ELBOW),
                _get_xyz(row, self._L_WRIST))
            angles["l_shoulder"] = _angle_3d(
                _get_xyz(row, self._L_ELBOW),
                _get_xyz(row, self._L_SHOULDER),
                _get_xyz(row, self._L_HIP))
            # Hip/knee — use if knee+ankle visible, else estimate from hip angle
            l_knee_vis = _get_visibility(row, self._L_KNEE) >= self.visibility_threshold
            l_ankle_vis = _get_visibility(row, self._L_ANKLE) >= self.visibility_threshold
            if l_knee_vis:
                angles["l_hip"] = _angle_3d(
                    _get_xyz(row, self._L_SHOULDER),
                    _get_xyz(row, self._L_HIP),
                    _get_xyz(row, self._L_KNEE))
                if l_ankle_vis:
                    angles["l_knee"] = _angle_3d(
                        _get_xyz(row, self._L_HIP),
                        _get_xyz(row, self._L_KNEE),
                        _get_xyz(row, self._L_ANKLE))

        # Compute right side if visible
        right_ok = _joints_visible(row, [self._R_SHOULDER, self._R_ELBOW,
                                          self._R_WRIST, self._R_HIP],
                                    self.visibility_threshold)
        if right_ok:
            angles["r_elbow"] = _angle_3d(
                _get_xyz(row, self._R_SHOULDER),
                _get_xyz(row, self._R_ELBOW),
                _get_xyz(row, self._R_WRIST))
            angles["r_shoulder"] = _angle_3d(
                _get_xyz(row, self._R_ELBOW),
                _get_xyz(row, self._R_SHOULDER),
                _get_xyz(row, self._R_HIP))
            r_knee_vis = _get_visibility(row, self._R_KNEE) >= self.visibility_threshold
            r_ankle_vis = _get_visibility(row, self._R_ANKLE) >= self.visibility_threshold
            if r_knee_vis:
                angles["r_hip"] = _angle_3d(
                    _get_xyz(row, self._R_SHOULDER),
                    _get_xyz(row, self._R_HIP),
                    _get_xyz(row, self._R_KNEE))
                if r_ankle_vis:
                    angles["r_knee"] = _angle_3d(
                        _get_xyz(row, self._R_HIP),
                        _get_xyz(row, self._R_KNEE),
                        _get_xyz(row, self._R_ANKLE))

        # Mirror missing side from the visible side
        mirror_pairs = [
            ("l_elbow", "r_elbow"), ("l_shoulder", "r_shoulder"),
            ("l_hip", "r_hip"), ("l_knee", "r_knee"),
        ]
        for left_key, right_key in mirror_pairs:
            if left_key in angles and right_key not in angles:
                angles[right_key] = angles[left_key]
            elif right_key in angles and left_key not in angles:
                angles[left_key] = angles[right_key]

        # Default any still-missing angles to safe values (straight)
        for key in ["l_elbow", "r_elbow", "l_shoulder", "r_shoulder",
                    "l_hip", "r_hip", "l_knee", "r_knee"]:
            if key not in angles:
                angles[key] = 170.0  # assume straight

        return angles

    # ── smoothing ──────────────────────────────

    def _smooth_angles(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Push raw angles into buffers and return moving averages."""
        smoothed: Dict[str, float] = {}
        for key, value in raw.items():
            self._angle_buffers[key].append(value)
            buf = self._angle_buffers[key]
            smoothed[key] = float(np.mean(list(buf)))
        return smoothed

    # ── posture validation ─────────────────────

    def is_pushup_posture(
        self,
        row: np.ndarray,
        angles: Dict[str, float],
    ) -> Tuple[bool, Dict[str, bool], list[str]]:
        """
        Check whether the person is in a valid push-up starting posture.

        Criteria:
            1. Body approximately horizontal (shoulder-hip Y-delta small)
            2. Back straight (hip angle within range)
            3. Knees extended
            4. Elbows initially extended (≥ ready threshold)

        Returns
        -------
        (is_valid, details, feedback)
        """
        feedback = []

        # 1. Body horizontal — compare shoulder-centre vs hip-centre Y
        shoulder_y = (
            _get_xyz(row, self._L_SHOULDER)[1]
            + _get_xyz(row, self._R_SHOULDER)[1]
        ) / 2.0
        hip_y = (
            _get_xyz(row, self._L_HIP)[1]
            + _get_xyz(row, self._R_HIP)[1]
        ) / 2.0
        y_delta = abs(shoulder_y - hip_y)
        body_horizontal = y_delta < self.horizontal_y_threshold
        if not body_horizontal:
            feedback.append("Adjust body alignment (Hips too high/low)")

        # 2. Back straight — average hip angle within range
        avg_hip = (angles["l_hip"] + angles["r_hip"]) / 2.0
        back_straight = self.hip_angle_range[0] <= avg_hip <= self.hip_angle_range[1]
        if not back_straight:
            feedback.append("Keep your back straight")

        # 3. Knees extended
        avg_knee = (angles["l_knee"] + angles["r_knee"]) / 2.0
        knees_extended = avg_knee >= self.knee_min_angle
        if not knees_extended:
            feedback.append("Straighten your knees (Knees down)")

        # 4. Elbows extended (ready position check — only matters during
        #    NOT_READY / GATING; once counting starts we skip this check)
        avg_elbow = (angles["l_elbow"] + angles["r_elbow"]) / 2.0
        elbows_ready = avg_elbow >= self.elbow_ready_min
        if not elbows_ready and self._state in (ValidatorState.NOT_READY, ValidatorState.GATING):
            feedback.append("Lock your elbows to start")

        # During active counting (UP/DOWN), elbows will flex — that's fine.
        if self._state in (ValidatorState.UP, ValidatorState.DOWN):
            elbows_ready = True

        details = {
            "body_horizontal": body_horizontal,
            "back_straight": back_straight,
            "knees_extended": knees_extended,
            "elbows_ready": elbows_ready,
        }
        is_valid = all(details.values())

        return is_valid, details, feedback

    # ── gating ─────────────────────────────────

    def _update_gate(self, is_valid: bool) -> bool:
        """
        Accumulate consecutive valid-posture frames.
        Returns True when the gate is open (enough frames accumulated).
        """
        if is_valid:
            self._gate_counter = min(self._gate_counter + 1,
                                     self.gate_frames)
            self._bad_posture_streak = 0
        else:
            self._bad_posture_streak += 1
            if self._bad_posture_streak >= self.posture_loss_tolerance:
                self._gate_counter = 0

        return self._gate_counter >= self.gate_frames

    # ── state machine ──────────────────────────

    def _update_state(
        self,
        avg_elbow: float,
        gate_open: bool,
        is_valid: bool,
        model_confidence: Optional[float],
    ) -> Tuple[bool, Optional[str]]:
        """
        Advance the state machine and return whether a rep was just counted.

        Transitions:
            NOT_READY → GATING  (valid posture detected)
            GATING    → UP      (gate opens)
            UP        → DOWN    (elbow < down threshold)
            DOWN      → UP      (elbow > up threshold — rep counted)
            Any       → NOT_READY (posture lost beyond tolerance)
        """
        rep_counted = False
        coaching_tip = None
        self._frames_since_last_rep += 1

        # ── Reset on prolonged bad posture ─────
        if (not is_valid
                and self._bad_posture_streak >= self.posture_loss_tolerance):
            if self._state != ValidatorState.NOT_READY:
                self._state = ValidatorState.NOT_READY
                self._gate_counter = 0
            return False, None

        # ── State transitions ──────────────────
        if self._state == ValidatorState.NOT_READY:
            if is_valid:
                self._state = ValidatorState.GATING

        elif self._state == ValidatorState.GATING:
            if gate_open:
                self._state = ValidatorState.UP

        elif self._state == ValidatorState.UP:
            if avg_elbow < self.elbow_down_angle:
                self._state = ValidatorState.DOWN
                self._deepest_elbow_this_rep = avg_elbow
            elif avg_elbow < self.elbow_ready_min - 10:
                # Slightly flexed but not yet in DOWN
                pass

        elif self._state == ValidatorState.DOWN:
            self._deepest_elbow_this_rep = min(self._deepest_elbow_this_rep, avg_elbow)
            
            if avg_elbow > self.elbow_up_angle:
                # Check cooldown to prevent jitter-based double signalling
                cooldown_ok = (
                    self._frames_since_last_rep >= self.cooldown_frames
                )
                if cooldown_ok:
                    # Provide feedback on depth if user barely hit the threshold
                    if self._deepest_elbow_this_rep > self.elbow_down_angle - 10:
                        coaching_tip = "Go lower for a full rep!"
                    
                    self._frames_since_last_rep = 0
                    rep_counted = True
                    self._deepest_elbow_this_rep = 180.0
                self._state = ValidatorState.UP
            elif avg_elbow > self.elbow_down_angle + 15:
                # Coming back up without hitting depth or full extension
                pass

        return rep_counted, coaching_tip

    # ── main entry point ───────────────────────

    def process_frame(
        self,
        keypoints_flat: np.ndarray,
        model_confidence: Optional[float] = None,
    ) -> FrameResult:
        """
        Process one frame of raw MediaPipe landmarks.

        Parameters
        ----------
        keypoints_flat : np.ndarray
            Flat (132,) array of [x, y, z, visibility] × 33 landmarks.
        model_confidence : float or None
            LSTM model's "Correct form" probability (0–1).
            Pass None to skip the confidence gate.

        Returns
        -------
        FrameResult with all per-frame information for HUD display.
        """
        result = FrameResult(
            state=self._state,
            gate_required=self.gate_frames,
            rep_count=self._rep_count,
        )

        # ── Step 1: compute angles ─────────────
        raw_angles = self.compute_angles(keypoints_flat)
        if raw_angles is None:
            # Not enough visible joints → treat as invalid
            self._update_gate(False)
            self._update_state(180.0, False, False, model_confidence)
            result.state = self._state
            result.rep_count = self._rep_count
            result.gate_progress = self._gate_counter
            result.feedback = ["Person not fully visible"]
            return result

        # ── Step 2: smooth angles ──────────────
        smoothed = self._smooth_angles(raw_angles)
        result.angles = smoothed

        # ── Step 3: validate posture ───────────
        is_valid, details, posture_feedback = self.is_pushup_posture(keypoints_flat, smoothed)
        result.is_valid_posture = is_valid
        result.posture_details = details
        result.feedback.extend(posture_feedback)

        # ── Step 4: update gating ──────────────
        gate_open = self._update_gate(is_valid)
        result.gate_progress = self._gate_counter

        # ── Step 5: update state machine ───────
        avg_elbow = (smoothed["l_elbow"] + smoothed["r_elbow"]) / 2.0
        rep_counted, state_feedback = self._update_state(
            avg_elbow, gate_open, is_valid, model_confidence,
        )
        # ── Step 6: manage feedback persistency ──
        now = time.time()
        
        # Add new tips to memory
        for tip in posture_feedback:
            self._feedback_memory[tip] = now + self.feedback_ttl
        if state_feedback:
            self._feedback_memory[state_feedback] = now + self.feedback_ttl
            
        # Clean up expired tips
        self._feedback_memory = {
            tip: expiry for tip, expiry in self._feedback_memory.items() 
            if expiry > now
        }
        
        # Populate result
        result.feedback = list(self._feedback_memory.keys())
            
        result.state = self._state
        result.rep_count = self._rep_count
        result.rep_just_counted = rep_counted
        result.rep_completed = rep_counted

        return result
