"""
rep_counter.py
==============
Reusable, real-time exercise repetition counter for Python fitness apps
using MediaPipe Pose + OpenCV.

Handles fast movements by combining:
  1. Two-state machine (UP ↔ DOWN) — counts only on full DOWN → UP cycle
  2. Moving-average angle smoothing (configurable window, default 5 frames)
  3. Landmark visibility gating (confidence > 0.7)
  4. Post-rep cooldown (default 300 ms) to prevent double-counting
  5. Lightweight design — only arithmetic per frame, no heavy allocations

Usage:
    counter = RepCounter(
        down_angle_threshold=90.0,   # e.g. knee < 90° = squat bottom
        up_angle_threshold=160.0,    # e.g. knee > 160° = standing
    )

    # Inside your pose callback / video loop:
    rep_just_counted = counter.update(knee_angle, landmark_confidence)
    print(f"Reps: {counter.rep_count}")
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum, auto
from typing import Optional


# ──────────────────────────────────────────────
# Movement State
# ──────────────────────────────────────────────

class MovementState(Enum):
    """Two phases of an exercise repetition."""
    UP   = auto()  # Extended / starting position (standing, arms extended, etc.)
    DOWN = auto()  # Contracted / bottom position (squat bottom, chest near floor, etc.)


# ──────────────────────────────────────────────
# RepCounter
# ──────────────────────────────────────────────

class RepCounter:
    """
    Counts exercise repetitions using a state-machine approach with
    angle smoothing, confidence gating, and cooldown.

    A rep is counted ONLY when ALL of the following are true:
        • The joint angle crosses from DOWN → UP (full range of motion)
        • Landmark confidence exceeds `min_confidence`
        • At least `cooldown_ms` milliseconds have passed since the last rep

    Parameters
    ----------
    down_angle_threshold : float
        Joint angle (degrees) at or below which the exercise is in the
        DOWN state. Example: 90° for squat knee angle at bottom.
    up_angle_threshold : float
        Joint angle (degrees) at or above which the exercise is in the
        UP state. Example: 160° for squat knee angle when standing.
    min_confidence : float
        Minimum MediaPipe landmark visibility required to process the
        frame. Frames below this are silently skipped — no state change,
        no counting. Default 0.7.
    cooldown_ms : int
        Milliseconds to wait after counting a rep before allowing the
        next one. Prevents double-counting when the user bounces at
        the top position. Default 300.
    smoothing_window : int
        Number of recent angle values used for the moving average.
        3–5 is ideal for 30 fps camera streams. Default 5.
    """

    def __init__(
        self,
        down_angle_threshold: float,
        up_angle_threshold: float,
        min_confidence: float = 0.7,
        cooldown_ms: int = 300,
        smoothing_window: int = 5,
    ):
        assert down_angle_threshold < up_angle_threshold, (
            "down_angle_threshold must be < up_angle_threshold "
            "(hysteresis band required)"
        )

        # Configuration
        self.down_angle_threshold = down_angle_threshold
        self.up_angle_threshold = up_angle_threshold
        self.min_confidence = min_confidence
        self.cooldown_ms = cooldown_ms
        self.smoothing_window = smoothing_window

        # Internal state
        self._state = MovementState.UP
        self._rep_count = 0
        self._last_rep_time: float = 0.0  # seconds (time.time())
        self._angle_buffer: deque[float] = deque(maxlen=smoothing_window)

    # ── Public properties ──────────────────────

    @property
    def rep_count(self) -> int:
        """Running total of valid repetitions."""
        return self._rep_count

    @property
    def state(self) -> MovementState:
        """Current movement state (UP or DOWN)."""
        return self._state

    @property
    def smoothed_angle(self) -> Optional[float]:
        """Most recent smoothed angle, or None if no frames processed."""
        if not self._angle_buffer:
            return None
        return sum(self._angle_buffer) / len(self._angle_buffer)

    # ── Main update ────────────────────────────

    def update(self, raw_angle: float, confidence: float) -> bool:
        """
        Process one frame's joint angle and landmark confidence.

        Call this exactly once per frame from your video loop.
        Returns True if a new rep was counted on THIS frame.

        Parameters
        ----------
        raw_angle : float
            The joint angle in degrees (e.g., knee angle for squats,
            elbow angle for push-ups).
        confidence : float
            MediaPipe landmark visibility (0.0–1.0). Pass the MINIMUM
            visibility across the joints used to compute `raw_angle`
            for maximum safety.
        """
        # ── Gate 1: Confidence check ───────────
        # Skip unreliable frames entirely. This prevents phantom reps
        # when the person is partially occluded or moving too fast for
        # MediaPipe to track accurately.
        if confidence < self.min_confidence:
            return False

        # ── Gate 2: Smooth the angle ───────────
        # Push the new value into the rolling buffer. The deque's
        # maxlen automatically drops the oldest value.
        self._angle_buffer.append(raw_angle)

        # Compute the moving average. This fills in gaps when a fast
        # movement causes one or two frames to be slightly off,
        # ensuring the state machine sees a continuous trajectory
        # instead of jumpy readings.
        angle = sum(self._angle_buffer) / len(self._angle_buffer)

        # ── Gate 3: State machine transition ───
        # We use hysteresis (two separate thresholds) so that noise
        # near a single threshold can't cause rapid UP↔DOWN flicker.
        #
        #   UP state:   angle must drop BELOW down_angle_threshold → DOWN
        #   DOWN state: angle must rise ABOVE up_angle_threshold   → UP
        #
        # A rep is counted ONLY on the DOWN → UP transition, meaning
        # the user must complete the FULL range of motion.

        rep_counted = False

        if self._state == MovementState.UP:
            # Waiting for the user to go down.
            if angle <= self.down_angle_threshold:
                self._state = MovementState.DOWN

        elif self._state == MovementState.DOWN:
            # Waiting for the user to come back up.
            if angle >= self.up_angle_threshold:
                # ── Gate 4: Cooldown check ─────
                # Prevent double-counting if the user bounces at top.
                now = time.time()
                elapsed_ms = (now - self._last_rep_time) * 1000.0
                cooldown_ok = elapsed_ms >= self.cooldown_ms

                if cooldown_ok:
                    self._rep_count += 1
                    self._last_rep_time = now
                    rep_counted = True

                # Transition back to UP regardless of cooldown,
                # so the state machine stays in sync with the body.
                self._state = MovementState.UP

        return rep_counted

    def reset(self) -> None:
        """Reset the counter and state to initial values.
        Call this when starting a new set or exercise."""
        self._rep_count = 0
        self._state = MovementState.UP
        self._last_rep_time = 0.0
        self._angle_buffer.clear()


# ──────────────────────────────────────────────
# Example: integration with MediaPipe + OpenCV
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    import numpy as np
    import mediapipe as mp

    # ── Helper: compute angle at joint b (a→b→c) ──
    def angle_3d(a, b, c) -> float:
        """Angle in degrees at point b formed by segments a→b and c→b."""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    # ── Helper: get (x, y, z) from a MediaPipe landmark ──
    def lm_xyz(landmark):
        return [landmark.x, landmark.y, landmark.z]

    # ── Initialise MediaPipe Pose ──────────────
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # ── Create the RepCounter ──────────────────
    # Example: squat counter using knee angle
    # DOWN = knee bent below 100°, UP = knee extended above 160°
    counter = RepCounter(
        down_angle_threshold=100.0,
        up_angle_threshold=160.0,
        min_confidence=0.7,
        cooldown_ms=300,
        smoothing_window=5,
    )

    # ── Open webcam ────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[INFO] RepCounter demo — Squat counter")
    print("[INFO] Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ── Compute knee angle (right side) ──
            # landmarks: 24=R_HIP, 26=R_KNEE, 28=R_ANKLE
            hip   = lm[24]
            knee  = lm[26]
            ankle = lm[28]

            knee_angle = angle_3d(
                lm_xyz(hip), lm_xyz(knee), lm_xyz(ankle))

            # Use the minimum visibility across the 3 joints
            min_vis = min(hip.visibility, knee.visibility, ankle.visibility)

            # ── Update the counter ─────────────
            rep_just_counted = counter.update(knee_angle, min_vis)

            if rep_just_counted:
                print(f"  ✓ Rep #{counter.rep_count}")

            # ── Draw HUD ──────────────────────
            # Background panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (350, 160), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

            state_color = (0, 220, 0) if counter.state == MovementState.UP else (0, 180, 255)
            smoothed = counter.smoothed_angle or 0.0

            cv2.putText(frame, f"Reps: {counter.rep_count}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"State: {counter.state.name}", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            cv2.putText(frame, f"Knee: {smoothed:.0f} deg", (15, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Conf: {min_vis:.0%}", (15, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("RepCounter Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Total reps: {counter.rep_count}")
