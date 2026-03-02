import os
import asyncio
import tempfile
import cv2
import numpy as np
import base64
import struct
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
from app_logic import (
    MoveNetDetector, PushUpSession, download_movenet, MOVENET_MODEL_URL,
    reconstruct_yuv420_to_bgr,
)

app = FastAPI(title="Real-time Push-up Detector")

# Configuration
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_MODEL_FILE = os.path.join(PROJECT_DIR, "pushup_lstm_model.h5")
MOVENET_MODEL_FILE = os.path.join(PROJECT_DIR, "movenet_thunder.tflite")

# Global models (loaded once)
print("[INFO] Loading models...")
if not os.path.exists(LSTM_MODEL_FILE):
    raise FileNotFoundError(f"LSTM model not found at {LSTM_MODEL_FILE}")

download_movenet(MOVENET_MODEL_URL, MOVENET_MODEL_FILE)
detector = MoveNetDetector(MOVENET_MODEL_FILE)
lstm_model = tf.keras.models.load_model(LSTM_MODEL_FILE)
print("[INFO] Models loaded successfully.")

# Support for static files (the UI)
if not os.path.exists("static"):
    os.makedirs("static")

@app.get("/")
async def get():
    with open(os.path.join("static", "index.html"), "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = PushUpSession(lstm_model, detector)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            # Data is expected to be a base64 encoded image string
            # Format: "data:image/jpeg;base64,..."
            if "," in data:
                header, encoded = data.split(",", 1)
            else:
                encoded = data
            
            binary_data = base64.b64decode(encoded)
            nparr = np.frombuffer(binary_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # Process frame
            result = session.process_frame(frame)
            
            # Send result back
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("[INFO] Client disconnected.")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/ws/android")
async def websocket_android_endpoint(websocket: WebSocket):
    """
    Binary WebSocket endpoint for Android.
    Receives raw YUV420 frames (24-byte header + Y/U/V planes),
    reconstructs BGR via OpenCV, then runs the inference pipeline.
    """
    await websocket.accept()
    session = PushUpSession(lstm_model, detector)
    print("[INFO] Android client connected to /ws/android")

    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"[DEBUG] Received frame: {len(data)} bytes")

            try:
                bgr_frame = reconstruct_yuv420_to_bgr(data)
                print(f"[DEBUG] Reconstructed BGR: {bgr_frame.shape}")
            except Exception as e:
                print(f"[WARN] YUV reconstruction failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            result = session.process_frame(bgr_frame)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("[INFO] Android client disconnected.")
    except Exception as e:
        print(f"[ERROR] Android WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    """Receive a video file as binary, process frames, stream results back."""
    await websocket.accept()
    session = PushUpSession(lstm_model, detector)
    tmp_path = None

    try:
        # 1. Receive the entire video file as binary
        print("[INFO] Video WebSocket: waiting for video data...")
        video_bytes = await websocket.receive_bytes()
        print(f"[INFO] Received video: {len(video_bytes)} bytes")

        # 2. Save to a temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)

        # 3. Open with OpenCV
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            await websocket.send_json({"error": "Could not open video file"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Process at ~10 FPS — skip frames if video FPS is higher
        skip = max(1, int(fps / 10))
        frame_delay = 1.0 / min(fps, 10.0)

        print(f"[INFO] Video: {total_frames} frames @ {fps:.1f} FPS, processing every {skip} frame(s)")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip != 0:
                continue

            # Process the frame through the same pipeline
            result = session.process_frame(frame)
            result["frame_index"] = frame_idx
            result["total_frames"] = total_frames

            # Encode frame to JPEG and then base64 to send back to client
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            result["image"] = f"data:image/jpeg;base64,{frame_base64}"

            await websocket.send_json(result)
            await asyncio.sleep(frame_delay)

        cap.release()

        # 4. Signal completion
        await websocket.send_json({"status": "done", "total_reps": session.real_rep_count})
        print(f"[INFO] Video processing done. Reps: {session.real_rep_count}")

    except WebSocketDisconnect:
        print("[INFO] Video client disconnected.")
    except Exception as e:
        print(f"[ERROR] Video WebSocket error: {e}")
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

