import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

# Ensure FAST RETRIEVAL VL DETECTOR is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import app, get_detector

def open_browser(url):
    time.sleep(1.5)
    print(f"\n[VisionLabel AI] Opening browser at {url} ...")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"

    print("==================================================================")
    print("      VisionLabel AI — Zero-Shot VL Image Annotation Studio       ")
    print("==================================================================")
    print(f"Server URL:    {url}")
    print("Framework:     Flask + PyTorch + MobileNetV3 + CLIP")
    print("Task:          Autonomous Object Detection & Annotation across N Images")
    print("==================================================================")

    # Pre-warm detector in background thread for instant response
    def warmup_detector():
        try:
            print("\n[VisionLabel AI] Pre-warming Fast Retrieval VL Model in memory...")
            get_detector()
            print("[VisionLabel AI] Model ready! Instant retrieval enabled.")
        except Exception as e:
            print(f"[VisionLabel AI] Model warmup notice: {e}")

    threading.Thread(target=warmup_detector, daemon=True).start()
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    app.run(host="127.0.0.1", port=port, debug=False)
