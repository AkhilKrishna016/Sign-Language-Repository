"""
OmniSign Server Launcher
Starts the FastAPI server with TFLite Neural Inference and opens the web application.
"""

import sys
import os
import webbrowser
import threading
import time

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import uvicorn
from server import app

def open_browser(port: int = 8000):
    time.sleep(1.8)
    url = f"http://localhost:{port}"
    print(f"\n[OmniSign] Opening browser interface at: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[OmniSign] Could not auto-open browser: {e}")

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    print("=" * 65)
    print("  OMNISIGN - TFLite Neural Object & Gesture Server")
    print("=" * 65)
    print(f"  -> Server Web UI: http://{host}:{port}")
    print(f"  -> Interactive API Docs: http://{host}:{port}/docs")
    print("=" * 65)

    # Launch browser in daemon thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Start FastAPI / Uvicorn Server
    uvicorn.run(app, host=host, port=port, log_level="info")
