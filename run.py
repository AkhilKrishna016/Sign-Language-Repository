"""
EchoSign Application Launcher

Starts the FastAPI server and opens the browser interface.
Run with:
    python run.py
"""

import sys
import os
import webbrowser
import threading
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import uvicorn
from backend.main import app

def open_browser(port: int = 8000):
    time.sleep(2.0)
    url = f"http://localhost:{port}"
    print(f"\n[EchoSign] Opening browser at: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[EchoSign] Could not auto-open browser: {e}")

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"

    print("=" * 65)
    print("  EchoSign - Bidirectional Sign Language Communication System")
    print("=" * 65)
    print(f"  -> Server:     http://{host}:{port}")
    print(f"  -> Prompt Lab: http://{host}:{port}#lab")
    print(f"  -> API Docs:   http://{host}:{port}/docs")
    print("=" * 65)

    # Launch browser in background thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Start FastAPI / Uvicorn Server
    uvicorn.run(app, host=host, port=port, log_level="info")
