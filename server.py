#!/usr/bin/env python3
import subprocess
import os

if __name__ == "__main__":
    port = os.environ.get("PORT", "8000")
    subprocess.run([
        "streamlit", "run", "app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false"
    ])