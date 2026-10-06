from flask import Flask, send_from_directory
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
APP_DIR = BASE / "app"

app = Flask(__name__, static_folder=str(APP_DIR), static_url_path="")

@app.route("/")
def home():
    return send_from_directory(APP_DIR, "index.html")
