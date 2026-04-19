from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
import subprocess
import threading
import sys

app = Flask(__name__, static_folder=".")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES_PATH = os.path.join(BASE_DIR, "sources.json")
VECTORIZE_PATH = os.path.join(BASE_DIR, "Vectorize.py")

vectorize_status = {"running": False, "log": [], "success": None}


def load_sources():
    if not os.path.exists(SOURCES_PATH):
        return []
    with open(SOURCES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_sources(sources):
    with open(SOURCES_PATH, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=4)


@app.route("/api/sources", methods=["GET"])
def get_sources():
    return jsonify(load_sources())


@app.route("/api/sources", methods=["POST"])
def add_source():
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URL is required"}), 400

    sources = load_sources()

    if url in sources:
        return jsonify({"error": "URL already exists"}), 409

    sources.append(url)
    save_sources(sources)
    return jsonify({"success": True, "sources": sources})


@app.route("/api/sources", methods=["DELETE"])
def remove_source():
    data = request.get_json()
    url = data.get("url", "").strip()

    sources = load_sources()

    if url not in sources:
        return jsonify({"error": "URL not found"}), 404

    sources.remove(url)
    save_sources(sources)
    return jsonify({"success": True, "sources": sources})


def run_vectorize():
    vectorize_status["running"] = True
    vectorize_status["log"] = []
    vectorize_status["success"] = None

    try:
        process = subprocess.Popen(
            [sys.executable, VECTORIZE_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=BASE_DIR
        )

        for line in process.stdout:
            vectorize_status["log"].append(line.strip())

        process.wait()
        vectorize_status["success"] = process.returncode == 0

    except Exception as e:
        vectorize_status["log"].append(f"Error: {e}")
        vectorize_status["success"] = False

    vectorize_status["running"] = False


@app.route("/api/vectorize", methods=["POST"])
def trigger_vectorize():
    if vectorize_status["running"]:
        return jsonify({"error": "Vectorization already running"}), 409

    thread = threading.Thread(target=run_vectorize, daemon=True)
    thread.start()
    return jsonify({"success": True, "message": "Vectorization started"})


@app.route("/api/vectorize/status", methods=["GET"])
def get_vectorize_status():
    return jsonify(vectorize_status)


@app.route("/")
def serve_admin():
    return send_from_directory(BASE_DIR, "Admin.html")


if __name__ == "__main__":
    app.run(debug=True, port=5050)