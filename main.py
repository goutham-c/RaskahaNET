import os
import sqlite3
import time
import shutil

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_setup import query_rag, seed_default_docs
from gemma_assistant import analyze_situation, text_only_chat

app = FastAPI(title="Disaster Rescue API")

# Allow the frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "locations.db"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pings (user_id TEXT, lat REAL, lng REAL, ts REAL, signal_type TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sos (user_id TEXT, lat REAL, lng REAL, ts REAL, message TEXT)"
    )
    conn.commit()
    conn.close()


@app.on_event("startup")
def startup():
    init_db()
    # Seed RAG store with default earthquake + cyclone guidance if empty
    try:
        if not query_rag("test", "earthquake", n_results=1):
            seed_default_docs()
    except Exception as e:
        print("Seed check skipped:", e)


# ---------- LOCATION TRACKING ----------

class LocationPing(BaseModel):
    user_id: str
    lat: float
    lng: float
    signal_type: str = "gps"  # gps, wifi, cell, satellite (mocked), thermal (mocked)


@app.post("/ping")
def ping_location(loc: LocationPing):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO pings VALUES (?,?,?,?,?)",
        (loc.user_id, loc.lat, loc.lng, time.time(), loc.signal_type),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/last-known/{user_id}")
def last_known(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT lat, lng, ts, signal_type FROM pings WHERE user_id=? ORDER BY ts DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    conn.close()
    if not row:
        return {"error": "no data"}
    return {"lat": row[0], "lng": row[1], "last_seen": row[2], "signal_type": row[3]}


@app.get("/all-active-users")
def all_active_users():
    """Returns the latest ping per user - useful for showing everyone on the map."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT user_id, lat, lng, MAX(ts), signal_type FROM pings GROUP BY user_id"""
    ).fetchall()
    conn.close()
    return [
        {"user_id": r[0], "lat": r[1], "lng": r[2], "last_seen": r[3], "signal_type": r[4]}
        for r in rows
    ]


# ---------- SOS ----------

class SOSRequest(BaseModel):
    user_id: str
    lat: float
    lng: float
    message: str = ""


@app.post("/sos")
def trigger_sos(sos: SOSRequest):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sos VALUES (?,?,?,?,?)",
        (sos.user_id, sos.lat, sos.lng, time.time(), sos.message),
    )
    conn.commit()
    conn.close()
    return {"status": "sos_received", "note": "Last known location will be used if signal is lost."}


@app.get("/active-sos")
def active_sos():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT user_id, lat, lng, ts, message FROM sos ORDER BY ts DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return [
        {"user_id": r[0], "lat": r[1], "lng": r[2], "ts": r[3], "message": r[4]}
        for r in rows
    ]


# ---------- CHAT ASSISTANT (RAG + GEMMA) ----------

@app.post("/analyze")
async def analyze(
    disaster_type: str = Form(...),
    message: str = Form(...),
    image: UploadFile = File(None),
):
    context = query_rag(message, disaster_type)

    image_path = None
    if image is not None:
        image_path = os.path.join(UPLOAD_DIR, image.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

    if image_path:
        result = analyze_situation(image_path, message, disaster_type, context)
    else:
        result = text_only_chat(message, disaster_type, context)

    return {"response": result, "context_used": context}


# ---------- EARLY WARNING / ALERTS ----------

@app.get("/alerts/earthquake")
def earthquake_alerts(min_magnitude: float = 4.5):
    """Live feed from USGS - real data, no API key required."""
    url = (
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        quakes = []
        for f in data.get("features", []):
            props = f["properties"]
            coords = f["geometry"]["coordinates"]
            quakes.append(
                {
                    "place": props.get("place"),
                    "magnitude": props.get("mag"),
                    "time": props.get("time"),
                    "lng": coords[0],
                    "lat": coords[1],
                }
            )
        return {"count": len(quakes), "quakes": quakes}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"status": "Disaster Rescue API is running"}
