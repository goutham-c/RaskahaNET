# Disaster Rescue Assistant — Build with Gemma Hackathon (Neonexus 36.1)

**Track:** AI for Humanitarian Technology

## Problem Statement
During earthquakes and cyclones, two things repeatedly cost lives: people don't get
actionable, situation-specific guidance in the critical minutes before/during a disaster,
and rescue teams lose track of where people were last located once communication signals
drop. Generic advice ("stay safe") isn't useful — someone standing near a collapsing wall
needs different instructions than someone trapped under debris. This project addresses
both problems with one system: a Gemma-powered assistant that reads a person's described
situation (and an optional photo) to give specific safety guidance grounded in real
disaster-response protocols, plus a live location-tracking layer so rescue teams always
have the last known coordinates of anyone in danger.

## Solution Overview
The app has three connected parts:
1. **Early warning layer** — pulls live earthquake data from USGS (real-time, public API)
   and displays it on a map so users see nearby risk before/as it happens.
2. **Gemma + RAG safety assistant** — a person describes their situation in plain language
   and can attach a photo. Gemma (via Google AI Studio, multimodal) analyzes the
   description and image, retrieves the most relevant guidance from a vector store seeded
   with earthquake and cyclone safety protocols (RAG), and returns specific, immediate
   steps grounded in that retrieved guidance rather than generic model knowledge.
3. **Location tracking + SOS** — the browser pings GPS coordinates to the backend every
   15 seconds. If a user triggers SOS or loses connectivity, the backend serves their
   **last known coordinates and timestamp** to a rescue-team view, so responders know
   exactly where to go even after signal loss.

## Architecture
```
Browser (Leaflet map + chat UI)
   |
   |-- GPS ping every 15s -----------> FastAPI /ping -----> SQLite (location history)
   |-- SOS button --------------------> FastAPI /sos ------> SQLite (SOS log)
   |-- Chat message + optional image -> FastAPI /analyze
   |                                        |
   |                                        |-- query_rag() -> ChromaDB vector store
   |                                        |                   (seeded disaster protocols)
   |                                        |-- analyze_situation() -> Gemma (Google AI
   |                                        |                   Studio, multimodal)
   |                                        |-- returns grounded, situation-specific advice
   |
   |<-- Live earthquake alerts ---------- FastAPI /alerts/earthquake <---- USGS public feed
```

## Gemma Integration (why Gemma, and how deeply it's used)
Gemma is not a bolt-on chatbot here — it's the reasoning core of the safety-response
pipeline:
- **Vision**: Gemma directly analyzes uploaded photos of the user's surroundings
  (e.g. structural damage, flooding, debris) to help assess severity — not just text input.
- **RAG-grounded generation**: rather than relying on Gemma's general knowledge, every
  response is conditioned on retrieved passages from a curated disaster-protocol vector
  store (ChromaDB + sentence-transformer embeddings), so advice stays accurate and
  source-grounded rather than hallucinated.
- **Disaster-type-aware prompting**: guidance is filtered by disaster type (earthquake vs
  cyclone) at retrieval time, so Gemma is always reasoning over the *relevant* subset of
  protocols, not a generic mixed corpus.
- We chose Gemma over closed alternatives because of its strong multimodal capability
  paired with the option for local/offline inference (via Ollama) in future iterations —
  critical for disaster scenarios where connectivity is often the first thing lost.

## Technical Stack
- **Backend**: Python, FastAPI, SQLite
- **AI**: Gemma 3 (multimodal) via Google AI Studio API, ChromaDB vector store,
  sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- **Frontend**: HTML/JS, Leaflet.js for mapping
- **Live data**: USGS Earthquake API (real-time, public, no key required)

## What's real vs. simulated in this prototype
Built and working: GPS-based live location tracking, real-time earthquake alerts, RAG +
Gemma multimodal chat assistant, SOS with last-known-location lookup.

Out of scope for a hackathon timeframe but architected for: wifi/cell-tower
triangulation, satellite imagery, and thermal/drone detection as additional
`signal_type` sources feeding the same `/ping` endpoint — the location-tracking schema
already supports this extension without redesign.

## Challenges
- Balancing "genuinely useful" advice with the reality that we can't validate real-time
  medical or structural claims — mitigated by grounding all output in vetted RAG sources
  rather than open-ended generation.
- Multi-signal location fusion (satellite/thermal/drone) requires infrastructure access
  outside hackathon scope; we designed the data model to support it without pretending to
  have built it.

## Future Work
- Real wifi/cell-signal triangulation and thermal/drone data ingestion as additional
  location sources
- Multi-language support for regional disaster response
- Offline/local Gemma inference (via Ollama) for zero-connectivity scenarios
- Dedicated rescue-team dashboard view aggregating all active SOS + last-known-locations

## Setup
```
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# add GOOGLE_API_KEY to .env
python rag_setup.py
python -m uvicorn main:app --reload --port 8000

# separate terminal
cd frontend
python -m http.server 5500
# open http://localhost:5500/index.html
```

## Team
Built by goutham & siddesh for Neonexus 36.1 — Build with Gemma Hackathon, IEEE Young
Professionals Bangalore Section.
