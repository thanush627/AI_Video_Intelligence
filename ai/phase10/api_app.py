import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Video Intelligence API", version="0.1.0")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/upload")
def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    destination = UPLOAD_DIR / file.filename
    with destination.open("wb") as f:
        f.write(file.file.read())
    return {"filename": file.filename, "path": str(destination)}


@app.post("/search")
def search(query: str) -> Dict[str, Any]:
    return {"query": query, "results": []}


@app.post("/voice")
def voice_query(file: Optional[UploadFile] = File(None), query: Optional[str] = None) -> Dict[str, Any]:
    if file is None and not query:
        raise HTTPException(status_code=400, detail="Provide audio file or text query")
    return {"query": query or "audio_received", "results": []}


@app.get("/clips/{event_id}")
def get_clip(event_id: str) -> JSONResponse:
    clip_path = UPLOAD_DIR / "clips" / f"{event_id}.mp4"
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="clip not found")
    return JSONResponse(content={"event_id": event_id, "path": str(clip_path)})
