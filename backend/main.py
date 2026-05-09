"""RoboMunch FastAPI server.

Endpoints:
    POST /api/generate-image   {prompt}            -> {image: data URL}
    POST /api/chat             {message, history}  -> {reply}
    POST /api/transcribe       multipart audio     -> {text}
"""
import base64
import io
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tasks import ChatTask, ImageGenerationTask, SpeechRecognitionTask


app = FastAPI(title="RoboMunch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

image_task = ImageGenerationTask()
chat_task = ChatTask()
asr_task = SpeechRecognitionTask()


class GenerateImageReq(BaseModel):
    prompt: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/generate-image")
def generate_image(req: GenerateImageReq):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")
    try:
        img = image_task.run(req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"image": f"data:image/png;base64,{b64}"}


@app.post("/api/chat")
def chat(req: ChatReq):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    history = [m.model_dump() for m in (req.history or [])]
    try:
        reply = chat_task.run(req.message, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
    return {"reply": reply}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    data = await audio.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        text = asr_task.run(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return {"text": text}


# Serve the frontend at "/" so a single `uvicorn main:app` runs everything.
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
