import asyncio
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# ── State ─────────────────────────────────────────────────────────────────────
whisper_model = None
job_store: dict[str, dict] = {}
job_queue: asyncio.Queue = asyncio.Queue()

ALLOWED_EXTENSIONS = {
    ".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac",
    ".webm", ".aac", ".opus", ".wma", ".mov",
}
MAX_JOBS = 25
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
    worker = asyncio.create_task(queue_worker())
    yield
    worker.cancel()


app = FastAPI(lifespan=lifespan)


# ── Queue worker ──────────────────────────────────────────────────────────────
async def queue_worker():
    while True:
        job_id = await job_queue.get()
        try:
            await transcribe_job(job_id)
        except Exception as exc:
            job_store[job_id]["status"] = "error"
            job_store[job_id]["error"] = str(exc)
        finally:
            job_queue.task_done()


async def transcribe_job(job_id: str) -> None:
    job = job_store[job_id]
    job["status"] = "processing"
    job["progress"] = 0

    input_path = job["input_path"]
    wav_path = input_path + ".wav"

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg conversion failed")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _do_transcribe, job_id, wav_path)

        job["result"] = result
        job["status"] = "done"
        job["progress"] = 100

    finally:
        for p in (input_path, wav_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def _do_transcribe(job_id: str, wav_path: str) -> dict:
    segments_iter, info = whisper_model.transcribe(
        wav_path, beam_size=5, word_timestamps=False
    )

    segments = []
    transcript_parts = []

    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        transcript_parts.append(seg.text.strip())
        if info.duration > 0:
            job_store[job_id]["progress"] = min(
                int((seg.end / info.duration) * 100), 98
            )

    return {
        "transcript": " ".join(transcript_parts),
        "segments": segments,
        "duration": round(info.duration, 1),
        "language": info.language,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("/static/index.html")


@app.post("/transcribe")
async def transcribe(files: list[UploadFile] = File(...)):
    job_ids = []

    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        # Evict oldest completed job if at capacity
        if len(job_store) >= MAX_JOBS:
            evict_id = next(
                (k for k, v in job_store.items() if v["status"] in ("done", "error")),
                None,
            )
            if evict_id:
                del job_store[evict_id]
            else:
                # All slots are active jobs — skip
                continue

        job_id = str(uuid.uuid4())
        suffix = ext or ".audio"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        written = 0
        too_large = False
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                too_large = True
                break
            tmp.write(chunk)
        tmp.close()

        if too_large:
            os.unlink(tmp.name)
            raise HTTPException(status_code=413, detail=f"{file.filename}: exceeds 2 GB limit")

        job_store[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "status": "queued",
            "progress": 0,
            "input_path": tmp.name,
            "result": None,
            "error": None,
        }

        await job_queue.put(job_id)
        job_ids.append(job_id)

    return {"job_ids": job_ids}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"],
    }
