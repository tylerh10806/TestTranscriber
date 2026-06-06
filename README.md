# TestTranscribe

Local audio transcription tool. Drop audio files in the browser, get timestamped transcripts back. Runs entirely on CPU — no GPU, no cloud, no data leaves the machine.

## Quick start

```bash
docker run -d -p 127.0.0.1:8000:8000 --restart unless-stopped tylerh10806/transcribe
```

Open **http://localhost:8000** — that's it. The Whisper medium model is baked into the image (~1.5 GB download on first pull).

## Stack

- **Engine**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper medium, int8 quantized
- **Backend**: FastAPI + uvicorn, sequential job queue
- **Frontend**: Single-file vanilla JS UI (dark, no frameworks)
- **Audio conversion**: ffmpeg (16kHz mono WAV before transcription)
- **Runtime**: Docker, bound to `127.0.0.1:8000` only

## Features

- Drag & drop multiple audio files at once
- Per-file status: queued → processing (with % progress) → done / error
- Collapsible timestamped segment view per file
- One-click copy per transcript, or copy with timestamps (formatted as `[MM:SS → MM:SS] text`)
- "Copy all" and "Copy all w/ timestamps" batch buttons
- Supported formats: mp3, m4a, wav, flac, ogg, opus, aac, webm, wma

## Speed

Whisper medium + int8 on CPU runs roughly 1–3× audio length (a 10-min clip ≈ 5–20 min depending on hardware). Files are processed sequentially — one CPU can't usefully parallelize Whisper.

## Building from source

```bash
git clone https://github.com/tylerh10806/TranscribeTest
cd TranscribeTest
docker compose up -d --build
```

## Project layout

```
Dockerfile          # ffmpeg + faster-whisper + model baked in at build time
docker-compose.yml  # single container, localhost only
app/
  main.py           # FastAPI: /transcribe (POST), /jobs/{id} (GET)
static/
  index.html        # self-contained UI
```
