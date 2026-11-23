# ClipAI — AI Shorts Maker 🎬⚡

ClipAI is an automated AI-powered tool that converts long-form YouTube videos into engaging, vertical 9:16 Shorts with AI Highlight Detection and Auto Captions.

Built to run locally on consumer hardware (tested on CPU-only Windows systems with 8GB RAM, no NVIDIA GPU required).

---

## ✨ Features

- **YouTube Downloader**: High-reliability download pipeline powered by `yt-dlp` with IPv4 routing, system Node.js JavaScript runtime integration for client challenges, per-video thread-safe locking, and granular error reporting.
- **AI Highlight Detection**: Automatic dialogue and hook detection using `faster-whisper` (Base model, INT8 CPU quantization, 4 threads). Heuristic scoring identifies candidate 20–30s viral windows for English, Hindi, and Punjabi speech.
- **Auto Captions / Subtitles**: Automatically extracts and synchronizes snappy 3–5 word subtitle cards formatted in ASS format using Windows native `Nirmala UI` font (full support for English, Hindi/Devanagari, and Punjabi/Gurmukhi).
- **Single-Pass 9:16 FFmpeg Rendering**: Creates vertical 1080x1920 video with blurred letterbox background and burns subtitles in a single filtergraph pass without secondary re-encoding.
- **Sequential Background Queue**: In-memory background task manager processes one FFmpeg job at a time to prevent CPU/RAM exhaustion, reporting real-time progress (`analyzing_highlights` -> `processing` -> `completed`).
- **Interactive Web UI**: Modern React + Vite frontend with real-time progress polling, responsive video previews, and one-click Short downloads.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, yt-dlp, faster-whisper, FFmpeg
- **Frontend**: React 19, Vite, Vanilla CSS
- **Media Engine**: FFmpeg with `libx264`, `boxblur`, and `ass` filters

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: 3.10+ (tested on 3.11)
- **Node.js**: v18+ (tested on v22)
- **FFmpeg**: Must be installed and available on your system `PATH` (both `ffmpeg` and `ffprobe`)

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
The FastAPI backend will start at `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The frontend will start at `http://localhost:5173`.

---

## 🧪 Running Tests

### Backend Unit Tests
```bash
# From repository root:
backend\venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py"
```

### Frontend Lint & Build
```bash
npm --prefix frontend run lint
npm --prefix frontend run build
```

---

## 🛡️ Privacy & Compliance
- ClipAI downloads publicly accessible videos for transformation and clipping.
- No user credentials, cookies, or authentication details are required, stored, or transmitted.
- Downloaded and processed media are kept local on disk inside `backend/downloads/` and `backend/processed/` and are excluded from Git version control.

---

## 📄 License
MIT License
