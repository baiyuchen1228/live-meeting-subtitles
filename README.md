


https://github.com/user-attachments/assets/f74c706b-0963-4978-afc6-3137943b2e62




# Live Meeting Subtitles

Real-time voice transcription and translation for video calls (Zoom, Google Meet, Teams, etc.)

Captures system audio, transcribes speech using Parakeet or Whisper, and displays subtitles in an overlay window: the English original with a Traditional Chinese translation underneath (the target language is configurable).

## Architecture

```
┌──────────────────────┐     WebSocket      ┌────────────────────────────┐
│   Windows Client     │  ───────────────►  │      WSL Server            │
│                      │                    │                            │
│  • WASAPI capture    │    audio bytes     │  • Parakeet / Whisper      │
│  • PyQt6 overlay     │  ◄───────────────  │  • Opus-MT / NLLB / DeepL  │
│                      │    JSON results    │    translation (CUDA)      │
└──────────────────────┘                    └────────────────────────────┘
```

**Why hybrid?**
- WSL has native CUDA support for fast GPU inference
- Windows has native WASAPI for system audio capture
- WebSocket connects them seamlessly

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (RTX 20xx or newer recommended)
- 8GB+ VRAM for large-v3 model (4GB+ for smaller models)

### Software
- Windows 11 with WSL2 (Ubuntu 22.04/24.04)
- Python 3.11+ (both Windows and WSL)
- NVIDIA drivers with CUDA support in WSL

## Installation

### 1. Clone the repository

```bash
# In WSL
cd ~/projects
git clone https://github.com/holydrug/live-meeting-subtitles.git
cd live-meeting-subtitles
```

### 2. Setup WSL Server

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# RTX 50 series (Blackwell) needs PyTorch nightly for Parakeet
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
pip install -r server/requirements.txt


# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Setup Windows Client

Open PowerShell and navigate to the project:

```powershell
# Navigate to project (via WSL path)
cd \\wsl.localhost\Ubuntu\home\YOUR_USERNAME\projects\live-meeting-subtitles

# Create virtual environment
python -m venv venv_win

# Allow the unsigned activate script in this window
# (scripts on \\wsl.localhost are treated as remote files)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate
.\venv_win\Scripts\Activate.ps1

# Install dependencies
pip install -r client/requirements.txt
```

## Usage

Start the server first, then the client.

### 1. Start the server (WSL terminal)

```bash
cd ~/projects/live-meeting-subtitles
source venv/bin/activate
./server.sh -t parakeet --translator opus
```

Use `./server.sh -i` to pick the transcriber and translator from a menu, and `./server.sh --stop` to stop it.
Models are downloaded on first run; the server is ready when it prints `Starting server on ws://0.0.0.0:9876`.

| Option | Values |
|---|---|
| `-t`, `--transcriber` | `parakeet` (faster, recommended), `whisper` |
| `--translator` | `opus` (Opus-MT en-zh on GPU + OpenCC, default), `local` (NLLB, often drops parts of sentences), `deepl` (needs `DEEPL_API_KEY`), `google` (free, may be blocked), `none` |
| `--port` | Server port (default `9876`) |
| `TARGET_LANG` env var | Target language (default `ZH-TW`), e.g. `TARGET_LANG=JA ./server.sh` |

The default translator (`opus`) translates English to `ZH-TW` (Traditional Chinese) or `ZH` (Simplified) only. For other targets (`JA`, `KO`, `RU`, `DE`, `FR`, `ES`, `IT`, `PT`, `UK`, `PL`) use `--translator local` or `deepl`.

Running `python -m server.main` directly also works; see `python -m server.main --help` for all options (`--model`, `--device`, `--target-lang`, `--source-lang`).

### 2. Start the client (PowerShell)

```powershell
cd \\wsl.localhost\Ubuntu\home\YOUR_USERNAME\projects\live-meeting-subtitles
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\client.ps1
```

Pick the audio device you hear the meeting on (speakers or headset). The client captures what that device plays (WASAPI loopback), so the other side of the call is transcribed, not your microphone.

| Command | What it does |
|---|---|
| `.\client.ps1 -ListDevices` | List loopback audio devices |
| `.\client.ps1 -Device "Headset"` | Use a device directly (partial name match) |
| `.\client.ps1 -NoOverlay` | Print subtitles in the console only |
| `.\client.ps1 -HostName 172.x.x.x` | Connect to a non-localhost server (WSL IP from `hostname -I`) |

### 3. Join the call

The overlay shows each sentence as it is recognized:
- **Gray**: live preview while the person is still speaking (updates about every 0.8 s)
- **White**: final English original, replaces the preview when the sentence ends
- **Blue**: translation (dimmed while it is still the preview's translation)

- **Move**: drag inside the box
- **Resize**: drag an edge or corner of the box. After a manual resize the height stays fixed: the newest subtitle sits at the bottom and older ones are pushed out at the top
- **Automatic height again**: double-click the box
- **Close**: click `×`

Size and position are remembered for the next run (Windows registry, `HKEY_CURRENT_USER\Software\live-meeting-subtitles\overlay`).

Tip: test with an English YouTube video before a real meeting.

### How the latency works

The client sends audio in 0.2 s chunks. The server uses Silero VAD to detect when the speaker pauses: a sentence is finalized 0.5 s after speech ends, or after 8 s of continuous speech (cut at a gap between words). While someone is speaking, the server sends a live preview of the unfinished sentence about every 0.8 s, so text appears about 1 s after speech starts. Final text is sent immediately and its translation follows in a separate message.

Parakeet runs the model directly on in-memory audio (no temp files or per-call dataloader), which takes about 60-110 ms per sentence on an RTX 4060.

## Configuration

Defaults are set by command-line options (see above). `config.yaml` documents the intended settings but is not loaded by the code yet.

## Troubleshooting

### "No loopback device found"
Make sure you have audio playing through the selected device. WASAPI loopback requires active audio output.

### CUDA not available in WSL
1. Update NVIDIA drivers on Windows (535+ recommended)
2. Ensure WSL2 is updated: `wsl --update`
3. Check with: `nvidia-smi` in WSL

### Connection refused
1. Check server is running on port 9876
2. Windows Firewall may block WSL connections — add exception

### `Activate.ps1 ... is not digitally signed`
PowerShell blocks unsigned scripts from `\\wsl.localhost`. Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window first, or call `.\venv_win\Scripts\python.exe` directly.

### `DLL load failed while importing QtWidgets`
Anaconda-based Python ships an older `msvcp140.dll` than PyQt6 6.8+ needs. `client/requirements.txt` pins `PyQt6<6.8`; reinstall with `pip install -r client/requirements.txt`.

### "No audio devices found"
The client needs `PyAudioWPatch` (not plain `pyaudio`): `pip install -r client/requirements.txt`.

### `Library libcublas.so.12 is not found` (Whisper)
PyTorch nightly ships CUDA 13, but faster-whisper's ctranslate2 needs CUDA 12 cuBLAS: `pip install nvidia-cublas-cu12` (included in `server/requirements.txt`). Parakeet does not need it.

### `Google Translate error: ... too many requests`
Google blocks some IPs (common on campus or shared networks) regardless of request rate. Use `--translator opus` (default) or `--translator deepl`.

### `You are sending unauthenticated requests to the HF Hub`
Harmless. Once models are downloaded, `export HF_HUB_OFFLINE=1` skips the online check and starts faster.

### High latency
- Use `-t parakeet` (faster than Whisper)
- Check the server log: the first sentence after startup should not be slow (models are warmed up at startup)
- Ensure GPU is not throttling (check temps)

## Project Structure

```
live-meeting-subtitles/
├── server/                 # WSL (Python + CUDA)
│   ├── main.py            # WebSocket server (VAD-based segmentation)
│   ├── transcribers/      # Parakeet / Whisper backends
│   └── translators/       # Translation backends
│       ├── google.py      # Google Translate (free)
│       ├── deepl.py       # DeepL API
│       ├── local.py       # NLLB local model
│       └── opus.py        # Opus-MT en-zh (default)
│
├── client/                 # Windows (Python)
│   ├── main.py            # WebSocket client + UI
│   ├── audio_capture.py   # WASAPI loopback
│   └── overlay.py         # PyQt6 overlay window
│
├── shared/                 # Common code
│   └── protocol.py        # WebSocket protocol
│
├── server.sh / server.ps1  # Server launchers
├── client.ps1 / client.sh  # Client launchers
└── config.yaml            # Intended defaults (not loaded yet)
```

## License

MIT
