"""
Stream each eval/data/<name>/audio.wav to the server in real time, like the
Windows client does, and record every message with its arrival time
(seconds from the start of the audio).

Results go to eval/results/<label>/<name>.json; score them with eval.score.

Usage:
  python -m eval.run --spawn                 # start a server on port 9877 for the run
  python -m eval.run --port 9876             # use a server that is already running
  python -m eval.run --spawn --label nllb-600m --server-args "--translator local"
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import wave

import numpy as np
import websockets

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EVAL_DIR)
DATA_DIR = os.path.join(EVAL_DIR, "data")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
SAMPLE_RATE = 16000


def git_label() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, text=True).strip()
    except Exception:
        return "run"


async def stream(path: str, port: int, chunk_s: float, tail_s: float = 3.0) -> list[dict]:
    with wave.open(path) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = np.concatenate([audio, np.zeros(int(SAMPLE_RATE * tail_s), dtype=np.int16)])
    chunk = int(SAMPLE_RATE * chunk_s)
    events: list[dict] = []

    async with websockets.connect(f"ws://localhost:{port}", max_size=None) as ws:
        await ws.recv()  # ready status
        t0 = time.time()

        async def receive():
            async for message in ws:
                data = json.loads(message)
                data["t"] = round(time.time() - t0, 3)
                events.append(data)

        receiver = asyncio.create_task(receive())
        for i in range(0, len(audio), chunk):
            # A real client can only send a chunk once it has been recorded
            await asyncio.sleep(max(0.0, t0 + (i + chunk) / SAMPLE_RATE - time.time()))
            await ws.send(audio[i:i + chunk].tobytes())
        await asyncio.sleep(2)
        receiver.cancel()
    return events


def spawn_server(port: int, extra: str, log_path: str) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "server.main", "--port", str(port)] + extra.split()
    # Unbuffered so "Starting server" reaches the log while we wait for it
    env = dict(os.environ, HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"), PYTHONUNBUFFERED="1")
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(300):
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        if "Starting server" in open(log_path, errors="replace").read():
            return proc
    proc.terminate()
    raise RuntimeError(f"server did not start in 300s, see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=9877)
    parser.add_argument("--spawn", action="store_true", help="Start a server for this run")
    parser.add_argument("--server-args", default="", help="Extra server.main arguments when spawning")
    parser.add_argument("--label", default=None, help="Results folder name (default: git commit)")
    parser.add_argument("--chunk", type=float, default=0.2, help="Seconds per audio chunk (client default 0.2)")
    parser.add_argument("--only", nargs="*", help="Only these video names")
    args = parser.parse_args()

    label = args.label or git_label()
    out_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    videos = [v for v in json.load(open(os.path.join(EVAL_DIR, "videos.json")))
              if not args.only or v["name"] in args.only]

    proc = spawn_server(args.port, args.server_args, os.path.join(out_dir, "server.log")) if args.spawn else None
    try:
        for v in videos:
            path = os.path.join(DATA_DIR, v["name"], "audio.wav")
            if not os.path.exists(path):
                print(f"[{v['name']}] missing audio, run: python -m eval.fetch")
                continue
            print(f"[{v['name']}] streaming {v['duration']}s in real time...", flush=True)
            events = asyncio.run(stream(path, args.port, args.chunk))
            json.dump({"video": v, "chunk": args.chunk, "server_args": args.server_args, "events": events},
                      open(os.path.join(out_dir, f"{v['name']}.json"), "w", encoding="utf-8"),
                      ensure_ascii=False)
            print(f"[{v['name']}] {sum(e['type'] == 'result' for e in events)} sentences", flush=True)
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=30)
    print(f"Results: {out_dir}\nScore with: python -m eval.score {label}")


if __name__ == "__main__":
    main()
