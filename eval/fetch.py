"""
Download test material listed in eval/videos.json into eval/data/.

For each video: a 16kHz mono WAV of the excerpt, and the human-made
(not auto-generated) English and Chinese subtitles, converted to JSON cues
with times relative to the excerpt start.

Usage:  python -m eval.fetch [--only NAME]
"""

import argparse
import glob
import html
import json
import os
import re
import subprocess
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVAL_DIR, "data")
YT = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings"]


def parse_vtt(path: str) -> list[dict]:
    """Parse a WebVTT file into [{start, end, text}] (seconds)."""
    def secs(ts: str) -> float:
        parts = ts.replace(",", ".").split(":")
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))

    cues, cur = [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        m = re.match(r"([\d:.,]+)\s+-->\s+([\d:.,]+)", line)
        if m:
            cur = {"start": secs(m.group(1)), "end": secs(m.group(2)), "text": ""}
            cues.append(cur)
        elif line and cur is not None:
            text = html.unescape(re.sub(r"<[^>]+>", "", line)).replace("\xa0", " ")  # strip inline tags, &nbsp;
            cur["text"] = (cur["text"] + " " + text).strip()
        elif not line:
            cur = None
    return [c for c in cues if c["text"]]


def fetch(video: dict) -> None:
    name, vid = video["name"], video["id"]
    url = f"https://www.youtube.com/watch?v={vid}"
    out_dir = os.path.join(DATA_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    start, dur = video["start"], video["duration"]

    wav = os.path.join(out_dir, "audio.wav")
    if not os.path.exists(wav):
        print(f"[{name}] downloading audio {start}s-{start + dur}s")
        raw = os.path.join(out_dir, "raw.%(ext)s")
        subprocess.run(YT + ["-f", "bestaudio", "--download-sections", f"*{start}-{start + dur}",
                             "-o", raw, url], check=True)
        src = next(p for p in glob.glob(os.path.join(out_dir, "raw.*")))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000",
                        "-sample_fmt", "s16", wav], check=True)
        os.remove(src)

    print(f"[{name}] downloading human subtitles")
    subprocess.run(YT + ["--skip-download", "--write-subs", "--sub-langs", "en.*,zh.*",
                         "--sub-format", "vtt", "-o", os.path.join(out_dir, "subs"), url], check=True)

    subs = {}
    for path in glob.glob(os.path.join(out_dir, "subs.*.vtt")):
        lang = path.split("subs.")[-1][:-len(".vtt")]
        key = "en" if lang.lower().startswith("en") else lang
        cues = [
            {"start": round(c["start"] - start, 2), "end": round(c["end"] - start, 2), "text": c["text"]}
            for c in parse_vtt(path)
            if c["end"] > start and c["start"] < start + dur
        ]
        subs[key] = cues
        print(f"[{name}]   {lang}: {len(cues)} cues in excerpt")
    json.dump(subs, open(os.path.join(out_dir, "reference.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="Only fetch this video name")
    args = parser.parse_args()
    videos = json.load(open(os.path.join(EVAL_DIR, "videos.json")))
    for v in videos:
        if args.only and v["name"] != args.only:
            continue
        fetch(v)


if __name__ == "__main__":
    main()
