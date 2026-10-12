"""
Score a run from eval.run against the human subtitles.

English accuracy:  word error rate (WER) of the final English text vs the
                   human English subtitles of the excerpt.
Latency:           Silero VAD splits the audio into utterances (>= 0.5s pauses).
                   first-text lag = first message after an utterance starts;
                   final lag      = first final result after it ends.
Translation:       chrF (0-100) vs human Chinese subtitles, when the video has
                   them (Simplified is converted to Traditional with OpenCC);
                   for every video, the share of sentences whose translation
                   is suspiciously short (likely dropped content).

Usage:  python -m eval.score LABEL [--samples 8]
"""

import argparse
import json
import os
import re
import statistics
import wave

import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVAL_DIR, "data")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# A translation with fewer than this many Chinese characters per English word
# is counted as "short". Human Chinese subtitles of these videos run ~1.3-1.6.
SHORT_RATIO = 0.8


# Spelling variants that are not recognition errors (human subtitles and ASR differ in style)
EN_VARIANTS = {
    "ok": "okay", "gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of",
    "catalogue": "catalog", "colour": "color", "behaviour": "behavior", "favourite": "favorite",
    "optimise": "optimize", "analyse": "analyze", "centre": "center",
}


def normalize_en(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)  # [music], (laughs)
    text = re.sub(r"^\s*[a-z ]{1,20}:\s", " ", text)   # speaker labels like "SAMI:"
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\b(uh|um|uhm|erm|hmm|mm)\b", " ", text)  # fillers: ASR drops them, verbatim subtitles keep them
    text = " ".join(EN_VARIANTS.get(w, w) for w in text.split())
    return re.sub(r"\s+", " ", text).strip()


def normalize_zh(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text)


def han_count(text: str) -> int:
    return len(re.findall(r"[一-鿿]", text))


def utterances(path: str) -> list[tuple[float, float]]:
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    with wave.open(path) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    speeches = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=0))
    return [(s["start"] / 16000, s["end"] / 16000) for s in speeches]


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 2)


def score_video(run: dict, samples: int) -> dict:
    import jiwer
    from sacrebleu.metrics import CHRF

    video, events = run["video"], run["events"]
    data_dir = os.path.join(DATA_DIR, video["name"])
    ref = json.load(open(os.path.join(data_dir, "reference.json"), encoding="utf-8"))

    finals = {e["id"]: e for e in events if e["type"] == "result" and e.get("original")}
    translations = {e["id"]: e["translated"] for e in events if e["type"] == "translation"}
    order = sorted(finals, key=lambda i: finals[i]["t"])

    # English accuracy
    hyp_en = normalize_en(" ".join(finals[i]["original"] for i in order))
    ref_en = normalize_en(" ".join(c["text"] for c in ref.get("en", [])))
    m = jiwer.process_words(ref_en, hyp_en) if ref_en and hyp_en else None
    english = {
        "wer": round(m.wer * 100, 1) if m else None,
        "substitutions": m.substitutions if m else None,
        "deletions": m.deletions if m else None,
        "insertions": m.insertions if m else None,
        "ref_words": len(ref_en.split()),
    }

    # Latency
    utts = utterances(os.path.join(data_dir, "audio.wav"))
    text_times = sorted(e["t"] for e in events if e["type"] in ("partial", "result") and e.get("original"))
    final_times = sorted(finals[i]["t"] for i in order)
    first_lags, final_lags = [], []
    for start, end in utts:
        nxt = next((t for t in text_times if t >= start), None)
        if nxt is not None and nxt <= end + 3:
            first_lags.append(nxt - start)
        nxt = next((t for t in final_times if t >= end), None)
        if nxt is not None and nxt - end <= 10:
            final_lags.append(nxt - end)
    trans_lags = [e["t"] - finals[e["id"]]["t"] for e in events
                  if e["type"] == "translation" and e["id"] in finals]
    latency = {
        "utterances": len(utts),
        "first_text_p50": pct(first_lags, 0.5), "first_text_p90": pct(first_lags, 0.9),
        "final_p50": pct(final_lags, 0.5), "final_p90": pct(final_lags, 0.9),
        "translation_after_final_p50": pct(trans_lags, 0.5),
    }
    if run.get("retranslated_from"):
        # eval.retranslate copies the transcription timing of another run and adds
        # bare translation time: no GPU queueing, no preview cost. Not a latency measurement.
        latency = {"utterances": len(utts), "not_measured": f"retranslated from {run['retranslated_from']}"}

    # Translation
    pairs = [(finals[i]["original"], translations.get(i, "")) for i in order]
    words = [len(o.split()) for o, _ in pairs]
    short = [(o, z) for (o, z), n in zip(pairs, words) if n >= 4 and han_count(z) < SHORT_RATIO * n]
    long_pairs = sum(n >= 4 for n in words)
    translation = {
        "sentences": len(pairs),
        "untranslated": sum(1 for _, z in pairs if not z),
        "short_share": round(100 * len(short) / long_pairs, 1) if long_pairs else None,
        "zh_chars_per_en_word": round(sum(han_count(z) for _, z in pairs) / max(1, sum(words)), 2),
    }
    zh_key = next((k for k in ref if k.lower() in ("zh-tw", "zh-hant")), None) or \
        next((k for k in ref if k.lower().startswith("zh")), None)
    if zh_key:
        ref_zh = " ".join(c["text"] for c in ref[zh_key])
        if zh_key.lower() not in ("zh-tw", "zh-hant"):
            import opencc
            ref_zh = opencc.OpenCC("s2twp").convert(ref_zh)
        hyp_zh = normalize_zh(" ".join(z for _, z in pairs))
        ref_zh = normalize_zh(ref_zh)
        translation["reference"] = zh_key
        translation["chrf"] = round(CHRF().corpus_score([hyp_zh], [[ref_zh]]).score, 1)
        translation["ref_zh_chars_per_en_word"] = round(han_count(ref_zh) / max(1, len(ref_en.split())), 2)

    step = max(1, len(pairs) // samples)
    return {
        "video": video,
        "english": english,
        "latency": latency,
        "translation": translation,
        "samples": [{"en": o, "zh": z} for o, z in pairs[::step][:samples]],
        "short_examples": [{"en": o, "zh": z} for o, z in short[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    run_dir = os.path.join(RESULTS_DIR, args.label)
    names = sorted(f[:-5] for f in os.listdir(run_dir) if f.endswith(".json") and f != "score.json")
    scores = [score_video(json.load(open(os.path.join(run_dir, f"{n}.json"), encoding="utf-8")), args.samples)
              for n in names]
    json.dump(scores, open(os.path.join(run_dir, "score.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    fmt = lambda v: "-" if v is None else str(v)
    print(f"{'video':<24}{'WER%':>6}{'first p50':>10}{'p90':>6}{'final p50':>10}{'p90':>6}"
          f"{'short%':>8}{'zh/word':>8}{'chrF':>6}")
    for s in scores:
        e, l, t = s["english"], s["latency"], s["translation"]
        print(f"{s['video']['name']:<24}{fmt(e['wer']):>6}{fmt(l.get('first_text_p50')):>10}{fmt(l.get('first_text_p90')):>6}"
              f"{fmt(l.get('final_p50')):>10}{fmt(l.get('final_p90')):>6}{fmt(t['short_share']):>8}"
              f"{fmt(t['zh_chars_per_en_word']):>8}{fmt(t.get('chrf')):>6}")
    if any("not_measured" in s["latency"] for s in scores):
        print("\nLatency columns are blank: this run came from eval.retranslate, which reuses another run's"
              "\ntranscription timing. Only eval.run measures latency.")
    print(f"\nDetails and sample translations: {os.path.join(run_dir, 'score.json')}")


if __name__ == "__main__":
    main()
