"""
Re-translate the final English of an earlier run with another translator,
without streaming the audio again (minutes instead of an hour).

The new run keeps the original transcription events and replaces each final
translation. Use it to compare translation quality only: it does not simulate
GPU queueing or the cost of preview translations, so eval.score leaves the
latency columns blank for it. Latency conclusions need a real eval.run.

Usage:
  python -m eval.retranslate --from baseline-nllb600m --label opus --translator opus
  python -m eval.retranslate --from baseline-nllb600m --label nllb1.3b --translator local \\
      --model facebook/nllb-200-distilled-1.3B
  python -m eval.retranslate --from baseline-nllb600m --label nllb600m-raw --no-clean
"""

import argparse
import json
import os
import statistics
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

from server.text_cleanup import clean_for_translation  # noqa: E402
from server.translators import create_translator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", required=True, help="Run label to take English from")
    parser.add_argument("--label", required=True, help="Label for the new run")
    parser.add_argument("--translator", default="local")
    parser.add_argument("--model", help="model_name passed to the translator")
    parser.add_argument("--target", default="ZH-TW")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove fillers before translating")
    args = parser.parse_args()

    kwargs = {"model_name": args.model} if args.model else {}
    translator = create_translator(args.translator, **kwargs)
    translator.translate("Hello.", args.target, "en")  # load + warm up

    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None

    src_dir, out_dir = os.path.join(RESULTS_DIR, args.source), os.path.join(RESULTS_DIR, args.label)
    os.makedirs(out_dir, exist_ok=True)
    times = []
    for name in sorted(f[:-5] for f in os.listdir(src_dir) if f.endswith(".json") and f != "score.json"):
        run = json.load(open(os.path.join(src_dir, f"{name}.json"), encoding="utf-8"))
        events = [e for e in run["events"] if e["type"] != "translation"]
        for e in [e for e in events if e["type"] == "result" and e.get("original")]:
            text = e["original"] if args.no_clean else clean_for_translation(e["original"])
            start = time.time()
            zh = translator.translate(text, args.target, "en") if text else ""
            elapsed = time.time() - start
            times.append(elapsed)
            if zh:
                events.append({"type": "translation", "id": e["id"], "translated": zh,
                               "t": round(e["t"] + elapsed, 3)})
        events.sort(key=lambda e: e["t"])
        run.update(events=events, translator=translator.name, cleaned=not args.no_clean, retranslated_from=args.source)
        json.dump(run, open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[{name}] done", flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch and torch.cuda.is_available() else None
    print(f"{translator.name}: {len(times)} sentences, median {statistics.median(times) * 1000:.0f} ms, "
          f"p90 {sorted(times)[int(0.9 * len(times))] * 1000:.0f} ms"
          + (f", GPU peak {peak:.2f} GB" if peak else ""))
    print(f"Score with: python -m eval.score {args.label}")


if __name__ == "__main__":
    main()
