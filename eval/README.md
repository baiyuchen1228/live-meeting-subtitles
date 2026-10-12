# Evaluation

Tests the full pipeline on real interview videos that have **human-made** subtitles
(YouTube auto-captions are not used as a reference).

```bash
source venv/bin/activate
pip install -r eval/requirements.txt

python -m eval.fetch                          # download excerpts + subtitles into eval/data/
python -m eval.run --spawn --label my-test    # stream each excerpt in real time (~55 min)
python -m eval.score my-test                  # print the table, write eval/results/my-test/score.json
```

- `--spawn` starts a server on port 9877 for the run (pass server options with
  `--server-args "--translator deepl"`), so a server on 9876 is not affected.
- `--only NAME ...` limits a run to some videos; names are in `videos.json`.
- `eval/data/` and `eval/results/` are git-ignored.

## Videos (`videos.json`)

| Name | Source | Excerpt | Reference |
|---|---|---|---|
| google-coding | Life at Google, coding interview | 2:00-12:00 | en |
| spotify-system-design | IGotAnOffer, system design | 2:00-12:00 | en (verbatim, with fillers) |
| india-system-design | Life at Google, Indian English | 2:00-12:00 | en, **zh-TW** |
| google-sysdesign-tips | Life at Google, advice | whole (5:24) | en, zh-Hans (converted to zh-TW) |
| pm-mock | IGotAnOffer, PM mock interview | 2:00-12:00 | en |
| pm-behavioral | IGotAnOffer, behavioral | 1:00-11:00 | en |

## Metrics

| Column | Meaning |
|---|---|
| WER% | Word error rate of the final English vs the human subtitles (lower is better). Fillers like "uh" are removed and spelling variants (ok/okay, gonna/going to) unified on both sides |
| first p50 / p90 | Seconds from the start of an utterance to the first text on screen (preview or final) |
| final p50 / p90 | Seconds from the end of an utterance to its final English |
| short% | Share of sentences (4+ words) whose translation has fewer than 0.8 Chinese characters per English word: likely dropped content |
| zh/word | Chinese characters per English word over the whole excerpt. Human Chinese subtitles of these videos run about 1.3-1.6 |
| chrF | Character n-gram overlap with the human Chinese subtitles (0-100, higher is better). Only for videos with a Chinese reference |

Utterances come from Silero VAD on the excerpt audio (split at 0.5 s pauses).
Subtitle excerpts are cut by cue time, so a few words at the excerpt edges may
not match the audio exactly.
