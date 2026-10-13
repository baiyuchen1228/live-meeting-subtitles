"""
Layout check for the subtitle overlay: feeds it previews that grow word by
word (like a long sentence being spoken) and reports any label that is
shorter than its text needs, or any part of the window that is off screen.
Saves screenshots so the result can be looked at.

Usage (Windows):  python -m client.overlay_check [--out DIR]
(set QT_QPA_PLATFORM=offscreen and QT_QPA_FONTDIR=C:\\Windows\\Fonts to run without showing a window)
"""

import argparse
import os
import sys

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QLabel

from client.overlay import OverlayWindow

EN = ("So the main issue is that the latency is too high and we need to fix it before the release "
      "next week, because the customers in Europe and North America are already complaining about "
      "the delay they see on every single request they make to the recommendation service.")
ZH = ("所以主要的問題是延遲太高，我們需要在下週發布之前修好，因為歐洲和北美的客戶已經在抱怨"
      "他們對推薦服務發出的每一個請求都會看到延遲。")


def problems(win: OverlayWindow) -> list[str]:
    out = []
    for label in win.findChildren(QLabel):
        if not label.isVisible() or not label.wordWrap() or not label.text():
            continue
        need = label.heightForWidth(label.width())
        if label.height() < need:
            out.append(f"clipped label: has {label.height()}px, needs {need}px: {label.text()[:40]!r}")
    screen = win.screen().availableGeometry()
    geo = win.frameGeometry()
    if geo.bottom() > screen.bottom() or geo.top() < screen.top():
        out.append(f"window {geo.top()}..{geo.bottom()} outside screen {screen.top()}..{screen.bottom()}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="overlay_check")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    app = QApplication(sys.argv)
    win = OverlayWindow()
    win.show()
    app.processEvents()

    failures = 0
    steps = 0

    def check(tag: str):
        nonlocal failures, steps
        app.processEvents()
        # processEvents() does not run deleteLater(); the real event loop does
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        steps += 1
        found = problems(win)
        if found:
            failures += 1
            print(f"[{tag}] " + " | ".join(found))
        return found

    # Five finished sentences, then one long sentence growing word by word
    for i in range(5):
        win.add_text(f"Sentence {i}: we should probably cache the results of this query.", "", i)
        check(f"final {i}")
        win.set_translation(i, "我們應該把這個查詢的結果快取起來。")
        check(f"translation {i}")

    words = EN.split()
    for n in range(3, len(words) + 1, 3):
        en = " ".join(words[:n])
        zh = ZH[: max(1, int(len(ZH) * n / len(words)))]
        found = check_after(win.set_partial, 5, en, zh, check, f"preview {n} words")
        if n in (12, 30, len(words) - len(words) % 3):
            win.grab().save(os.path.join(args.out, f"preview_{n:02d}.png"))
    win.add_text(EN, "", 5)
    check("final long")
    win.set_translation(5, ZH)
    check("final long translation")
    win.grab().save(os.path.join(args.out, "final.png"))

    # More long sentences than the screen can hold: oldest entries must give way
    for i in range(6, 12):
        win.add_text(EN, ZH, i)
        check(f"long final {i}")
    entries = [win.text_layout.itemAt(i).widget() for i in range(1, win.text_layout.count())]
    shown = [w for w in entries if w.isVisible()]
    if not entries[-1].isVisible():
        failures += 1
        print("newest entry is not visible")
    # The window should be no taller than the entries it shows
    content = sum(w.height() for w in shown)
    print(f"overflow: {len(shown)} of {len(entries)} entries shown, window {win.height()}px, entries {content}px")
    if win.height() > content + 120:
        failures += 1
        print("window has large empty space")
    win.grab().save(os.path.join(args.out, "overflow.png"))

    print(f"{steps} updates, {failures} with clipped text or off-screen window")
    print(f"window {win.width()}x{win.height()} at y={win.y()}, screen height {win.screen().availableGeometry().height()}")
    return 1 if failures else 0


def check_after(fn, *args):
    *fn_args, check, tag = args
    fn(*fn_args)
    return check(tag)


if __name__ == "__main__":
    sys.exit(main())
