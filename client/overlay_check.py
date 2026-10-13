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
    win = OverlayWindow(remember=False)
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

    # User-chosen size: height stays fixed, newest entry sits at the bottom
    win.set_user_size(480, 320)
    for n in range(4, 40, 6):
        win.set_partial(20, " ".join(EN.split()[:n]), ZH[: n * 2])
        check(f"fixed-size preview {n} words")
    win.add_text(EN, ZH, 20)
    check("fixed-size final")
    entries = [win.text_layout.itemAt(i).widget() for i in range(1, win.text_layout.count())]
    newest = entries[-1]
    gap = win.container.height() - newest.geometry().bottom() - 12  # 12 = container bottom margin
    print(f"fixed size: window {win.width()}x{win.height()}, gap under newest entry {gap}px")
    if (win.width(), win.height()) != (480, 320) or not newest.isVisible() or gap > 20:
        failures += 1
        print("fixed size not kept or newest entry not at the bottom")
    win.grab().save(os.path.join(args.out, "fixed_size.png"))

    # A box too small for even one sentence: keep the end of the text visible
    win.set_user_size(360, 160)
    check("tiny fixed size")
    labels = [l for l in entries[-1].findChildren(QLabel) if l.isVisible()]
    ends_ok = all(l.text() == l.property("full_text")
                  or (l.text().startswith("…") and l.property("full_text").endswith(l.text()[1:]))
                  for l in labels)
    print(f"tiny box: {[l.text()[:25] + '...' + l.text()[-12:] for l in labels]}")
    if not ends_ok:
        failures += 1
        print("tiny box does not show the end of the sentence")
    win.grab().save(os.path.join(args.out, "tiny.png"))
    win.set_user_size(480, 320)
    check("widened again")

    # Double-click behaviour: back to a height that follows the content
    win.set_auto_height()
    check("auto height again")
    shown = [w for w in entries if w.isVisible()]
    print(f"auto height: window {win.width()}x{win.height()}, {len(shown)} entries shown")

    # Resize bands: corners and edges just inside the box
    box = win.container.geometry()
    from PyQt6.QtCore import QPoint, Qt
    expect = {
        "top-left": (QPoint(box.left() + 2, box.top() + 2), Qt.Edge.LeftEdge | Qt.Edge.TopEdge),
        "bottom-right": (QPoint(box.right() - 2, box.bottom() - 2), Qt.Edge.RightEdge | Qt.Edge.BottomEdge),
        "right": (QPoint(box.right() - 2, box.center().y()), Qt.Edge.RightEdge),
        "middle": (box.center(), Qt.Edge(0)),
    }
    for name, (pos, edges) in expect.items():
        if win._edges_at(pos) != edges:
            failures += 1
            print(f"resize band {name}: got {win._edges_at(pos)}, expected {edges}")

    # Size, position and fixed height survive a restart
    first = OverlayWindow(settings_name="overlay-check")
    first.show()
    app.processEvents()
    first.set_user_size(520, 300)
    first.move(150, 120)
    app.processEvents()
    first.close()
    second = OverlayWindow(settings_name="overlay-check")
    second.show()
    app.processEvents()
    restored = (second.x(), second.y(), second.width(), second.height(), second._fixed_height)
    print(f"restored after restart: x={restored[0]} y={restored[1]} {restored[2]}x{restored[3]} fixed={restored[4]}")
    if restored != (150, 120, 520, 300, 300):
        failures += 1
        print("size/position not restored")
    second._settings.clear()
    second.close()

    print(f"{steps} updates, {failures} with clipped text or off-screen window")
    print(f"window {win.width()}x{win.height()} at y={win.y()}, screen height {win.screen().availableGeometry().height()}")
    return 1 if failures else 0


def check_after(fn, *args):
    *fn_args, check, tag = args
    fn(*fn_args)
    return check(tag)


if __name__ == "__main__":
    sys.exit(main())
