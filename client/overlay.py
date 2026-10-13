"""
Overlay UI for displaying transcription and translation.
Semi-transparent window that stays on top.

Drag inside the box to move it; drag its edges or corners to resize. After a
manual resize the height stays fixed (newest subtitle at the bottom, older ones
pushed out at the top); double-click the box to go back to automatic height.
Size and position are remembered between runs.
"""

from collections import deque
from dataclasses import dataclass
import ctypes
import ctypes.wintypes
import logging
import os
import sys
import tempfile
import time

from PyQt6.QtWidgets import (
    QApplication,
    QLayout,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSettings, QTimer
from PyQt6.QtGui import QFont, QMouseEvent, QCursor

# Width of the band inside the box's edges that resizes the window
RESIZE_MARGIN = 8
MIN_WIDTH, MIN_HEIGHT = 320, 120

# Windows sends these when its native resize loop (startSystemResize) starts and ends
WM_ENTERSIZEMOVE, WM_EXITSIZEMOVE = 0x0231, 0x0232
WM_SYSCOMMAND, WM_SIZING = 0x0112, 0x0214

# OVERLAY_DEBUG=1 logs window sizing decisions to %TEMP%\overlay_debug.log
log = logging.getLogger("overlay")
if os.environ.get("OVERLAY_DEBUG"):
    _handler = logging.FileHandler(os.path.join(tempfile.gettempdir(), "overlay_debug.log"), mode="w", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S"))
    log.addHandler(_handler)
    log.setLevel(logging.DEBUG)


@dataclass
class TextEntry:
    original: str
    translated: str
    timestamp: float
    id: int = -1
    partial: bool = False  # live preview, still being spoken
    translation_pending: bool = False  # showing a preview's translation until the final one arrives


class OverlayWindow(QWidget):
    """Semi-transparent overlay for subtitles."""

    # Signal for thread-safe updates
    text_received = pyqtSignal(str, str, int)  # original, translated, id
    translation_received = pyqtSignal(int, str)  # id, translated
    partial_received = pyqtSignal(int, str, str)  # id, original, translated
    status_received = pyqtSignal(str)

    def __init__(
        self,
        width: int = 700,
        max_lines: int = 6,
        font_size: int = 15,
        opacity: float = 0.9,
        position: str = "bottom-right",
        remember: bool = True,
        settings_name: str = "overlay",
    ):
        super().__init__()

        self.max_lines = max_lines
        self.font_size = font_size
        self._entries: deque[TextEntry] = deque(maxlen=max_lines)
        self._entry_widgets: list[QWidget] = []
        self._drag_pos: QPoint | None = None

        # None = height follows the content; a number = height set by the user
        self._fixed_height: int | None = None
        self._user_resizing = False
        self._native_resize_events = False  # True once Windows has reported a resize loop
        self._fitted_width = 0
        self._settings = QSettings("live-meeting-subtitles", settings_name) if remember else None
        # Without native resize-loop messages, treat a pause in resize events as the end
        self._save_timer = QTimer(self, singleShot=True, interval=400)
        self._save_timer.timeout.connect(self._end_user_change)

        # Window setup
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        self._setup_ui(opacity)
        self.resize(width, self.height())
        if not self._restore_settings():
            self._position_window(position)

        # Connect signals
        self.text_received.connect(self._on_text)
        self.translation_received.connect(self._on_translation)
        self.partial_received.connect(self._on_partial)
        self.status_received.connect(self._on_status)

    def _setup_ui(self, opacity: float):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        # The window size is set by _fit_to_content() or the user. Without this the
        # layout raises the window's minimum height to the labels' minimum size hints,
        # which overrides resize() (window taller than the screen, fixed height ignored)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        # Container with background
        self.container = QWidget()
        # Only the container gets the background (a bare "QWidget" selector would
        # also give every subtitle entry its own box)
        self.container.setObjectName("container")
        self.container.setStyleSheet(f"""
            QWidget#container {{
                background-color: rgba(20, 20, 30, {int(opacity * 255)});
                border-radius: 12px;
            }}
        """)

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(16, 12, 16, 12)
        container_layout.setSpacing(8)

        # Header
        header = QHBoxLayout()

        self.status_label = QLabel("Connecting...")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(self.status_label)

        header.addStretch()

        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #666;
                border: none;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover { color: #ff5555; }
        """)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)

        container_layout.addLayout(header)

        # Takes the spare room when the height is fixed, so subtitles sit at the bottom
        container_layout.addStretch(1)

        # Text area
        self.text_layout = QVBoxLayout()
        self.text_layout.setSpacing(10)
        container_layout.addLayout(self.text_layout)

        # Placeholder
        self.placeholder = QLabel("Waiting for speech...")
        self.placeholder.setStyleSheet(f"color: #555; font-size: {self.font_size}px; font-style: italic;")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_layout.addWidget(self.placeholder)

        layout.addWidget(self.container)
        self._track_mouse(self)

    @staticmethod
    def _track_mouse(widget: QWidget):
        """Hover moves only reach widgets with tracking on; needed for the resize cursor."""
        widget.setMouseTracking(True)
        for child in widget.findChildren(QWidget):
            child.setMouseTracking(True)

    def _position_window(self, position: str):
        screen = QApplication.primaryScreen().availableGeometry()
        margin = 30

        if "right" in position:
            x = screen.right() - self.width() - margin
        else:
            x = screen.left() + margin

        if "bottom" in position:
            y = screen.bottom() - self.height() - margin
        else:
            y = screen.top() + margin

        self.move(x, y)

    # Remembered size and position

    def _restore_settings(self) -> bool:
        if self._settings is None:
            return False
        geometry = self._settings.value("geometry")
        if not geometry:
            return False
        self.restoreGeometry(geometry)  # also pulls the window back on screen if needed
        fixed = int(self._settings.value("fixed_height", 0) or 0)
        self._fixed_height = fixed if fixed > 0 else None
        return True

    def _save_settings(self):
        if self._settings is None:
            return
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("fixed_height", self._fixed_height or 0)
        self._settings.sync()

    def _end_user_change(self):
        log.debug("save timer fired: native=%s resizing=%s", self._native_resize_events, self._user_resizing)
        if self._native_resize_events:
            return  # WM_EXITSIZEMOVE ends the resize; a pause mid-drag must not
        self._user_resizing = False
        self._save_settings()

    def _finish_user_resize(self):
        log.debug("user resize finished: %dx%d fixed=%s", self.width(), self.height(), self._fixed_height)
        self._user_resizing = False
        self._save_settings()

    def nativeEvent(self, event_type, message):
        if sys.platform == "win32" and bytes(event_type) == b"windows_generic_MSG":
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message in (WM_ENTERSIZEMOVE, WM_EXITSIZEMOVE, WM_SYSCOMMAND):
                log.debug("native msg 0x%04x wParam=0x%x resizing=%s", msg.message, msg.wParam, self._user_resizing)
            if msg.message == WM_ENTERSIZEMOVE and self._user_resizing:
                self._native_resize_events = True
            elif msg.message == WM_EXITSIZEMOVE and self._user_resizing:
                # This window gets no resize events during the drag: Windows applies the
                # final size right after this message. Keep treating resizes as the user's
                # until then, or the new height is taken for our own and undone.
                QTimer.singleShot(150, self._finish_user_resize)
        # (False, 0) = not handled, same as the default. Do not call
        # super().nativeEvent(): with PyQt6 6.7 that crashes the process (access violation)
        return False, 0

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def add_text(self, original: str, translated: str, entry_id: int = -1):
        """Thread-safe method to add text."""
        self.text_received.emit(original, translated, entry_id)

    def set_partial(self, entry_id: int, original: str, translated: str):
        """Thread-safe method to show or update a live preview."""
        self.partial_received.emit(entry_id, original, translated)

    def set_translation(self, entry_id: int, translated: str):
        """Thread-safe method to fill in the translation of an existing entry."""
        self.translation_received.emit(entry_id, translated)

    def set_status(self, status: str):
        """Thread-safe method to update status."""
        self.status_received.emit(status)

    def set_user_size(self, width: int, height: int):
        """Apply a size chosen by the user: the height stays fixed from now on."""
        self._fixed_height = max(MIN_HEIGHT, height)
        self.resize(max(MIN_WIDTH, width), self._fixed_height)
        self._fit_to_content()

    def set_auto_height(self):
        """Go back to a height that follows the content."""
        log.debug("set_auto_height")
        self._fixed_height = None
        self._fit_to_content()
        self._save_settings()

    def _find_entry(self, entry_id: int) -> TextEntry | None:
        if entry_id < 0:
            return None
        return next((e for e in self._entries if e.id == entry_id), None)

    def _on_text(self, original: str, translated: str, entry_id: int):
        entry = self._find_entry(entry_id)

        if not original:
            # Final pass found no speech: drop the preview
            if entry:
                self._entries = deque((e for e in self._entries if e is not entry), maxlen=self.max_lines)
                self._rebuild_display()
            return

        self.placeholder.hide()
        if entry:
            # Replace the preview in place; keep its translation until the final one arrives
            entry.original = original
            entry.partial = False
            if translated:
                entry.translated = translated
                entry.translation_pending = False
            else:
                entry.translation_pending = bool(entry.translated)
        else:
            self._entries.append(TextEntry(original, translated, time.time(), entry_id))
        self._rebuild_display()

    def _on_partial(self, entry_id: int, original: str, translated: str):
        entry = self._find_entry(entry_id)
        if entry and not entry.partial:
            return  # already final
        self.placeholder.hide()
        if entry:
            entry.original = original
            entry.translated = translated or entry.translated
        else:
            self._entries.append(TextEntry(original, translated, time.time(), entry_id, partial=True))
        self._rebuild_display()

    def _on_translation(self, entry_id: int, translated: str):
        entry = self._find_entry(entry_id)
        if entry:
            entry.translated = translated
            entry.translation_pending = False
            self._rebuild_display()

    def _on_status(self, status: str):
        self.status_label.setText(status)

    def _rebuild_display(self):
        # Clear old entries; hide them right away, deleteLater only runs on the next loop
        while self.text_layout.count() > 1:
            item = self.text_layout.takeAt(1)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()

        self._entry_widgets = [self._create_entry(entry) for entry in self._entries]
        for widget in self._entry_widgets:
            self.text_layout.addWidget(widget)
            self._track_mouse(widget)

        self._fit_to_content()

    def _fit_to_content(self):
        """Fit the subtitles to the window.

        Word-wrapped labels need the layout's height-for-width at the actual
        width (adjustSize() guesses a width and cut lines off). With a fixed
        height, the oldest entries are hidden until the rest fit. With automatic
        height, the window takes the height it needs up to the screen height;
        in the lower half of the screen its bottom edge stays put.
        """
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        limit = self._fixed_height or screen.height() - 2 * 30
        layout = self.layout()

        # addWidget() only shows a widget on the next event loop pass, and the layout
        # skips hidden widgets, so show every entry before measuring
        for widget in self._entry_widgets:
            widget.show()
            for label in widget.findChildren(QLabel):  # undo an earlier _elide_start()
                full = label.property("full_text")
                if full and label.text() != full:
                    label.setText(full)
        visible = list(self._entry_widgets)
        while True:
            height = self._needed_height()
            if height <= limit or len(visible) <= 1:
                break
            visible.pop(0).hide()  # hidden widgets take no space in the layout
        if height > limit and visible:
            # The newest entry alone is taller than the window: drop the start of its
            # text, since the latest words are the ones being read
            height = self._elide_start(visible[-1], limit)
        self._fitted_width = self.width()

        log.debug("fit: fixed=%s limit=%s needed=%s size=%dx%d visible=%d/%d",
                  self._fixed_height, limit, height, self.width(), self.height(),
                  len(visible), len(self._entry_widgets))
        if self._fixed_height:
            if self.height() != self._fixed_height:
                self.resize(self.width(), self._fixed_height)
            return

        height = max(MIN_HEIGHT, min(height, limit))
        geo = self.geometry()
        grow_upward = geo.center().y() > screen.center().y()
        bottom = geo.bottom()
        self.resize(self.width(), height)
        if grow_upward:
            self.move(geo.x(), bottom - height + 1)
        log.debug("fit auto: resized to %dx%d at y=%d", self.width(), height, self.y())

    def hasHeightForWidth(self) -> bool:
        # Otherwise Qt on Windows re-sizes the top-level window to the layout's
        # height-for-width after each resize, overriding the user's fixed height.
        # _fit_to_content() asks the layout directly instead.
        return False

    def _needed_height(self) -> int:
        # Layouts cache heights at every level: the entry layouts (label text changed),
        # the text layout (entries hidden) and the outer layout (container height)
        for widget in self._entry_widgets:
            widget.layout().invalidate()
            widget.updateGeometry()
        self.text_layout.invalidate()
        self.container.updateGeometry()
        self.layout().invalidate()
        self.layout().activate()
        return self.layout().totalHeightForWidth(self.width())

    def _elide_start(self, widget: QWidget, limit: int) -> int:
        """Shorten the entry's labels from the front ("…" + the end) until it fits."""
        labels = [label for label in widget.findChildren(QLabel) if label.text()]
        height = self._needed_height()
        while height > limit:
            changed = False
            for label in labels:
                text = label.text().lstrip("…")
                words = text.split(" ")
                if len(words) > 3:
                    text = " ".join(words[1:])  # English: drop a word
                elif len(text) > 4 and " " not in text:
                    text = text[2:]  # Chinese: drop two characters
                else:
                    continue
                label.setText("…" + text)
                changed = True
            if not changed:
                break
            height = self._needed_height()
        return height

    def resizeEvent(self, event):
        super().resizeEvent(event)
        log.debug("resizeEvent %dx%d -> %dx%d resizing=%s fixed=%s", event.oldSize().width(),
                  event.oldSize().height(), event.size().width(), event.size().height(),
                  self._user_resizing, self._fixed_height)
        if self._user_resizing:
            # The system resize loop sends these while the user drags an edge
            self._fixed_height = self.height()
            self._save_timer.start()
        if self._user_resizing or self.width() != self._fitted_width:
            self._fit_to_content()

    def moveEvent(self, event):
        super().moveEvent(event)
        if self._user_resizing:
            self._save_timer.start()

    def _create_entry(self, entry: TextEntry) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        font = QFont("Segoe UI", self.font_size)

        # Original
        orig_label = QLabel(entry.original)
        orig_label.setProperty("full_text", entry.original)
        orig_label.setFont(font)
        orig_label.setStyleSheet("color: #9a9a9a;" if entry.partial else "color: #ffffff;")
        orig_label.setWordWrap(True)
        layout.addWidget(orig_label)

        # Translation
        if entry.translated:
            trans_label = QLabel(entry.translated)
            trans_label.setProperty("full_text", entry.translated)
            trans_label.setFont(QFont("Microsoft JhengHei UI", self.font_size))
            dim = entry.partial or entry.translation_pending
            trans_label.setStyleSheet("color: #6f8fb3;" if dim else "color: #7cb7ff;")
            trans_label.setWordWrap(True)
            layout.addWidget(trans_label)

        return widget

    def clear(self):
        self._entries.clear()
        self._rebuild_display()
        self.placeholder.show()

    # Mouse: drag inside to move, drag the edges to resize, double-click for auto height

    def _edges_at(self, pos: QPoint) -> Qt.Edge:
        """Window edges whose resize band (just inside the box) contains pos."""
        box = self.container.geometry()
        edges = Qt.Edge(0)
        if box.left() <= pos.x() < box.left() + RESIZE_MARGIN:
            edges |= Qt.Edge.LeftEdge
        elif box.right() - RESIZE_MARGIN < pos.x() <= box.right():
            edges |= Qt.Edge.RightEdge
        if box.top() <= pos.y() < box.top() + RESIZE_MARGIN:
            edges |= Qt.Edge.TopEdge
        elif box.bottom() - RESIZE_MARGIN < pos.y() <= box.bottom():
            edges |= Qt.Edge.BottomEdge
        return edges

    @staticmethod
    def _cursor_for(edges: Qt.Edge) -> Qt.CursorShape:
        horizontal = bool(edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge))
        vertical = bool(edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge))
        if horizontal and vertical:
            main_diagonal = edges in (Qt.Edge.LeftEdge | Qt.Edge.TopEdge, Qt.Edge.RightEdge | Qt.Edge.BottomEdge)
            return Qt.CursorShape.SizeFDiagCursor if main_diagonal else Qt.CursorShape.SizeBDiagCursor
        if horizontal:
            return Qt.CursorShape.SizeHorCursor
        if vertical:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        edges = self._edges_at(event.position().toPoint())
        started = bool(edges) and self.windowHandle() is not None and self.windowHandle().startSystemResize(edges)
        log.debug("press at %s edges=%s startSystemResize=%s", event.position().toPoint(), edges, started)
        if started:
            self._user_resizing = True
            return
        self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            return
        if event.buttons() == Qt.MouseButton.NoButton:
            shape = self._cursor_for(self._edges_at(event.position().toPoint()))
            if self.cursor().shape() != shape:
                self.setCursor(QCursor(shape))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag_pos is not None:
            self._drag_pos = None
            self._save_settings()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        log.debug("double click at %s", event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and self._fixed_height:
            self.set_auto_height()

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)
