"""
Overlay UI for displaying transcription and translation.
Semi-transparent window that stays on top.
"""

from collections import deque
from dataclasses import dataclass
import time

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QFont, QMouseEvent


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
    ):
        super().__init__()

        self.max_lines = max_lines
        self.font_size = font_size
        self._entries: deque[TextEntry] = deque(maxlen=max_lines)
        self._drag_pos: QPoint | None = None

        # Window setup
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(width)

        self._setup_ui(opacity)
        self._position_window(position)

        # Connect signals
        self.text_received.connect(self._on_text)
        self.translation_received.connect(self._on_translation)
        self.partial_received.connect(self._on_partial)
        self.status_received.connect(self._on_status)

    def _setup_ui(self, opacity: float):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

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

    def _position_window(self, position: str):
        screen = QApplication.primaryScreen().availableGeometry()
        margin = 30
        self.adjustSize()

        if "right" in position:
            x = screen.right() - self.width() - margin
        else:
            x = screen.left() + margin

        if "bottom" in position:
            y = screen.bottom() - self.height() - margin
        else:
            y = screen.top() + margin

        self.move(x, y)

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

        widgets = [self._create_entry(entry) for entry in self._entries]
        for widget in widgets:
            self.text_layout.addWidget(widget)
            # addWidget() only shows the widget on the next event loop pass, and the
            # layout skips hidden widgets, so show it now for the height calculation
            widget.show()

        self._fit_to_content(widgets)

    def _fit_to_content(self, widgets: list[QWidget]):
        """Size the window to its wrapped text.

        adjustSize() sizes word-wrapped labels for a guessed width, so long lines
        were cut off. Ask the layout for the height it needs at the actual width
        instead. If that does not fit on screen, hide the oldest entries so the
        newest one is always complete. In the lower half of the screen the bottom
        edge stays put and the window grows upward.
        """
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        max_height = screen.height() - 2 * 30
        layout = self.layout()

        visible = list(widgets)
        while True:
            # Hiding a widget only invalidates the inner layout; the outer layout keeps
            # a cached height for the container until the container reports a change
            self.text_layout.invalidate()
            self.container.updateGeometry()
            layout.invalidate()
            layout.activate()
            height = layout.totalHeightForWidth(self.width())
            if height <= max_height or len(visible) <= 1:
                break
            visible.pop(0).hide()  # hidden widgets take no space in the layout
        height = min(height, max_height)

        geo = self.geometry()
        grow_upward = geo.center().y() > screen.center().y()
        bottom = geo.bottom()
        self.setFixedHeight(height)
        if grow_upward:
            self.move(geo.x(), bottom - height + 1)

    def _create_entry(self, entry: TextEntry) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        font = QFont("Segoe UI", self.font_size)

        # Original
        orig_label = QLabel(entry.original)
        orig_label.setFont(font)
        orig_label.setStyleSheet("color: #9a9a9a;" if entry.partial else "color: #ffffff;")
        orig_label.setWordWrap(True)
        layout.addWidget(orig_label)

        # Translation
        if entry.translated:
            trans_label = QLabel(entry.translated)
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

    # Drag to move
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None
