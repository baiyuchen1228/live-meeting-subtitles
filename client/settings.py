"""
Client settings that persist between runs, stored next to the overlay's
(Windows registry: HKEY_CURRENT_USER\\Software\\live-meeting-subtitles\\client).
"""

from PyQt6.QtCore import QSettings


def _settings() -> QSettings:
    return QSettings("live-meeting-subtitles", "client")


def last_device() -> str | None:
    """Name of the audio device that was last opened successfully."""
    name = _settings().value("last_device")
    return str(name) if name else None


def save_last_device(name: str) -> None:
    settings = _settings()
    settings.setValue("last_device", name)
    settings.sync()
