"""
WebSocket server for voice transcription and translation.
Runs in WSL with CUDA support.
"""

import asyncio
import json
import time
import argparse
import numpy as np

try:
    import websockets
except ImportError:
    websockets = None

try:
    from faster_whisper.vad import get_speech_timestamps, VadOptions
except ImportError:
    get_speech_timestamps = None

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from shared.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MessageType,
    TranscriptionResult,
    StatusMessage,
    encode_message,
    decode_message,
)
from server.transcribers import create_transcriber, BaseTranscriber
from server.translators import create_translator, Translator


class TranscriptionServer:
    """WebSocket server for real-time transcription."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        transcriber_engine: str = "whisper",
        model: str = "large-v3",
        device: str = "cuda",
        translation_provider: str = "google",
        target_language: str = "ZH-TW",
        source_language: str = "en",
    ):
        if websockets is None:
            raise ImportError("websockets required: pip install websockets")

        self.host = host
        self.port = port
        self.target_language = target_language
        self.source_language = source_language
        self.transcriber_engine = transcriber_engine

        # Initialize transcriber using factory
        if transcriber_engine == "parakeet":
            self.transcriber: BaseTranscriber = create_transcriber(
                "parakeet",
                device=device,
                language=source_language,
            )
        else:
            self.transcriber: BaseTranscriber = create_transcriber(
                "whisper",
                model_name=model,
                device=device,
                language=source_language,
            )
        print(f"Transcriber: {transcriber_engine}")

        # Initialize translator
        self.translator: Translator | None = None
        self.translation_provider = translation_provider
        if translation_provider and translation_provider != "none":
            try:
                self.translator = create_translator(translation_provider)
                print(f"Translator: {self.translator.name}")
            except Exception as e:
                print(f"Warning: Could not initialize translator: {e}")

        # Audio buffer for accumulating chunks
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_duration = 0.0
        self._sample_rate = 16000
        self._end_silence = 0.5  # Silence after speech that ends an utterance
        self._max_duration = 8.0  # Maximum seconds before forced processing
        self._partial_interval = 0.8  # Seconds between live previews of an unfinished utterance
        self._trailing_silence = 0.0  # Silence after the last speech in the buffer (from VAD)
        self._vad_options = (
            VadOptions(min_silence_duration_ms=100, min_speech_duration_ms=250, speech_pad_ms=0)
            if get_speech_timestamps else None
        )
        self._next_id = 0  # id of the utterance currently being buffered

        # Live previews run in the background; the lock keeps them and the
        # final pass from using the GPU at the same time
        self._gpu_lock = asyncio.Lock()
        self._partial_task: asyncio.Task | None = None
        self._last_partial_time = 0.0
        self._partial_sent_for: int | None = None
        self._partial_cache: tuple[str, str] = ("", "")  # (original, translated)

        self._clients: set = set()

    async def start(self):
        """Start the WebSocket server."""
        # Load model before accepting connections
        print("Loading transcription model...")
        self.transcriber.load()
        self._warmup()

        print(f"Starting server on ws://{self.host}:{self.port}")
        async with websockets.serve(self._handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever

    def _warmup(self):
        """Run each model once so the first real sentence isn't slowed by lazy loading."""
        print("Warming up models...")
        try:
            noise = (np.random.randn(self._sample_rate) * 0.1).astype(np.float32)
            self.transcriber.transcribe(noise, self._sample_rate)
            if self.translator:
                self.translator.translate("Hello.", self.target_language, self.source_language)
        except Exception as e:
            print(f"Warmup failed (continuing): {e}")

    async def _handle_client(self, websocket):
        """Handle a client connection."""
        self._clients.add(websocket)
        client_id = id(websocket)
        print(f"Client connected: {client_id}")

        # Send ready status
        await websocket.send(encode_message({
            "type": MessageType.STATUS,
            "status": "ready",
            "message": "Server ready",
            "model_loaded": True,
        }))

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary message = audio data
                    await self._handle_audio(websocket, message)
                else:
                    # Text message = JSON command
                    await self._handle_command(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            print(f"Client disconnected: {client_id}")

    async def _handle_audio(self, websocket, data: bytes):
        """Handle incoming audio chunk."""
        # Convert bytes to float32 numpy array
        # Expecting: int16, mono, 16kHz
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Add to buffer
        self._audio_buffer.append(audio)
        self._buffer_duration += len(audio) / self._sample_rate

        cut = self._find_cut_point()
        if cut is not None:
            await self._process_buffer(websocket, cut)
        else:
            self._maybe_start_partial(websocket)

    def _maybe_start_partial(self, websocket):
        """Start a background preview of the utterance so far, if one is due."""
        # Silence-only buffers are trimmed to 0.5s, so >= 1s means there is speech
        if self._buffer_duration < 1.0:
            return
        if self._partial_task and not self._partial_task.done():
            return
        if time.time() - self._last_partial_time < self._partial_interval:
            return
        # The speaker has paused: the final pass is probably moments away, keep the GPU free for it
        if self._trailing_silence >= 0.2:
            return

        self._last_partial_time = time.time()
        audio = np.concatenate(self._audio_buffer)
        self._partial_task = asyncio.create_task(
            self._send_partial(websocket, self._next_id, audio)
        )

    async def _send_partial(self, websocket, utterance_id: int, audio: np.ndarray):
        """Transcribe and translate an unfinished utterance and send it as a preview."""
        loop = asyncio.get_running_loop()
        try:
            async with self._gpu_lock:
                if utterance_id != self._next_id:
                    return  # the final result for this utterance was already sent
                result = await loop.run_in_executor(
                    None, self.transcriber.transcribe, audio, self._sample_rate
                )
                if result is None or not result.text:
                    return

                cached_original, translated = self._partial_cache
                if self.translator and result.text != cached_original:
                    translated = await loop.run_in_executor(
                        None,
                        self.translator.translate,
                        result.text,
                        self.target_language,
                        result.language,
                    )
                self._partial_cache = (result.text, translated)

                if utterance_id != self._next_id:
                    return
                await websocket.send(encode_message({
                    "type": MessageType.PARTIAL,
                    "id": utterance_id,
                    "original": result.text,
                    "translated": translated,
                }))
                self._partial_sent_for = utterance_id
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"Preview error: {e}")

    def _find_cut_point(self) -> int | None:
        """Decide how many buffered samples to process now (None = keep waiting)."""
        sr = self._sample_rate
        buffer = np.concatenate(self._audio_buffer)

        if self._vad_options is None:
            # Fallback without VAD: fixed volume threshold on the last 0.25s
            tail = buffer[-sr // 4:]
            if self._buffer_duration >= 2.0 and np.sqrt(np.mean(tail ** 2)) < 0.01:
                return len(buffer)
            return len(buffer) if self._buffer_duration >= self._max_duration else None

        speeches = get_speech_timestamps(buffer, self._vad_options)

        if not speeches:
            # Only silence so far: drop it, but keep the last 0.5s in case speech starts there
            keep = buffer[-sr // 2:]
            self._audio_buffer = [keep]
            self._buffer_duration = len(keep) / sr
            return None

        # Speech followed by a pause: the utterance is complete
        self._trailing_silence = (len(buffer) - speeches[-1]["end"]) / sr
        if self._trailing_silence >= self._end_silence:
            return len(buffer)

        # Too long without a pause: cut at the last gap between words, if there is one
        if self._buffer_duration >= self._max_duration:
            if len(speeches) > 1:
                return (speeches[-2]["end"] + speeches[-1]["start"]) // 2
            return self._quietest_point(buffer)

        return None

    def _quietest_point(self, buffer: np.ndarray, window: float = 1.5) -> int:
        """Find the quietest 100ms stretch in the last `window` seconds (likely a gap between words).
        A 100ms average ignores the short dips inside a word, which noise can make look quiet."""
        frame = self._sample_rate // 100  # 10ms
        start = max(0, len(buffer) - int(window * self._sample_rate))
        n = (len(buffer) - start) // frame
        if n < 10:
            return len(buffer)
        energy = np.mean(buffer[start:start + n * frame].reshape(n, frame) ** 2, axis=1)
        smoothed = np.convolve(energy, np.ones(10) / 10, mode="valid")
        quietest = int(np.argmin(smoothed))
        return start + (quietest + 5) * frame

    async def _process_buffer(self, websocket, cut: int):
        """Process the first `cut` samples of the buffer and keep the rest."""
        if not self._audio_buffer:
            return

        buffer = np.concatenate(self._audio_buffer)
        full_audio, rest = buffer[:cut], buffer[cut:]
        self._audio_buffer = [rest] if len(rest) else []
        self._buffer_duration = len(rest) / self._sample_rate

        loop = asyncio.get_running_loop()
        async with self._gpu_lock:
            # From here on this utterance is final: later previews for it are dropped
            result_id = self._next_id
            self._next_id += 1
            self._partial_cache = ("", "")

            # Transcribe (run in thread to not block)
            result = await loop.run_in_executor(
                None,
                self.transcriber.transcribe,
                full_audio,
                self._sample_rate,
            )

            if result is None or not result.text:
                if self._partial_sent_for == result_id:
                    # A preview is on screen but the final pass found nothing: remove it
                    await websocket.send(encode_message({
                        "type": MessageType.RESULT, "id": result_id, "original": "", "translated": "",
                    }))
                return

            await self._send_final(websocket, loop, result_id, result)

    async def _send_final(self, websocket, loop, result_id: int, result):
        """Send the final original immediately, then its translation as a separate message."""
        await websocket.send(encode_message({
            "type": MessageType.RESULT,
            "id": result_id,
            "original": result.text,
            "translated": "",
            "language": result.language,
            "confidence": result.confidence,
            "timestamp": time.time(),
        }))
        print(f"\n[{result.language}] {result.text}")

        if not self.translator:
            return

        try:
            translated = await loop.run_in_executor(
                None,
                self.translator.translate,
                result.text,
                self.target_language,
                result.language,
            )
        except Exception as e:
            print(f"Translation error: {e}")
            return

        if translated:
            await websocket.send(encode_message({
                "type": MessageType.TRANSLATION,
                "id": result_id,
                "translated": translated,
            }))
            print(f"[{self.target_language}] {translated}")

    async def _handle_command(self, websocket, message: str):
        """Handle JSON command."""
        try:
            data = decode_message(message)
            msg_type = data.get("type")

            if msg_type == MessageType.PING:
                await websocket.send(encode_message({"type": MessageType.PONG}))

            elif msg_type == MessageType.SET_CONFIG:
                # Update configuration
                if "translation_provider" in data:
                    provider = data["translation_provider"]
                    if provider != self.translation_provider:
                        try:
                            self.translator = create_translator(provider) if provider else None
                            self.translation_provider = provider
                            print(f"Switched translator to: {provider}")
                        except Exception as e:
                            await websocket.send(encode_message({
                                "type": MessageType.ERROR,
                                "message": f"Failed to switch translator: {e}",
                            }))

                if "target_language" in data:
                    self.target_language = data["target_language"]

                if "source_language" in data:
                    self.source_language = data["source_language"]

        except json.JSONDecodeError as e:
            await websocket.send(encode_message({
                "type": MessageType.ERROR,
                "message": f"Invalid JSON: {e}",
            }))


def main():
    parser = argparse.ArgumentParser(description="Voice Transcription Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--transcriber", default="whisper",
                        choices=["whisper", "parakeet"],
                        help="Transcription engine (parakeet is ~2x faster)")
    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--translator", default="google",
                        choices=["deepl", "google", "local", "none"])
    parser.add_argument("--target-lang", default="ZH-TW", help="Target language for translation")
    parser.add_argument("--source-lang", default="en", help="Source language (or 'auto')")

    args = parser.parse_args()

    server = TranscriptionServer(
        host=args.host,
        port=args.port,
        transcriber_engine=args.transcriber,
        model=args.model,
        device=args.device,
        translation_provider=args.translator if args.translator != "none" else None,
        target_language=args.target_lang,
        source_language=args.source_lang if args.source_lang != "auto" else None,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
