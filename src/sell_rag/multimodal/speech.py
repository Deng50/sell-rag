from __future__ import annotations

import base64
import collections
import io
import queue
import re
import threading
import time
import wave
from difflib import get_close_matches
from typing import Callable, Protocol
from urllib.parse import urlencode

import httpx

from sell_rag.domain import Transcript


class SpeechProvider(Protocol):
    name: str

    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> Transcript: ...
    def synthesize(self, text: str) -> bytes: ...


class FakeSpeech:
    name = "fake-speech-v1"

    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> Transcript:
        text = audio.decode("utf-8", errors="ignore").strip() or "测试语音"
        return Transcript(text=text, confidence=0.99, needs_confirmation=False)

    def synthesize(self, text: str) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * min(16000, max(1600, len(text) * 800)))
        return output.getvalue()


class BaiduSpeech:
    name = "baidu-speech-v1"

    def __init__(self, api_key: str, secret_key: str, confirm_threshold: float = 0.65):
        self.api_key = api_key
        self.secret_key = secret_key
        self.confirm_threshold = confirm_threshold
        self._token: str | None = None
        self._expires_at = 0.0

    def _access_token(self) -> str:
        if self._token and time.time() < self._expires_at:
            return self._token
        response = httpx.post(
            "https://aip.baidubce.com/oauth/2.0/token",
            params={"grant_type": "client_credentials", "client_id": self.api_key,
                    "client_secret": self.secret_key}, timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        self._expires_at = time.time() + int(payload.get("expires_in", 2592000)) - 60
        return self._token

    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> Transcript:
        response = httpx.post(
            "https://vop.baidu.com/server_api",
            json={
                "format": "pcm", "rate": sample_rate, "channel": 1, "cuid": "sell-rag-terminal",
                "token": self._access_token(), "len": len(audio),
                "speech": base64.b64encode(audio).decode("ascii"),
            }, timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("err_no") != 0 or not payload.get("result"):
            return Transcript(text="", confidence=0.0, needs_confirmation=True)
        confidence = payload.get("confidence")
        return Transcript(
            text=payload["result"][0], confidence=float(confidence) if confidence is not None else None,
            needs_confirmation=confidence is not None and float(confidence) < self.confirm_threshold,
            alternatives=payload["result"][1:],
        )

    def synthesize(self, text: str) -> bytes:
        payload = urlencode({
            "tex": text, "tok": self._access_token(), "cuid": "sell-rag-terminal",
            "ctp": 1, "lan": "zh", "spd": 6, "pit": 5, "vol": 5, "per": 0, "aue": 6,
        })
        response = httpx.post(
            "https://tsn.baidu.com/text2audio", content=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30,
        )
        response.raise_for_status()
        if "audio" not in response.headers.get("content-type", ""):
            raise RuntimeError(f"百度语音合成失败: {response.text[:300]}")
        return response.content


class SpeechNormalizer:
    def __init__(self, hotwords: list[str] | None = None, confirm_threshold: float = 0.65):
        self.hotwords = hotwords or []
        self.confirm_threshold = confirm_threshold

    def normalize(self, transcript: Transcript) -> Transcript:
        text = re.sub(r"\s+", " ", transcript.text).strip("，。！？ ")
        alternatives = list(transcript.alternatives)
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            match = get_close_matches(token, self.hotwords, n=2, cutoff=0.72)
            if len(match) == 1 and match[0] != token:
                text = text.replace(token, match[0])
            elif len(match) > 1:
                alternatives.extend(match)
        ambiguous = len(set(alternatives)) > 1
        low_confidence = transcript.confidence is not None and transcript.confidence < self.confirm_threshold
        return transcript.model_copy(update={
            "text": text, "alternatives": list(dict.fromkeys(alternatives)),
            "needs_confirmation": transcript.needs_confirmation or low_confidence or ambiguous,
        })


class StreamingTTSPlayer:
    """Sentence-level producer/consumer playback with cooperative barge-in."""

    def __init__(self, provider: SpeechProvider, play: Callable[[bytes], None],
                 stop_playback: Callable[[], None] | None = None):
        self.provider = provider
        self.play = play
        self.stop_playback = stop_playback
        self._stop = threading.Event()
        self._threads: tuple[threading.Thread, ...] = ()
        self._play_lock = threading.Lock()
        self.last_error: Exception | None = None

    def speak(self, text: str) -> None:
        self.barge_in()
        stop = self._stop = threading.Event()
        audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=3)
        self.last_error = None
        sentences = [item.strip() for item in re.split(r"(?<=[。！？；.!?])", text) if item.strip()]

        def enqueue(audio: bytes | None) -> None:
            while not stop.is_set():
                try:
                    audio_queue.put(audio, timeout=0.05)
                    return
                except queue.Full:
                    continue

        def produce() -> None:
            try:
                for sentence in sentences:
                    if stop.is_set():
                        break
                    enqueue(self.provider.synthesize(sentence))
            except Exception as exc:
                self.last_error = exc
            finally:
                enqueue(None)

        def consume() -> None:
            try:
                while not stop.is_set():
                    try:
                        audio = audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if audio is None:
                        break
                    with self._play_lock:
                        if not stop.is_set():
                            self.play(audio)
            except Exception as exc:
                self.last_error = exc
                stop.set()

        self._threads = (threading.Thread(target=produce, daemon=True),
                         threading.Thread(target=consume, daemon=True))
        for thread in self._threads:
            thread.start()

    def barge_in(self) -> None:
        self._stop.set()
        if self.stop_playback:
            self.stop_playback()

    def wait(self, timeout: float = 30) -> bool:
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        return all(not thread.is_alive() for thread in self._threads)


class VADRecorder:
    """16 kHz mono recorder with WebRTC VAD endpoint detection."""

    def __init__(self, aggressiveness: int = 2, silence_ms: int = 700, max_seconds: int = 12):
        self.aggressiveness = aggressiveness
        self.silence_ms = silence_ms
        self.max_seconds = max_seconds

    def record(self) -> bytes:
        import pyaudio
        import webrtcvad

        rate, frame_ms = 16000, 30
        frame_size = rate * frame_ms // 1000
        vad = webrtcvad.Vad(self.aggressiveness)
        audio = pyaudio.PyAudio()
        stream = None
        frames: list[bytes] = []
        pre_roll: collections.deque[bytes] = collections.deque(maxlen=10)
        speech_started, silence_frames = False, 0
        try:
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                                frames_per_buffer=frame_size)
            for _ in range(self.max_seconds * 1000 // frame_ms):
                frame = stream.read(frame_size, exception_on_overflow=False)
                active = vad.is_speech(frame, rate)
                if not speech_started:
                    pre_roll.append(frame)
                    if active:
                        speech_started = True
                        frames.extend(pre_roll)
                else:
                    frames.append(frame)
                    silence_frames = 0 if active else silence_frames + 1
                    if silence_frames * frame_ms >= self.silence_ms:
                        break
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()
        return b"".join(frames)
