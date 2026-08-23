from __future__ import annotations

import base64
import collections
import json
import queue
import re
import threading
import time
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
        return text.encode("utf-8")


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
    def __init__(self, hotwords: list[str] | None = None):
        self.hotwords = hotwords or []

    def normalize(self, transcript: Transcript) -> Transcript:
        text = re.sub(r"\s+", "", transcript.text).strip("，。！？ ")
        alternatives = list(transcript.alternatives)
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            match = get_close_matches(token, self.hotwords, n=2, cutoff=0.72)
            if len(match) == 1 and match[0] != token:
                text = text.replace(token, match[0])
            elif len(match) > 1:
                alternatives.extend(match)
        ambiguous = len(set(alternatives)) > 1
        low_confidence = transcript.confidence is not None and transcript.confidence < 0.65
        return transcript.model_copy(update={
            "text": text, "alternatives": list(dict.fromkeys(alternatives)),
            "needs_confirmation": transcript.needs_confirmation or low_confidence or ambiguous,
        })


class StreamingTTSPlayer:
    """Sentence-level producer/consumer playback with cooperative barge-in."""

    def __init__(self, provider: SpeechProvider, play: Callable[[bytes], None]):
        self.provider = provider
        self.play = play
        self._stop = threading.Event()
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=3)

    def speak(self, text: str) -> None:
        self._stop.clear()
        sentences = [item.strip() for item in re.split(r"(?<=[。！？；])", text) if item.strip()]

        def produce() -> None:
            for sentence in sentences:
                if self._stop.is_set():
                    break
                self._queue.put(self.provider.synthesize(sentence))
            self._queue.put(None)

        def consume() -> None:
            while not self._stop.is_set():
                audio = self._queue.get()
                if audio is None:
                    break
                self.play(audio)

        threading.Thread(target=produce, daemon=True).start()
        threading.Thread(target=consume, daemon=True).start()

    def barge_in(self) -> None:
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


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
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                            frames_per_buffer=frame_size)
        frames: list[bytes] = []
        pre_roll: collections.deque[bytes] = collections.deque(maxlen=10)
        speech_started, silence_frames = False, 0
        try:
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
            stream.stop_stream()
            stream.close()
            audio.terminate()
        return b"".join(frames)
