from __future__ import annotations

import binascii
import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave
import zlib
from functools import lru_cache
from io import BytesIO
from typing import Any


_SAMPLE_RATE = 16_000


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def _inside_polygon(
    x: float,
    y: float,
    points: list[tuple[float, float]],
) -> bool:
    inside = False
    previous = points[-1]
    for current in points:
        x1, y1 = current
        x2, y2 = previous
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _draw_star(
    pixels: bytearray,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float,
) -> None:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        point_radius = radius if index % 2 == 0 else radius * 0.382
        points.append(
            (
                center_x + math.cos(angle) * point_radius,
                center_y + math.sin(angle) * point_radius,
            )
        )
    for y in range(
        max(0, int(center_y - radius - 1)),
        min(height, int(center_y + radius + 2)),
    ):
        for x in range(
            max(0, int(center_x - radius - 1)),
            min(width, int(center_x + radius + 2)),
        ):
            if _inside_polygon(x + 0.5, y + 0.5, points):
                offset = (y * width + x) * 3
                pixels[offset : offset + 3] = b"\xff\xff\xff"


def _american_flag_png(width: int = 570, height: int = 300) -> bytes:
    red = b"\xb2\x22\x34"
    white = b"\xff\xff\xff"
    blue = b"\x3c\x3b\x6e"
    canton_width = int(width * 0.4)
    canton_height = int(height * 7 / 13)
    pixels = bytearray()
    for y in range(height):
        stripe = min(12, y * 13 // height)
        row = bytearray((red if stripe % 2 == 0 else white) * width)
        if y < canton_height:
            row[: canton_width * 3] = blue * canton_width
        pixels.extend(row)
    unit_x = canton_width / 12
    unit_y = canton_height / 10
    for row_index in range(9):
        columns = range(1, 12, 2) if row_index % 2 == 0 else range(2, 11, 2)
        for column in columns:
            _draw_star(
                pixels,
                width,
                height,
                column * unit_x,
                (row_index + 1) * unit_y,
                min(unit_x, unit_y) * 0.38,
            )
    scanlines = b"".join(
        b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3]
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _note_frequency(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _synthesize_wav(notes: list[tuple[int, float]]) -> bytes:
    samples: list[int] = []
    gap_samples = int(_SAMPLE_RATE * 0.025)
    for midi_note, duration in notes:
        note_samples = max(1, int(_SAMPLE_RATE * duration))
        fade_samples = min(int(_SAMPLE_RATE * 0.012), note_samples // 3)
        frequency = _note_frequency(midi_note)
        for index in range(note_samples):
            envelope = 1.0
            if fade_samples:
                envelope = min(
                    1.0,
                    index / fade_samples,
                    (note_samples - 1 - index) / fade_samples,
                )
            phase = 2.0 * math.pi * frequency * index / _SAMPLE_RATE
            value = (
                math.sin(phase) * 0.72
                + math.sin(phase * 2.0) * 0.18
                + math.sin(phase * 3.0) * 0.10
            )
            samples.append(int(12_000 * envelope * value))
        samples.extend([0] * gap_samples)
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return output.getvalue()


def _ascending_notes() -> list[tuple[int, float]]:
    return [(note, 0.38) for note in (60, 62, 64, 65, 67, 69, 71, 72)]


def _twinkle_notes() -> list[tuple[int, float]]:
    phrase_a = [
        (60, 0.24), (60, 0.24), (67, 0.24), (67, 0.24),
        (69, 0.24), (69, 0.24), (67, 0.48),
        (65, 0.24), (65, 0.24), (64, 0.24), (64, 0.24),
        (62, 0.24), (62, 0.24), (60, 0.48),
    ]
    phrase_b = [
        (67, 0.24), (67, 0.24), (65, 0.24), (65, 0.24),
        (64, 0.24), (64, 0.24), (62, 0.48),
    ]
    return phrase_a + phrase_b + phrase_b + phrase_a


def _spoken_phrase_wav() -> bytes:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        raise RuntimeError("Windows Speech Synthesizer is unavailable")
    script = r"""
& {
  $OutputPath = $env:EMULLM_TEST_SPEECH_OUTPUT
  Add-Type -AssemblyName System.Speech
  $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
  )
  $voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try {
    $voice.SetOutputToWaveFile($OutputPath, $format)
    $voice.SpeakSsml('<speak version="1.0" xml:lang="en-US">The cow jumped over the moon.<break time="1200ms"/>Yesterday, I think.</speak>')
  } finally {
    $voice.Dispose()
  }
}
"""
    with tempfile.TemporaryDirectory(prefix="emullm-speech-") as directory:
        output = f"{directory}\\spoken-phrase.wav"
        env = os.environ.copy()
        env["EMULLM_TEST_SPEECH_OUTPUT"] = output
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"speech synthesis failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        with open(output, "rb") as handle:
            return handle.read()


@lru_cache(maxsize=1)
def test_media_samples() -> dict[str, dict[str, Any]]:
    definitions = [
        (
            "american-flag",
            "american-flag.png",
            "image/png",
            "American flag vision sample",
            _american_flag_png(),
        ),
        (
            "ascending-tones",
            "ascending-tones.wav",
            "audio/wav",
            "Eight ascending notes from C4 through C5",
            _synthesize_wav(_ascending_notes()),
        ),
        (
            "twinkle",
            "twinkle-twinkle-little-star.wav",
            "audio/wav",
            "Public-domain Twinkle, Twinkle, Little Star melody",
            _synthesize_wav(_twinkle_notes()),
        ),
        (
            "spoken-phrase",
            "cow-moon-yesterday-i-think.wav",
            "audio/wav",
            'Speech: "The cow jumped over the moon." Pause. "Yesterday, I think."',
            _spoken_phrase_wav(),
        ),
    ]
    return {
        sample_id: {
            "id": sample_id,
            "name": name,
            "mime_type": mime_type,
            "description": description,
            "data": data,
        }
        for sample_id, name, mime_type, description, data in definitions
    }
