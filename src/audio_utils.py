"""
Audio Utilities -- Week 3

Functions for speech recognition (ASR) and text-to-speech (TTS).

ASR:
  transcribe_with_whisper(audio_path)       - OpenAI Whisper transcription
  transcribe_with_faster_whisper(audio_path) - Faster-Whisper (CTranslate2)

TTS:
  synthesize_with_kokoro(text)     - Kokoro TTS (82M params, fast)
  synthesize_with_edge_tts(text)   - Edge TTS (free Microsoft cloud)
  compare_tts(text)                - Compare TTS outputs
"""

import os
import json
from typing import Dict, List, Optional, Any


def transcribe_with_whisper(
    audio_path: str,
    model_size: str = "base",
) -> Dict[str, Any]:
    """
    Transcribe audio using OpenAI Whisper.

    Args:
        audio_path: Path to audio file (mp3, wav, m4a, webm)
        model_size: Model size ("tiny", "base", "small", "medium", "large")

    Returns:
        Dict with 'text', 'language', 'segments', 'method', 'elapsed_seconds' keys
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Install: pip install openai-whisper\nAlso: brew install ffmpeg (macOS)")

    import time

    print("=" * 55)
    print(f"WHISPER ASR (model: {model_size})")
    print("=" * 55)
    print(f"Audio: {audio_path}")

    start = time.time()
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    elapsed = time.time() - start

    output = {
        "text": result["text"].strip(),
        "language": result.get("language", "unknown"),
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.get("segments", [])
        ],
        "method": f"whisper-{model_size}",
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Language: {output['language']}")
    print(f"  Segments: {len(output['segments'])}")
    print(f"  Transcribed in {elapsed:.1f}s")
    print(f"  Text: {output['text'][:200]}...")
    return output


def transcribe_with_faster_whisper(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Dict[str, Any]:
    """
    Transcribe audio using faster-whisper (CTranslate2 backend).

    Faster-whisper provides 4x speedup over OpenAI Whisper with
    lower memory usage. Supports the turbo model for best speed/accuracy.

    Args:
        audio_path: Path to audio file
        model_size: Model size ("tiny", "base", "small", "medium", "large-v3", "turbo")
        device: "cpu" or "cuda"
        compute_type: "int8", "float16", or "float32"

    Returns:
        Dict with 'text', 'segments', 'method', 'elapsed_seconds' keys
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("Install: pip install faster-whisper")

    import time

    print("=" * 55)
    print(f"FASTER-WHISPER ASR (model: {model_size}, {device}/{compute_type})")
    print("=" * 55)
    print(f"Audio: {audio_path}")

    start = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(audio_path)

    segments = []
    full_text_parts = []
    for segment in segments_iter:
        segments.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        })
        full_text_parts.append(segment.text.strip())

    elapsed = time.time() - start
    full_text = " ".join(full_text_parts)

    output = {
        "text": full_text,
        "language": info.language if info else "unknown",
        "language_probability": round(info.language_probability, 3) if info else 0,
        "segments": segments,
        "method": f"faster-whisper-{model_size}",
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Language: {output['language']} (prob: {output['language_probability']})")
    print(f"  Segments: {len(segments)}")
    print(f"  Transcribed in {elapsed:.1f}s")
    print(f"  Text: {full_text[:200]}...")
    return output


def synthesize_with_kokoro(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    save_path: Optional[str] = "outputs/kokoro_output.wav",
) -> Dict[str, Any]:
    """
    Synthesize speech using Kokoro TTS.

    Kokoro is a lightweight (82M params) TTS model with the highest
    MOS score (4.2) among open-source models. Extremely fast on CPU.

    Args:
        text: Text to synthesize
        voice: Voice name (e.g., "af_heart", "am_adam", "bf_emma")
        speed: Speech speed multiplier
        save_path: Path to save WAV output (None to skip)

    Returns:
        Dict with 'audio_path', 'method', 'duration_seconds', 'elapsed_seconds' keys
    """
    try:
        import kokoro
    except ImportError:
        raise ImportError(
            "Install: pip install kokoro\n"
            "Also: pip install soundfile  (for WAV output)"
        )

    import time

    print("=" * 55)
    print(f"KOKORO TTS (voice: {voice}, speed: {speed}x)")
    print("=" * 55)
    print(f"Text: {text[:100]}...")

    start = time.time()

    pipeline = kokoro.KPipeline(lang_code="a")  # 'a' for American English
    generator = pipeline(text, voice=voice, speed=speed)

    # Collect all audio chunks
    all_audio = []
    for i, (gs, ps, audio) in enumerate(generator):
        all_audio.append(audio)

    elapsed = time.time() - start

    result = {
        "method": "kokoro",
        "voice": voice,
        "elapsed_seconds": round(elapsed, 2),
        "chunks": len(all_audio),
    }

    if save_path and all_audio:
        import soundfile as sf
        import numpy as np

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        combined = np.concatenate(all_audio)
        sf.write(save_path, combined, 24000)
        result["audio_path"] = save_path
        result["duration_seconds"] = round(len(combined) / 24000, 2)
        print(f"  Generated {result['duration_seconds']}s audio in {elapsed:.1f}s")
        print(f"  Saved to: {save_path}")
    else:
        print(f"  Generated {len(all_audio)} chunks in {elapsed:.1f}s")

    return result


def synthesize_with_edge_tts(
    text: str,
    voice: str = "en-US-AriaNeural",
    save_path: Optional[str] = "outputs/edge_tts_output.mp3",
) -> Dict[str, Any]:
    """
    Synthesize speech using Microsoft Edge TTS (free, cloud-based).

    Edge TTS provides high-quality neural voices for free via the
    Microsoft Edge read-aloud API. No API key needed.

    Args:
        text: Text to synthesize
        voice: Voice name (e.g., "en-US-AriaNeural", "en-US-GuyNeural")
        save_path: Path to save MP3 output

    Returns:
        Dict with 'audio_path', 'method', 'elapsed_seconds' keys
    """
    try:
        import edge_tts
    except ImportError:
        raise ImportError("Install: pip install edge-tts")

    import asyncio
    import time

    print("=" * 55)
    print(f"EDGE TTS (voice: {voice})")
    print("=" * 55)
    print(f"Text: {text[:100]}...")

    start = time.time()

    async def _synthesize():
        communicate = edge_tts.Communicate(text, voice)
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            await communicate.save(save_path)

    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        loop.run_until_complete(_synthesize())
    except RuntimeError:
        asyncio.run(_synthesize())

    elapsed = time.time() - start

    result = {
        "audio_path": save_path,
        "method": "edge-tts",
        "voice": voice,
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Synthesized in {elapsed:.1f}s")
    if save_path:
        print(f"  Saved to: {save_path}")

    return result


def compare_tts(
    text: str,
    save_dir: str = "outputs",
) -> Dict[str, Any]:
    """
    Compare TTS outputs from available engines.

    Args:
        text: Text to synthesize
        save_dir: Directory to save audio files

    Returns:
        Dict with results from each TTS engine
    """
    print("=" * 55)
    print("TTS COMPARISON")
    print("=" * 55)
    print(f"Text: {text[:100]}...\n")

    results = {}

    # Try Kokoro
    try:
        results["kokoro"] = synthesize_with_kokoro(
            text, save_path=os.path.join(save_dir, "tts_kokoro.wav")
        )
    except (ImportError, Exception) as e:
        print(f"  Kokoro failed: {e}")
        results["kokoro"] = {"error": str(e)}

    # Try Edge TTS
    try:
        results["edge_tts"] = synthesize_with_edge_tts(
            text, save_path=os.path.join(save_dir, "tts_edge.mp3")
        )
    except (ImportError, Exception) as e:
        print(f"  Edge TTS failed: {e}")
        results["edge_tts"] = {"error": str(e)}

    print(f"\n--- Comparison ---")
    for method, res in results.items():
        if "error" in res:
            print(f"  {method}: ERROR - {res['error']}")
        else:
            print(f"  {method}: {res['elapsed_seconds']:.1f}s")

    return results
