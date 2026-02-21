"""Manual test script for batch STT endpoint.

Usage:
    python test_stt_batch.py --api-key YOUR_API_KEY audio.opus
    python test_stt_batch.py --api-key YOUR_API_KEY audio.opus --url https://modulate-prototype-apis.com/api/velma-2-stt-batch
    python test_stt_batch.py --api-key YOUR_API_KEY audio.opus --disable-speaker-diarization
    python test_stt_batch.py --api-key YOUR_API_KEY audio.opus --pii-phi-tagging
    python test_stt_batch.py --api-key YOUR_API_KEY audio.opus --emotion-signal --accent-signal
"""

import argparse
import asyncio
import time

import aiohttp

DEFAULT_URL = "https://modulate-prototype-apis.com/api/velma-2-stt-batch"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test a batch STT endpoint")
    parser.add_argument("audio_file", help="Path to audio file (supported: aac, aiff, flac, mp3, mp4, mov, ogg, opus, wav, webm)")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Full URL (default: {DEFAULT_URL})")
    parser.add_argument("--disable-speaker-diarization", action="store_true", default=False)
    parser.add_argument("--emotion-signal", action="store_true", default=False)
    parser.add_argument("--accent-signal", action="store_true", default=False)
    parser.add_argument("--pii-phi-tagging", action="store_true", default=False)
    args = parser.parse_args()

    print(f"URL:  {args.url}")
    print(f"File: {args.audio_file}")

    headers = {"X-API-Key": args.api_key}

    speaker_diarization = not args.disable_speaker_diarization

    data = aiohttp.FormData()
    data.add_field(
        "upload_file",
        open(args.audio_file, "rb"),
        filename=args.audio_file.rsplit("/", 1)[-1],
        content_type="application/octet-stream",
    )
    data.add_field("speaker_diarization", str(speaker_diarization).lower())
    data.add_field("emotion_signal", str(args.emotion_signal).lower())
    data.add_field("accent_signal", str(args.accent_signal).lower())
    data.add_field("pii_phi_tagging", str(args.pii_phi_tagging).lower())

    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        async with session.post(args.url, headers=headers, data=data) as resp:
            elapsed = time.perf_counter() - start
            print(f"\nStatus: {resp.status} ({elapsed:.2f}s)")

            if resp.status != 200:
                print(f"Error: {await resp.text()}")
                return

            result = await resp.json()

    print(f"Duration: {result.get('duration_ms', '?')}ms")
    print(f"\nText:\n  {result.get('text', '')}")

    utterances = result.get("utterances", [])
    if utterances:
        print(f"\nUtterances ({len(utterances)}):")
        for u in utterances:
            parts = [
                f"  [{u['speaker']}] ({u['language']})",
                f"{u['start_ms']}-{u['start_ms'] + u['duration_ms']}ms:",
                u["text"],
            ]
            if u.get("emotion") is not None:
                parts.append(f"[{u['emotion']}]")
            if u.get("accent") is not None:
                parts.append(f"({u['accent']})")
            print(" ".join(parts))


if __name__ == "__main__":
    asyncio.run(main())
