"""Manual test script for streaming STT endpoint.

Usage:
    python test_stt_streaming.py --api-key YOUR_API_KEY audio.opus
    python test_stt_streaming.py --api-key YOUR_API_KEY audio.opus --url https://modulate-prototype-apis.com/api/velma-2-stt-streaming
    python test_stt_streaming.py --api-key YOUR_API_KEY audio.opus --disable-speaker-diarization
    python test_stt_streaming.py --api-key YOUR_API_KEY audio.opus --emotion-signal --accent-signal
    python test_stt_streaming.py --api-key YOUR_API_KEY audio.opus --pii-phi-tagging
"""

import argparse
import asyncio
import json
import time

import aiohttp

DEFAULT_URL = "https://modulate-prototype-apis.com/api/velma-2-stt-streaming"
CHUNK_SIZE = 8192


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test a streaming STT endpoint")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Full URL (default: {DEFAULT_URL})")
    parser.add_argument("--disable-speaker-diarization", action="store_true", default=False,
                        help="Disable speaker diarization (enabled by default)")
    parser.add_argument("--emotion-signal", action="store_true", default=False,
                        help="Enable emotion detection")
    parser.add_argument("--accent-signal", action="store_true", default=False,
                        help="Enable accent detection")
    parser.add_argument("--pii-phi-tagging", action="store_true", default=False,
                        help="Enable PII/PHI tagging")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    # Build ws:// or wss:// URL from http:// or https://
    ws_url = args.url
    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[len("https://"):]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[len("http://"):]

    speaker_diarization = not args.disable_speaker_diarization

    sep = "&" if "?" in ws_url else "?"
    ws_url += (
        f"{sep}api_key={args.api_key}"
        f"&speaker_diarization={str(speaker_diarization).lower()}"
        f"&emotion_signal={str(args.emotion_signal).lower()}"
        f"&accent_signal={str(args.accent_signal).lower()}"
        f"&pii_phi_tagging={str(args.pii_phi_tagging).lower()}"
    )

    print(f"URL:        {ws_url}")
    print(f"File:       {args.audio_file}")
    print(f"Chunk:      {args.chunk_size} bytes")
    print(f"Diarize:    {speaker_diarization}")
    print(f"Emotion:    {args.emotion_signal}")
    print(f"Accent:     {args.accent_signal}")
    print(f"PII/PHI:    {args.pii_phi_tagging}")
    print()

    utterances: list[dict] = []
    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:

            async def send_audio() -> None:
                # Pace audio at approximately real-time playback speed
                # (~32kbps estimate → 4000 bytes/sec)
                seconds_per_chunk = args.chunk_size / 4000

                total_bytes = 0
                chunk_count = 0
                with open(args.audio_file, "rb") as f:
                    while True:
                        chunk = f.read(args.chunk_size)
                        if not chunk:
                            break
                        chunk_count += 1
                        total_bytes += len(chunk)
                        await ws.send_bytes(chunk)
                        await asyncio.sleep(seconds_per_chunk)

                await ws.send_str("")
                elapsed = time.perf_counter() - start
                print(
                    f"[send] Complete: {chunk_count} chunks, "
                    f"{total_bytes:,} bytes in {elapsed:.2f}s"
                )

            send_task = asyncio.create_task(send_audio())

            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        elapsed = time.perf_counter() - start

                        if data.get("type") == "utterance":
                            u = data["utterance"]
                            utterances.append(u)
                            emotion_str = f" emotion={u['emotion']}" if u.get("emotion") else ""
                            accent_str = f" accent={u['accent']}" if u.get("accent") else ""
                            print(
                                f"[{elapsed:6.2f}s] "
                                f"[speaker {u['speaker']}] ({u['language']}) "
                                f"{u['start_ms']}-{u['start_ms'] + u['duration_ms']}ms"
                                f"{emotion_str}{accent_str}: {u['text']}"
                            )
                        elif data.get("type") == "done":
                            print(
                                f"\n[{elapsed:6.2f}s] Done. "
                                f"Audio duration: {data.get('duration_ms', '?')}ms"
                            )
                            break
                        elif data.get("type") == "error":
                            print(f"\n[{elapsed:6.2f}s] Error: {data.get('error', '?')}")
                            break

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"\nWebSocket error: {ws.exception()}")
                        break
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        print("\nWebSocket closed")
                        break
            finally:
                if not send_task.done():
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass

    total_elapsed = time.perf_counter() - start
    print(f"\nTotal: {len(utterances)} utterances in {total_elapsed:.2f}s")

    if utterances:
        full_text = " ".join(u["text"] for u in utterances)
        print(f"\nFull text:\n  {full_text}")


if __name__ == "__main__":
    asyncio.run(main())
