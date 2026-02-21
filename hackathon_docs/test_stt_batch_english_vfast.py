"""Manual test script for English batch STT (vfast) endpoint.

Usage:
    python test_stt_batch_english_vfast.py --api-key YOUR_API_KEY audio.opus
    python test_stt_batch_english_vfast.py --api-key YOUR_API_KEY audio.opus --url https://modulate-prototype-apis.com/api/velma-2-stt-batch-english-vfast
"""

import argparse
import asyncio
import time

import aiohttp

DEFAULT_URL = "https://modulate-prototype-apis.com/api/velma-2-stt-batch-english-vfast"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test the English batch STT (vfast) endpoint")
    parser.add_argument("audio_file", help="Path to audio file (.opus format)")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Full URL (default: {DEFAULT_URL})")
    args = parser.parse_args()

    print(f"URL:  {args.url}")
    print(f"File: {args.audio_file}")

    headers = {"X-API-Key": args.api_key}

    data = aiohttp.FormData()
    data.add_field(
        "upload_file",
        open(args.audio_file, "rb"),
        filename=args.audio_file.rsplit("/", 1)[-1],
        content_type="application/octet-stream",
    )

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


if __name__ == "__main__":
    asyncio.run(main())
