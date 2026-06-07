from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts


OUTPUT_PATH = Path("outputs/current_voice.mp3")
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"


async def synthesize(text: str, voice: str = DEFAULT_VOICE) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%",
        pitch="+0Hz",
    )
    await communicate.save(str(OUTPUT_PATH))


def main() -> None:
    print("Visual Novel voice generator")
    print(f"Voice: {DEFAULT_VOICE}")
    print(f"Output: {OUTPUT_PATH}")
    print("Type text and press Enter. Empty input exits.")

    while True:
        try:
            text = input("\nText> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not text:
            print("Bye.")
            return

        try:
            asyncio.run(synthesize(text))
        except Exception as exc:
            print(f"Failed: {exc}")
            continue

        print(f"Saved and overwritten: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
