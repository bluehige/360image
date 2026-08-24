from __future__ import annotations

import json
import sys
from pathlib import Path

from gradio_client import Client

from generate_batch import OUT_DIR, SPACE, generate_one


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <frame-number>")
    frame_no = int(sys.argv[1])
    if not 1 <= frame_no <= 23:
        raise SystemExit("Frame must be between 1 and 23")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Client(SPACE, verbose=True)
    item = generate_one(client, frame_no)
    metadata = OUT_DIR / f"single_F{frame_no:02d}_metadata.json"
    metadata.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
