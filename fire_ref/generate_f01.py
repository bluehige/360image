from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file
from PIL import Image, ImageEnhance

SPACE = "black-forest-labs/FLUX.1-Kontext-Dev"
REF_PATH = Path("fire_ref/F00_ref_256.jpg")
OUT_DIR = Path("fire_output")
PROMPT = (
    "Edit the supplied reference photograph into ONE single full-frame photorealistic 16:9 corporate fire-safety training photograph. "
    "Preserve the exact same Vietnamese female instructor: same face, hairstyle, age, body proportions, navy short-sleeve polo with one thin fluorescent-yellow reflective stripe, black cargo trousers, and the same indoor fire-safety training room. "
    "Preserve the same realistic red 4 kg ABC dry-powder fire extinguisher with black hose and correct mechanical structure. "
    "The instructor looks directly at the camera and raises her right hand in a natural friendly wave. Place the extinguisher upright on the floor immediately beside her. "
    "Keep only a small controlled flame in the rectangular metal training fire tray in the left background. "
    "Use realistic commercial photography, natural skin texture, anatomically correct hands, deep focus, sharp details across the frame, neutral professional lighting. "
    "Do not add text, captions, signs, logos, posters, infographics, split panels, storyboard layout, collage, blur padding, vignette, illustration, 3D render, CGI, painterly texture, extra people, oversized fire, or distorted equipment."
)


def extract_path(value: Any) -> Path:
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict):
        for key in ("path", "name", "url"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return Path(candidate)
    if isinstance(value, (list, tuple)) and value:
        return extract_path(value[0])
    raise TypeError(f"Unsupported result: {value!r}")


def save_exact(source: Path, target: Path) -> None:
    image = Image.open(source).convert("RGB")
    ratio = image.width / image.height
    target_ratio = 16 / 9
    if ratio > target_ratio:
        width = round(image.height * target_ratio)
        left = (image.width - width) // 2
        image = image.crop((left, 0, left + width, image.height))
    elif ratio < target_ratio:
        height = round(image.width / target_ratio)
        top = (image.height - height) // 2
        image = image.crop((0, top, image.width, top + height))
    image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
    image = ImageEnhance.Sharpness(image).enhance(1.06)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "PNG", compress_level=3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Client(SPACE, verbose=True)
    try:
        schema = client.view_api(return_format="dict", print_info=False)
        (OUT_DIR / "api_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        (OUT_DIR / "api_schema_error.txt").write_text(repr(exc), encoding="utf-8")

    started = time.time()
    result = client.predict(
        input_image=handle_file(str(REF_PATH)),
        prompt=PROMPT,
        seed=42001,
        randomize_seed=False,
        guidance_scale=2.5,
        steps=28,
        api_name="/infer",
    )
    elapsed = round(time.time() - started, 3)
    (OUT_DIR / "raw_result.txt").write_text(repr(result), encoding="utf-8")
    save_exact(extract_path(result), OUT_DIR / "F01_1920x1080.png")
    metadata = {
        "space": SPACE,
        "seed": 42001,
        "guidance_scale": 2.5,
        "steps": 28,
        "elapsed_seconds": elapsed,
        "prompt": PROMPT,
    }
    (OUT_DIR / "F01_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
