from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from gradio_client import Client, handle_file
from PIL import Image, ImageFilter

SPACE = "black-forest-labs/FLUX.1-Kontext-Dev"
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "assets" / "F00_reference.jpg"
OUT = ROOT / "generated_fire_frames"

FRAME_PROMPTS = {
    "F01": (
        "Edit the input photo into ONE single full-screen photorealistic 16:9 corporate safety-training photograph. "
        "Keep exactly the same Vietnamese female instructor: same face, hairstyle, body proportions, navy short-sleeve polo with the same thin yellow reflective stripe, black work trousers, and the same industrial fire-safety training room. "
        "She is now looking at the camera and naturally waving with her right hand. Place the same red 4 kg ABC dry-powder extinguisher upright on the floor beside her. Keep a small controlled flame in the metal fire tray in the left background. "
        "Professional real photography, natural skin and hands, physically correct extinguisher, full-frame sharp deep focus, realistic lighting. "
        "Do not make a storyboard, contact sheet, split screen, collage, table, poster, infographic, illustration, 3D render, CGI, blur padding, vignette, title, caption, logo, number, icon, or any text."
    ),
}


def find_api_name(client: Client) -> str:
    api = client.view_api(return_format="dict")
    print(json.dumps(api, ensure_ascii=False, indent=2, default=str))
    named = api.get("named_endpoints", {}) if isinstance(api, dict) else {}
    if "/infer" in named:
        return "/infer"
    for name, spec in named.items():
        params = spec.get("parameters", []) if isinstance(spec, dict) else []
        if len(params) >= 6:
            return name
    if named:
        return next(iter(named))
    raise RuntimeError("No named Gradio endpoint was exposed by the Space")


def normalize_result(result: object) -> Path:
    first = result[0] if isinstance(result, (tuple, list)) else result
    if isinstance(first, dict):
        value = first.get("path") or first.get("url") or first.get("name")
    else:
        value = first
    if not value:
        raise RuntimeError(f"Could not resolve generated file from result: {result!r}")
    return Path(str(value))


def exact_1920x1080(src: Path, dst: Path) -> None:
    image = Image.open(src).convert("RGB")
    target_ratio = 16 / 9
    ratio = image.width / image.height
    if ratio > target_ratio:
        new_w = round(image.height * target_ratio)
        left = (image.width - new_w) // 2
        image = image.crop((left, 0, left + new_w, image.height))
    elif ratio < target_ratio:
        new_h = round(image.width / target_ratio)
        top = (image.height - new_h) // 2
        image = image.crop((0, top, image.width, top + new_h))
    image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=55, threshold=3))
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, "PNG", compress_level=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(FRAME_PROMPTS), default="F01")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ref_1280 = OUT / "F00_reference_1280x720.jpg"
    Image.open(REF).convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS).save(
        ref_1280, "JPEG", quality=92, subsampling=0
    )

    client = Client(SPACE, verbose=True)
    api_name = find_api_name(client)
    print(f"Using API endpoint: {api_name}")

    frame = args.only
    prompt = FRAME_PROMPTS[frame]
    seed = 240801
    started = time.time()
    result = client.predict(
        handle_file(str(ref_1280)),
        prompt,
        seed,
        False,
        3.0,
        30,
        api_name=api_name,
    )
    print(f"Raw result: {result!r}")
    raw = normalize_result(result)
    raw_copy = OUT / f"{frame}_raw{raw.suffix or '.webp'}"
    shutil.copy2(raw, raw_copy)
    exact_1920x1080(raw_copy, OUT / f"{frame}_1920x1080.png")
    print(f"Generated {frame} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
