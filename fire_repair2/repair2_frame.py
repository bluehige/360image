from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file
from PIL import Image, ImageEnhance

SPACE = "black-forest-labs/FLUX.1-Kontext-Dev"
OUT_DIR = Path("repair2_output")
GUIDANCE = 4.5
STEPS = 36

PROMPTS: dict[int, str] = {
    3: (
        "Edit this single photorealistic fire-training photograph. Move the same Vietnamese female instructor immediately beside the open emergency exit on the right. Give her a serious alert expression, turn her body toward the exit, show her stepping backward out of the room, and place the mobile phone against her ear. Remove the extinguisher from her hands because she has stopped trying to fight the larger smoky fire behind her. Keep one coherent 16:9 photograph; no text, graphics, collage, illustration or extra limbs."
    ),
    5: (
        "Edit this close-up into a true macro inspection photograph of a fire extinguisher valve. Fill the center with a physically correct round pressure gauge whose needle is clearly inside the green operating zone. Beside it show an intact metal safety pin through the squeeze lever and an intact tamper seal. Show two anatomically correct hands inspecting the gauge and pin. Crop out the face, body and fire. No labels, text, diagrams or split panels."
    ),
    8: (
        "Edit this close-up to show the exact safety-pin removal action. Keep the same realistic red ABC extinguisher head. The left hand firmly holds the carrying handle. The right hand grips the circular metal pin ring and is pulling the long metal safety pin completely out from between the two squeeze levers, breaking a small tamper seal. Show the pin, ring, valve and lever clearly. Crop out the face and fire; no text or graphics."
    ),
    9: (
        "Edit the woman's hand position only. Her left hand must rest correctly on the carrying handle and squeeze lever. Her right hand must firmly hold the black flexible hose near its nozzle, lifting it away from the cylinder. Point the nozzle diagonally downward toward a safe empty floor area. The hose must not dangle loose. Do not discharge powder. Keep the same woman, uniform and room; no text or collage."
    ),
    15: (
        "Edit this into a low side-angle liquid-fire demonstration. Replace the deep metal fire tray with a shallow rectangular liquid-fire training pan containing only a small controlled surface flame. The instructor must hold the hose and send a soft white dry-powder stream almost horizontally across the top of the liquid so it gently blankets the surface. Do not blast downward, splash liquid or enlarge the flame. One realistic 16:9 photograph only."
    ),
    17: (
        "Correct the extinguisher into a physically accurate portable CO2 extinguisher. Use a smooth red high-pressure cylinder with NO pressure gauge, a valve and carrying handle at the top, a short black hose, and one large black conical discharge horn with a clearly visible insulated hand grip behind the horn. Remove the ABC powder hose and pressure-gauge assembly. The instructor holds only the insulated grip and cylinder handle, never touching the horn. Remove all fire; add only very light frost on the horn."
    ),
    20: (
        "Edit this into post-fire monitoring. Remove every visible flame from the metal tray. Leave dark extinguished material and just one thin pale wisp of smoke. Move the instructor to a safe distance near the emergency exit, watching the tray with an alert serious expression while keeping the extinguisher ready. Do not show active spraying. No text or graphics."
    ),
    22: (
        "Create a clean PASS recap hero photograph from this image. Move the same instructor and extinguisher to the right third of the frame and leave the entire left half as natural empty grey training-room wall and floor, with no grey overlay box. Show the removed metal safety pin in her left hand. Her right hand holds the black hose near the nozzle, pointing it low toward an empty safe floor area; the extinguisher rests upright on the floor and the squeeze lever is ready. Remove the active fire. No text, diagrams or panels."
    ),
    23: (
        "Make this closing photograph completely safe. Remove every flame, ember and smoke plume from the metal training tray so the tray is visibly cold and empty. Keep the extinguisher upright on the floor beside the instructor, never inside the tray. Preserve her natural thumbs-up, friendly smile, navy uniform and visible emergency exit. Leave clean natural space for later editing; no text, icons or graphics."
    ),
}

USE_REPAIR1 = {5, 8, 17, 23}


def extract_path(value: Any) -> Path:
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict):
        for key in ("path", "name", "url"):
            val = value.get(key)
            if isinstance(val, str):
                return Path(val)
    if isinstance(value, (list, tuple)) and value:
        return extract_path(value[0])
    raise TypeError(f"Unsupported result: {value!r}")


def find_input(frame_no: int) -> Path:
    if frame_no in USE_REPAIR1:
        hits = list(Path("baseline_repair1").rglob(f"F{frame_no:02d}_REPAIRED_1920x1080.png"))
        if hits:
            return hits[0]
    names = [f"F{frame_no:02d}_1920x1080.png", f"F{frame_no}_1920x1080.png"]
    for root in [Path("baseline_missing"), Path("baseline_main")]:
        for name in names:
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"Source F{frame_no:02d} not found")


def save_exact(source: Path, target: Path) -> None:
    image = Image.open(source).convert("RGB")
    target_ratio = 16 / 9
    ratio = image.width / image.height
    if ratio > target_ratio:
        width = round(image.height * target_ratio)
        left = (image.width - width) // 2
        image = image.crop((left, 0, left + width, image.height))
    elif ratio < target_ratio:
        height = round(image.width / target_ratio)
        top = (image.height - height) // 2
        image = image.crop((0, top, image.width, top + height))
    image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
    image = ImageEnhance.Sharpness(image).enhance(1.05)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "PNG", compress_level=3)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <frame-number>")
    frame_no = int(sys.argv[1])
    prompt = PROMPTS[frame_no]
    source = find_input(frame_no)
    seed = 63000 + frame_no
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Client(SPACE, verbose=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            started = time.time()
            result = client.predict(
                input_image=handle_file(str(source)),
                prompt=prompt,
                seed=seed,
                randomize_seed=False,
                guidance_scale=GUIDANCE,
                steps=STEPS,
                api_name="/infer",
            )
            elapsed = round(time.time() - started, 3)
            generated = extract_path(result)
            suffix = generated.suffix or ".webp"
            raw = OUT_DIR / f"F{frame_no:02d}_repair2_source{suffix}"
            shutil.copy2(generated, raw)
            final = OUT_DIR / f"F{frame_no:02d}_REPAIR2_1920x1080.png"
            save_exact(generated, final)
            meta = {
                "frame": f"F{frame_no:02d}",
                "source": str(source),
                "seed": seed,
                "guidance_scale": GUIDANCE,
                "steps": STEPS,
                "attempt": attempt,
                "elapsed_seconds": elapsed,
                "prompt": prompt,
            }
            (OUT_DIR / f"F{frame_no:02d}_repair2_metadata.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
            return
        except Exception as exc:
            last_error = exc
            (OUT_DIR / f"F{frame_no:02d}_repair2_attempt_{attempt}_error.txt").write_text(repr(exc), encoding="utf-8")
            time.sleep(7 * attempt)
            client = Client(SPACE, verbose=True)
    raise RuntimeError(f"Repair2 failed: {last_error!r}")


if __name__ == "__main__":
    main()
