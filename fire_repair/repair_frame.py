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
OUT_DIR = Path("repair_output")
GUIDANCE = 3.5
STEPS = 32

COMMON = (
    "Edit this exact photograph into one single photorealistic 16:9 fire-safety training video frame. "
    "Keep the same Vietnamese female instructor's identity, black hair, navy polo with the fluorescent-yellow stripe, black trousers, and the same grey indoor training facility unless the instruction explicitly changes the camera crop or equipment. "
    "Use realistic human anatomy, correct hands, physically correct extinguisher valves, handles, hose and nozzle, deep focus, neutral professional lighting, and clean commercial photography. "
    "Do not create a storyboard, collage, split panel, diagram, illustration, 3D render, CGI, blur padding, vignette, captions, logos, readable generated text, extra limbs, extra fingers, fused hands, duplicate equipment, or warped hardware. "
)

REPAIRS: dict[int, str] = {
    3: (
        "Change the instructor from a smiling presentation pose to a serious emergency-withdrawal action. Move her close to the open emergency exit at the right, make her step backward toward the exit, hold the mobile phone to her ear, and lower the extinguisher because she is not attempting suppression. Keep the larger smoky fire safely farther behind her. The exit path must be clear."
    ),
    4: (
        "Turn this into an extinguisher-selection demonstration. Place three clearly different real extinguishers upright in a row on the clean floor: the existing red ABC dry-powder extinguisher with black hose, a red CO2 extinguisher with a large black horn, and a water-or-foam extinguisher with a different hose assembly. The instructor points to and selects the ABC powder extinguisher. Reduce the background training flame to very small or remove it."
    ),
    5: (
        "Change the camera to an extreme close-up of the top of the ABC extinguisher. Fill most of the frame with the realistic pressure gauge, intact safety pin, tamper seal, squeeze lever, valve and hose connection. Show two correct hands inspecting them. The pressure-gauge needle must be centered in the green operating range. Crop out most of the instructor's body and remove the fire from view."
    ),
    7: (
        "Pull the camera far back into a wide full-body safety-positioning shot. Place the instructor in a stable stance with the open emergency exit directly behind her. Put the small metal fire tray three to four meters in front of her. Make the light smoke drift away from her toward the fire so she is clearly upwind. She holds the extinguisher ready but does not discharge it."
    ),
    8: (
        "Change to an extreme close-up of only the extinguisher head and the instructor's hands. One hand firmly holds the carrying handle while the other hand breaks the tamper seal and pulls the metal safety pin completely out through its ring. Show an accurate pressure gauge, valve, lever, pin and hose connection. Crop out the face and fire."
    ),
    9: (
        "Change the instructor's grip into the correct ready position after the safety pin has been removed. One hand rests on the squeeze lever and carrying handle; the other hand holds the black hose close to the nozzle. Point the nozzle safely downward and away from people. Do not discharge powder. Show both hands and the hose clearly in a medium close-up."
    ),
    14: (
        "Extinguish the training fire completely. Remove every visible flame from the metal tray and replace it with only a small amount of pale residual smoke. Show the instructor performing one final short powder sweep over the cold tray, with a light white stream at the base. The result must look calm and fully extinguished."
    ),
    15: (
        "Change the fire tray into a shallow liquid-fire training pan viewed from a low side angle. The instructor applies a soft dry-powder stream at a low, grazing angle across the liquid surface so the powder gently blankets it. Do not shoot a strong jet downward, do not splash liquid, and do not spread the flame. Keep the flame small and controlled."
    ),
    17: (
        "Replace the ABC powder extinguisher with a physically correct red CO2 extinguisher that has no pressure gauge and has a large black discharge horn. Remove the active fire. The instructor holds the cylinder by its carrying handle and grips only the insulated handle behind the horn; her hands do not touch the cold horn. Add a light natural frost coating on the horn."
    ),
    18: (
        "Turn this into a confined-room evacuation scene. Add two adult trainees walking calmly out through the open emergency exit. The instructor stands close to the doorway, gestures them out first, and then holds the extinguisher with her back toward the exit. Keep only light distant smoke and a clear escape route; remove the active flame from the foreground."
    ),
    20: (
        "Change to a post-extinguishment monitoring scene. Remove all visible flames from the metal tray and leave only dark smoldering material with one thin wisp of pale smoke. Place the instructor at a safe distance near the exit, watching the tray attentively while keeping the extinguisher ready."
    ),
    21: (
        "Turn the room into an orderly extinguisher-maintenance area. Remove the fire and smoke completely. Show the instructor placing the used extinguisher into a separate service rack with a plain blank red maintenance tag. Place one fully charged replacement extinguisher upright beside it, with an intact pin and normal gauge."
    ),
    22: (
        "Create a clean medium hero shot for a PASS-method recap. Move the instructor to the right side of the frame and leave generous empty grey space on the left for later editorial text. Show the removed safety pin clearly in one hand; the other hand controls the black hose with the nozzle pointed low toward a safe base-of-fire direction, while the squeeze lever is ready. Remove the active fire."
    ),
    23: (
        "Create a safe closing hero shot. Remove the extinguisher that is incorrectly standing inside the fire tray. Extinguish the tray completely so it is cold, empty of flames and smoke. Place one upright red ABC extinguisher safely on the floor beside the instructor. Keep her natural thumbs-up and friendly professional smile, with the emergency exit visible behind her and clean empty space for later closing text."
    ),
}


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


def find_input(frame_no: int) -> Path:
    names = [f"F{frame_no:02d}_1920x1080.png", f"F{frame_no}_1920x1080.png"]
    roots = [Path("baseline_missing"), Path("baseline_main")]
    for root in roots:
        for name in names:
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"Baseline F{frame_no:02d} not found")


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
    image = ImageEnhance.Sharpness(image).enhance(1.05)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "PNG", compress_level=3)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <frame-number>")
    frame_no = int(sys.argv[1])
    if frame_no not in REPAIRS:
        raise SystemExit(f"No repair prompt for F{frame_no:02d}")
    source_image = find_input(frame_no)
    prompt = COMMON + REPAIRS[frame_no]
    seed = 53000 + frame_no
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Client(SPACE, verbose=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            started = time.time()
            result = client.predict(
                input_image=handle_file(str(source_image)),
                prompt=prompt,
                seed=seed,
                randomize_seed=False,
                guidance_scale=GUIDANCE,
                steps=STEPS,
                api_name="/infer",
            )
            elapsed = round(time.time() - started, 3)
            generated = extract_path(result)
            raw_suffix = generated.suffix or ".webp"
            raw_target = OUT_DIR / f"F{frame_no:02d}_repair_source{raw_suffix}"
            shutil.copy2(generated, raw_target)
            final_target = OUT_DIR / f"F{frame_no:02d}_REPAIRED_1920x1080.png"
            save_exact(generated, final_target)
            metadata = {
                "frame": f"F{frame_no:02d}",
                "baseline": str(source_image),
                "seed": seed,
                "guidance_scale": GUIDANCE,
                "steps": STEPS,
                "attempt": attempt,
                "elapsed_seconds": elapsed,
                "prompt": prompt,
            }
            (OUT_DIR / f"F{frame_no:02d}_repair_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
            return
        except Exception as exc:
            last_error = exc
            (OUT_DIR / f"F{frame_no:02d}_repair_attempt_{attempt}_error.txt").write_text(repr(exc), encoding="utf-8")
            time.sleep(7 * attempt)
            client = Client(SPACE, verbose=True)
    raise RuntimeError(f"Repair failed after retries: {last_error!r}")


if __name__ == "__main__":
    main()
