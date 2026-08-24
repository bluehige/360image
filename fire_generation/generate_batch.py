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
REF_PATH = Path("fire_ref/F00_ref_256.jpg")
OUT_DIR = Path("fire_output")
GUIDANCE = 2.5
STEPS = 28

COMMON = (
    "Edit the supplied reference photograph into ONE single full-frame photorealistic 16:9 corporate fire-safety training photograph. "
    "Preserve the exact same Vietnamese female instructor from the input: same face, black hair tied low, age, body proportions, navy short-sleeve polo with one thin fluorescent-yellow reflective stripe, black cargo trousers, and black safety shoes. "
    "Preserve the same clean indoor fire-safety training center, grey walls and floor, realistic equipment, professional neutral lighting, natural skin texture, anatomically correct hands, and physically correct extinguisher mechanics. "
    "The image must look like a real commercial safety-training video still with deep focus and sharp detail across the full frame. "
    "Do not add any readable text, captions, numbers, logos, posters, infographics, split panels, storyboard layout, collage, blur padding, vignette, illustration, 3D render, CGI, painterly texture, duplicated body parts, extra fingers, fused hands, warped hoses, impossible extinguisher parts, or oversized theatrical flames. "
)

SCENES: dict[int, str] = {
    1: (
        "The instructor looks directly at the camera and raises her right hand in a natural friendly wave while securely holding the same realistic red 4 kg ABC dry-powder extinguisher with her other hand. "
        "A rectangular metal training fire tray with only a small controlled flame remains in the left background. Keep the exit door visible."
    ),
    2: (
        "Wide safety-training scene. The instructor has noticed the small fire tray and reaches to press a red wall-mounted manual fire-alarm call point. "
        "Two adult trainees in the far background calmly move toward the emergency exit. The fire remains small and controlled. Her face and uniform must remain identical to the reference."
    ),
    3: (
        "The controlled fire has grown moderately and darker smoke is increasing. The instructor does not attempt suppression; she backs toward the clearly visible open exit door while holding a mobile phone to call emergency services. "
        "Her escape route is unobstructed. Show one continuous realistic scene, not symbols or graphics."
    ),
    4: (
        "Equipment-selection scene in the same training center. Three real portable extinguishers stand in one row: a red ABC dry-powder extinguisher with black hose, a CO2 extinguisher with a black discharge horn, and a water-or-foam extinguisher with a different hose assembly. "
        "The same instructor clearly selects and lifts the red ABC dry-powder extinguisher. No labels need to be readable."
    ),
    5: (
        "Detailed medium close-up of the same instructor inspecting the red ABC extinguisher before use. Show the intact cylinder, secure hose, tamper seal, safety pin, and a realistic pressure gauge with the needle centered in the green operating zone. "
        "Keep part of her face and identical uniform visible behind the equipment; all hands and hardware must be anatomically and mechanically correct."
    ),
    6: (
        "Full-body side view. The instructor walks toward the small controlled fire tray while carrying the ABC extinguisher by its top handle and gently rocking the cylinder as appropriate for a powder extinguisher. "
        "The exit remains behind her and the walkway is clear. Capture a realistic mid-step action photograph."
    ),
    7: (
        "Wide establishing shot that clearly shows safe positioning: the emergency exit is directly behind the instructor, the small metal fire tray is three to four meters in front of her, and the light smoke drifts away from her toward the fire. "
        "She stands upwind in a stable stance holding the extinguisher ready, with no discharge yet."
    ),
    8: (
        "Extreme close-up of the same red ABC extinguisher head. One anatomically correct hand holds the carrying handle while the other breaks the tamper seal and pulls the metal safety pin completely out of the squeeze lever. "
        "Show accurate pin, ring, valve, gauge, lever, and hose connections; no face is required but retain the same navy sleeve and yellow reflective stripe."
    ),
    9: (
        "Medium close-up of the same instructor after removing the pin. One hand controls the squeeze lever and the other firmly holds the black hose near the nozzle. "
        "The nozzle points safely downward and away from all people. No powder is discharged yet. Keep her exact face and uniform consistent."
    ),
    10: (
        "Side wide shot. From about one and a half to two meters away, the instructor adopts a balanced stance and aims the nozzle precisely at the base of the small flame in the metal training tray. "
        "The extinguisher is upright, the hose is naturally curved, and no discharge has begun."
    ),
    11: (
        "Medium action close-up. The instructor decisively squeezes the extinguisher lever with one hand while holding the hose with the other. "
        "The first compact stream of white dry powder begins at the nozzle and is directed at the base of the fire. Show correct hand placement and realistic pressure discharge."
    ),
    12: (
        "Wide action shot. A continuous white dry-powder stream sweeps horizontally from one side of the metal fire tray to the other, covering the full base of the small fire. "
        "The same instructor maintains a stable stance and correct two-hand control; the flame is becoming smaller."
    ),
    13: (
        "Side medium-wide action shot. While maintaining the dry-powder stream at the fire base, the instructor takes one careful step forward as the flame weakens. "
        "Keep a safe distance, the exit behind her, and show the flame substantially smaller than before."
    ),
    14: (
        "The fire in the metal training tray is fully extinguished: no visible flames, only a small amount of residual pale smoke. "
        "The instructor performs one final brief controlled sweep over the tray before releasing the lever. The scene must look realistic and calm."
    ),
    15: (
        "Low side-view demonstration of a shallow liquid-fire training pan with a small controlled flame. The same instructor applies dry powder gently at a low angle so it settles across and blankets the liquid surface. "
        "Do not show a strong jet striking down into the liquid, splashing liquid, or spreading fire."
    ),
    16: (
        "Electrical-fire safety demonstration in the same facility. The instructor stands beside a realistic electrical training cabinet with a universal yellow electrical-hazard triangle; the main isolator is visibly switched off. "
        "She holds an appropriate extinguisher ready and does not use water. Show only slight residual smoke, no exposed live contact."
    ),
    17: (
        "CO2 extinguisher caution demonstration. The same instructor holds a realistic red CO2 extinguisher by its carrying handle and grips only the insulated handle behind the large black discharge horn. "
        "The horn has a light realistic frost coating, and her other hand does not touch the cold metal or horn. No active fire."
    ),
    18: (
        "Wide interior doorway scene with light distant smoke. The instructor first guides two adult trainees out through the open emergency exit, then remains close to the doorway holding the extinguisher with her back toward the exit. "
        "Everyone moves calmly; the route is clear and she avoids entering dense smoke."
    ),
    19: (
        "A fire behind an interior doorway has become too large to control. The instructor has stopped using the extinguisher, retreats through the exit, pulls the fire door toward closed with one hand, and holds a mobile phone in the other to call emergency services. "
        "Show a safe realistic withdrawal; no readable phone number or graphic icon."
    ),
    20: (
        "Post-extinguishment monitoring scene. From a safe distance near the exit, the same instructor watches the metal tray containing dark smoldering material and only a thin wisp of smoke. "
        "She keeps the extinguisher ready, looks alert, and there are no visible flames."
    ),
    21: (
        "Maintenance-area scene. The instructor places the used extinguisher in a separate service rack with a plain blank red maintenance tag, while a fully charged replacement extinguisher stands ready beside it with its gauge in the normal zone. "
        "Show orderly realistic storage and no readable generated text."
    ),
    22: (
        "Clean medium hero shot for a PASS-method recap. The same instructor correctly holds the red ABC extinguisher: the removed safety pin is clearly visible in one hand, the other hand controls the black hose with the nozzle pointed low toward a safe base-of-fire direction, and the lever is ready. "
        "Leave generous clean negative space on the left for later editorial overlays, but do not generate any text or graphics. No active flame."
    ),
    23: (
        "Closing hero shot in the same training room. The same instructor stands beside an upright red ABC extinguisher and gives a natural thumbs-up with a professional friendly smile. "
        "The emergency exit is visible in the background and the metal training tray is completely cold with no flame and no smoke. Leave clean space for later closing text, but generate no text or symbols."
    ),
}

BATCHES: dict[str, list[int]] = {
    "01-05": [1, 2, 3, 4, 5],
    "06-10": [6, 7, 8, 9, 10],
    "11-15": [11, 12, 13, 14, 15],
    "16-20": [16, 17, 18, 19, 20],
    "21-23": [21, 22, 23],
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
    raise TypeError(f"Unsupported Gradio result: {value!r}")


def save_exact(source: Path, target: Path) -> None:
    image = Image.open(source).convert("RGB")
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
    image = ImageEnhance.Sharpness(image).enhance(1.06)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "PNG", compress_level=3)


def generate_one(client: Client, frame_no: int) -> dict[str, Any]:
    prompt = COMMON + SCENES[frame_no]
    seed = 43000 + frame_no
    error: Exception | None = None
    for attempt in range(1, 4):
        try:
            started = time.time()
            result = client.predict(
                input_image=handle_file(str(REF_PATH)),
                prompt=prompt,
                seed=seed,
                randomize_seed=False,
                guidance_scale=GUIDANCE,
                steps=STEPS,
                api_name="/infer",
            )
            elapsed = round(time.time() - started, 3)
            source = extract_path(result)
            raw_dir = OUT_DIR / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            suffix = source.suffix if source.suffix else ".webp"
            raw_target = raw_dir / f"F{frame_no:02d}_source{suffix}"
            shutil.copy2(source, raw_target)
            final_target = OUT_DIR / f"F{frame_no:02d}_1920x1080.png"
            save_exact(source, final_target)
            return {
                "frame": f"F{frame_no:02d}",
                "seed": seed,
                "attempt": attempt,
                "elapsed_seconds": elapsed,
                "raw_file": str(raw_target),
                "final_file": str(final_target),
                "prompt": prompt,
            }
        except Exception as exc:
            error = exc
            (OUT_DIR / f"F{frame_no:02d}_attempt_{attempt}_error.txt").write_text(repr(exc), encoding="utf-8")
            time.sleep(8 * attempt)
            client = Client(SPACE, verbose=True)
    raise RuntimeError(f"F{frame_no:02d} failed after retries: {error!r}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in BATCHES:
        raise SystemExit(f"Usage: {sys.argv[0]} <{'|'.join(BATCHES)}> ")
    batch = sys.argv[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT_DIR / f"batch_{batch}_metadata.json"
    results: list[dict[str, Any]] = []
    client = Client(SPACE, verbose=True)
    try:
        schema = client.view_api(return_format="dict", print_info=False)
        (OUT_DIR / "api_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        (OUT_DIR / "api_schema_error.txt").write_text(repr(exc), encoding="utf-8")

    for frame_no in BATCHES[batch]:
        item = generate_one(client, frame_no)
        results.append(item)
        metadata_path.write_text(json.dumps({"batch": batch, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)
        time.sleep(4)


if __name__ == "__main__":
    main()
