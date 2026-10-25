import os
import sys
import re
import warnings
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import MllamaForConditionalGeneration
    ModelClass = MllamaForConditionalGeneration
except ImportError:
    from transformers import AutoModelForVision2Seq
    ModelClass = AutoModelForVision2Seq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration & Paths ---
MODEL_DIR = r"D:\gen jewels\web scraping\models\Llama-3.2-11B-Vision-Instruct"
TARGET_IMAGE = r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod24_Angle1_1200x1200_7030ff_clean_black_bg.png"
SAVE_DEBUG_CROPS = True

# =====================================================================
# NEUTRAL DYNAMIC VISUAL GROUNDING PROMPTS (ZERO LEAKAGE)
# =====================================================================

PROMPT_GROUNDING_AND_MACRO = """You are a jewelry visual inspection system.
Inspect this full jewelry piece on a black background.
1. State the Jewelry Category, Silhouette Shape, Gold Tone, and Relief Workmanship.
2. Propose normalized bounding coordinates [ymin, xmin, ymax, xmax] (0 to 1000 scale) for these 3 structural zones:
   - ZONE_UPPER: Upper terminal, top stud, or upper arm section.
   - ZONE_MID_SIDE: Mid-collar band, flanking side section, or ring shank.
   - ZONE_CENTER_BOTTOM: Main central focal pendant/cluster and bottom hanging fringe.

OUTPUT FORMAT:
Macro_Overview: [Category, shape, metal finish, and relief style]
ZONE_UPPER: [ymin, xmin, ymax, xmax]
ZONE_MID_SIDE: [ymin, xmin, ymax, xmax]
ZONE_CENTER_BOTTOM: [ymin, xmin, ymax, xmax]"""

PROMPT_ZOOM_UPPER = """Inspect this zoomed-in UPPER / TERMINAL jewelry section.
Describe ONLY visible physical shapes and colors:
- Upper_Gemstones: State color, cut, and halo setting of any gemstones in this section, or 'None'.
- Upper_Sculpted_Figures: Describe any sculpted standing figures, dancers, seated figures, birds, or leaf patterns."""

PROMPT_ZOOM_MID_SIDE = """Inspect this zoomed-in MID-BAND / SIDE COLLAR section.
Describe ONLY visible physical shapes and colors:
- Side_Medallions_And_Stones: Describe any circular stone-inlaid medallions or accent stones along this band, or 'None'.
- Repeated_Band_Carvings: Look closely inside each repeated arched niche/frame along the curved band. Describe the figure posture (seated in meditation/lotus posture, standing, or abstract floral pattern)."""

PROMPT_ZOOM_CENTER_BOTTOM = """Inspect this zoomed-in CENTER FOCAL & LOWER FRINGE section from center down to the lowest drop tip.
Describe ONLY visible physical shapes and colors:
- Center_Focal_Motif: Describe the main central gemstone cluster or sculpted relief motif and its flanking accents.
- Lower_Sub_Pendant: Inspect the small gold ornament or figurine suspended directly below the central cluster. Describe its exact shape, posture, or state 'None'.
- Bottom_Hanging_Fringe: Describe the hanging elements along the bottom rim (gold beads/ghungroos, gemstone droplets, material, and color), or state 'None'."""


# =====================================================================
# OPENCV GOLD-CONTOUR BOUNDING & FULL-VERTICAL DROP ENFORCEMENT
# =====================================================================

def get_gold_mask_bounds(np_img: np.ndarray):
    mask = np.any(np_img > 15, axis=-1)
    coords = np.argwhere(mask)
    if coords.size == 0:
        h, w = np_img.shape[:2]
        return mask, 0, 0, w, h
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    return mask, int(xmin), int(ymin), int(xmax), int(ymax)


def snap_box_to_gold_pixels(pil_img: Image.Image, mask: np.ndarray, box_coords: tuple, pad_ratio: float = 0.04) -> Image.Image:
    w, h = pil_img.size
    x1, y1, x2, y2 = [max(0, min(val, lim)) for val, lim in zip(box_coords, [w, h, w, h])]
    if x2 <= x1 + 20 or y2 <= y1 + 20:
        return pil_img.resize((896, 896), Image.Resampling.LANCZOS)

    sub_mask = mask[y1:y2, x1:x2]
    coords = np.argwhere(sub_mask)

    if coords.size > 50:
        sy_min, sx_min = coords.min(axis=0)
        sy_max, sx_max = coords.max(axis=0)
        x1, y1, x2, y2 = x1 + int(sx_min), y1 + int(sy_min), x1 + int(sx_max), y1 + int(sy_max)

    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)

    crop = pil_img.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    side = max(cw, ch)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return canvas.resize((896, 896), Image.Resampling.LANCZOS)


def build_grounded_crops(pil_img: Image.Image):
    """Generates 3 anatomical zooms with bottom-edge locking to prevent drop truncation."""
    np_img = np.array(pil_img)
    mask, xmin, ymin, xmax, ymax = get_gold_mask_bounds(np_img)
    w, h = pil_img.size
    bw, bh = max(1, xmax - xmin), max(1, ymax - ymin)

    # Check for U-shaped hollow collar vs solid profile
    upper_center_slice = mask[ymin:ymin + int(bh * 0.45), xmin + int(bw * 0.38):xmin + int(bw * 0.62)]
    is_hollow_u_shape = (np.mean(upper_center_slice) < 0.15)

    if is_hollow_u_shape:
        fb_upper = (xmin, ymin, xmin + int(bw * 0.36), ymin + int(bh * 0.52))
        fb_mid = (xmin + int(bw * 0.04), ymin + int(bh * 0.38), xmin + int(bw * 0.46), ymin + int(bh * 0.86))
        # Lock bottom boundary to the lowest gold pixel (ymax) with vertical padding
        fb_center = (xmin + int(bw * 0.22), ymin + int(bh * 0.45), xmin + int(bw * 0.78), min(h, ymax + int(bh * 0.04)))
    else:
        fb_upper = (xmin, ymin, xmin + int(bw * 0.45), ymin + int(bh * 0.55))
        fb_mid = (xmin, ymin + int(bh * 0.15), xmin + int(bw * 0.48), ymin + int(bh * 0.80))
        fb_center = (xmin + int(bw * 0.18), ymin + int(bh * 0.15), xmin + int(bw * 0.82), min(h, ymax + int(bh * 0.04)))

    crop_upper = snap_box_to_gold_pixels(pil_img, mask, fb_upper)
    crop_mid = snap_box_to_gold_pixels(pil_img, mask, fb_mid)
    crop_center = snap_box_to_gold_pixels(pil_img, mask, fb_center)

    if SAVE_DEBUG_CROPS:
        crop_upper.save("grounded_crop_1_upper.png")
        crop_mid.save("grounded_crop_2_mid_side.png")
        crop_center.save("grounded_crop_3_center_bottom.png")

    return crop_upper, crop_mid, crop_center


# =====================================================================
# INFERENCE & CLEAN PARSER
# =====================================================================

def query_vlm(model, processor, image: Image.Image, prompt_text: str, max_tokens: int = 240) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    formatted = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=formatted, return_tensors="pt").to("cuda:0")

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    in_len = inputs["input_ids"].shape[1]
    raw = processor.decode(out_ids[0][in_len:], skip_special_tokens=True).strip()
    raw = re.sub(r"\(\s*e\.g\.[^)]*\)", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\b(likely|possibly|perhaps|probably|appears to be)\s+", "", raw, flags=re.IGNORECASE)
    return raw.strip()


def main():
    print("🚀 [1/4] Loading Llama-3.2-11B-Vision-Instruct (4-bit NF4) onto GPU 0...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_image_tiles"):
        processor.image_processor.max_image_tiles = 4

    model = ModelClass.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()

    if not os.path.exists(TARGET_IMAGE):
        print(f"❌ Target image path does not exist: {TARGET_IMAGE}")
        return

    raw_image = Image.open(TARGET_IMAGE).convert("RGB")

    print(f"🧭 [2/4] Stage 1: Macro Scan & Bottom-Locked Slicing on '{Path(TARGET_IMAGE).name}'...")
    macro_overview = query_vlm(model, processor, raw_image, PROMPT_GROUNDING_AND_MACRO, max_tokens=180)
    crop_upper, crop_mid, crop_center = build_grounded_crops(raw_image)

    print("🔍 [3/4] Stage 2: Scanning 3 High-Res (896x896) Grounded Zones...")
    res_upper = query_vlm(model, processor, crop_upper, PROMPT_ZOOM_UPPER, max_tokens=220)
    res_mid = query_vlm(model, processor, crop_mid, PROMPT_ZOOM_MID_SIDE, max_tokens=220)
    res_center = query_vlm(model, processor, crop_center, PROMPT_ZOOM_CENTER_BOTTOM, max_tokens=240)

    print("\n" + "=" * 85)
    print("🏛️ VISUAL GROUNDING MULTI-ZONE VLM REPORT (READY FOR DOWNSTREAM LLM):")
    print("=" * 85)
    print("➤ [MACRO OVERVIEW]:")
    print(macro_overview)
    print("-" * 85)
    print("➤ [ZONE 1: UPPER / TERMINAL HIGH-RES SCAN]:")
    print(res_upper)
    print("-" * 85)
    print("➤ [ZONE 2: MID-BAND / SIDE COLLAR HIGH-RES SCAN]:")
    print(res_mid)
    print("-" * 85)
    print("➤ [ZONE 3: CENTER FOCAL & BOTTOM FRINGE HIGH-RES SCAN]:")
    print(res_center)
    print("=" * 85)

if __name__ == "__main__":
    main()