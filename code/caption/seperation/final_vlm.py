import cv2
import numpy as np
import re
import torch
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList
)
from pathlib import Path

# ==========================================
# 1. INPUT IMAGES & NATIVE VLM CONFIG
# ==========================================
TARGET_IMAGES = [
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod38_Angle1_1200x1200_8477ef_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod24_Angle1_1200x1200_7030ff_clean_black_bg.png"
]

LOCAL_MODEL_PATH = r"D:\gen jewels\web scraping\models\Llama-3.2-11B-Vision-Instruct"
NATIVE_VLM_RES = (1120, 1120)  # Exact 2x2 HD tile resolution for Llama-3.2-Vision
PADDING_RATIO = 0.05           # Tight 5% margin around gold pixels

# ==========================================
# 2. GENERATED-ONLY REPETITION PENALTY
# ==========================================
class GeneratedOnlyRepetitionPenalty(LogitsProcessor):
    """
    Applies repetition penalty ONLY to newly generated tokens (ignoring prompt tokens
    and Llama-3.2's <|image|> token 128256). Prevents phrase loops without
    penalizing prompt vocabulary.
    """
    def __init__(self, prompt_length: int, penalty: float = 1.12):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] <= self.prompt_length:
            return scores

        gen_ids = input_ids[:, self.prompt_length:]
        vocab_limit = scores.shape[-1]

        for i in range(gen_ids.shape[0]):
            valid_ids = torch.unique(gen_ids[i][gen_ids[i] < vocab_limit]).unsqueeze(0)
            if valid_ids.numel() > 0:
                score = torch.gather(scores[i:i+1], 1, valid_ids)
                score = torch.where(score < 0, score * self.penalty, score / self.penalty)
                scores[i:i+1].scatter_(1, valid_ids, score)
        return scores

# ==========================================
# 3. HOLE-CLOSED OPENCV GEOMETRY (Dynamic)
# ==========================================
def find_dynamic_zones(mask):
    """
    1. Closes internal filigree/stone holes so porous clusters aren't ignored.
    2. Finds Zone 1 (Thickest Focal Center) via Distance Transform.
    3. Masks out Zone 1 and runs Distance Transform again to find Zone 2 (Side Structure).
    """
    h, w = mask.shape
    max_dim = max(h, w)

    k_size = max(15, int(max_dim * 0.04))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ZONE 1: Primary Focal Mass
    dist_1 = cv2.distanceTransform(closed_mask, cv2.DIST_L2, 5)
    _, max_val_1, _, focal_pt = cv2.minMaxLoc(dist_1)

    # Mask out Zone 1 to isolate remaining side band/arm structure
    structural_mask = closed_mask.copy()
    exclude_radius = int(max_dim * 0.28)
    cv2.circle(structural_mask, focal_pt, exclude_radius, 0, -1)

    # ZONE 2: Secondary Repeating Structure
    dist_2 = cv2.distanceTransform(structural_mask, cv2.DIST_L2, 5)
    _, max_val_2, _, side_pt = cv2.minMaxLoc(dist_2)

    has_side_structure = max_val_2 > (max_dim * 0.015)
    return focal_pt, side_pt, has_side_structure

# ==========================================
# 4. TIGHT CROPPING & 1120x1120 UPSCALING
# ==========================================
def extract_upscaled_pil(image, mask, center_pt, box_ratio):
    h, w = image.shape[:2]
    cx, cy = center_pt
    half_box = int(max(h, w) * box_ratio) // 2

    x1, y1 = max(0, cx - half_box), max(0, cy - half_box)
    x2, y2 = min(w, cx + half_box), min(h, cy + half_box)

    local_mask = mask[y1:y2, x1:x2]
    coords = np.argwhere(local_mask > 0)

    if coords.size > 0:
        sy_min, sx_min = coords.min(axis=0)
        sy_max, sx_max = coords.max(axis=0)
        x1_t, y1_t = x1 + sx_min, y1 + sy_min
        x2_t, y2_t = x1 + sx_max, y1 + sy_max
    else:
        x1_t, y1_t, x2_t, y2_t = x1, y1, x2, y2

    pad = int(max(x2_t - x1_t, y2_t - y1_t) * PADDING_RATIO)
    x1_f, y1_f = max(0, x1_t - pad), max(0, y1_t - pad)
    x2_f, y2_f = min(w, x2_t + pad), min(h, y2_t + pad)

    cropped = image[y1_f:y2_f, x1_f:x2_f]
    side = max(cropped.shape[0], cropped.shape[1])

    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - cropped.shape[0]) // 2
    x_off = (side - cropped.shape[1]) // 2
    canvas[y_off:y_off + cropped.shape[0], x_off:x_off + cropped.shape[1]] = cropped

    upscaled_bgr = cv2.resize(canvas, NATIVE_VLM_RES, interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))

def prepare_full_image_pil(image, mask):
    coords = np.argwhere(mask > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = image[y_min:y_max, x_min:x_max]
    else:
        cropped = image

    side = int(max(cropped.shape[0], cropped.shape[1]) * 1.08)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - cropped.shape[0]) // 2
    x_off = (side - cropped.shape[1]) // 2
    canvas[y_off:y_off + cropped.shape[0], x_off:x_off + cropped.shape[1]] = cropped

    upscaled_bgr = cv2.resize(canvas, NATIVE_VLM_RES, interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))

# ==========================================
# 5. SINGLE-IMAGE VLM INFERENCE HELPER
# ==========================================
def ask_vlm(model, processor, pil_image, prompt, max_tokens=240):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        images=pil_image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    prompt_len = inputs.input_ids.shape[-1]
    logits_processors = LogitsProcessorList([
        GeneratedOnlyRepetitionPenalty(prompt_length=prompt_len, penalty=1.12)
    ])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            logits_processor=logits_processors,
            temperature=None,
            top_p=None
        )

    return processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

# ==========================================
# 6. PURE "DESCRIBE EVERYTHING" PROMPTS (No Leading Questions)
# ==========================================
def build_open_description_prompts(global_report):
    """
    Uses Pass 1 to lock plain-gold vs gemstone reality, then uses an open-ended
    'describe everything visible' command with ZERO leading nouns (no 'birds',
    'figures', or 'hanging drops' in the prompt).
    """
    lower_text = global_report.lower()
    is_plain_gold = bool(re.search(
        r"(no\s+visible\s+(embedded\s+)?(gem)?stones|no\s+(embedded\s+)?(gem)?stones|without\s+(precious\s+)?(gem)?stones|100%\s+solid)",
        lower_text
    ))

    if is_plain_gold:
        material_note = (
            "This piece is 100% solid yellow gold with zero gemstones or crystals. "
            "Describe only the sculpted gold metalwork."
        )
    else:
        material_note = (
            "This piece is yellow gold set with gemstones. "
            "Include the exact colors, shapes, and settings of all visible stones alongside the gold metalwork."
        )

    prompt_center = (
        f"This is a high-resolution close-up of the primary focal section of the jewelry. {material_note}\n"
        "Describe everything physically visible in this image in complete detail, covering the center, "
        "the surrounding frame, and the bottom edge. Name the exact shapes, carvings, or stone arrangements "
        "you see without guessing or mentioning the black background."
    )

    prompt_side = (
        f"This is a high-resolution close-up of the side section of the jewelry. {material_note}\n"
        "Describe everything physically visible in this image in complete detail. Explain how the segments or band "
        "are shaped and connected, what exact pattern or relief is inside them, and what the outer edges look like. "
        "Do not guess unseen objects and do not mention the black background."
    )

    return is_plain_gold, prompt_center, prompt_side

# ==========================================
# 7. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    print("🚀 [1/3] Loading LOCAL Llama-3.2-11B-Vision-Instruct (4-bit NF4) onto GPU...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        local_files_only=True
    )
    print("✅ Model Loaded Successfully!\n")

    PROMPT_GLOBAL = (
        "Describe this full piece of jewelry completely. State the jewelry category, overall silhouette shape, "
        "metal color and finish, and list the exact colors of any embedded gemstones (or explicitly state "
        "'No visible embedded stones' if it is 100% solid metal)."
    )

    for img_path in TARGET_IMAGES:
        if not Path(img_path).exists():
            print(f"❌ File not found: {img_path}")
            continue

        filename = Path(img_path).name
        print("=====================================================================")
        print(f"🧭 [2/3] Dynamic OpenCV Geometry Extraction: {filename}")

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Find exact mathematical centers with hole-closing
        focal_pt, side_pt, has_side = find_dynamic_zones(mask)

        # Generate separate 1120x1120 high-res PIL images in RAM
        pil_full = prepare_full_image_pil(img, mask)
        pil_focal = extract_upscaled_pil(img, mask, focal_pt, box_ratio=0.45)

        print("🔍 [3/3] Running Separate High-Res VLM Inspections...")

        # PASS 1: Global Overview
        global_report = ask_vlm(model, processor, pil_full, PROMPT_GLOBAL, max_tokens=180)
        is_plain_gold, prompt_center, prompt_side = build_open_description_prompts(global_report)
        mode_label = "100% Plain Gold Mode" if is_plain_gold else "Gemstone + Gold Mode"
        print(f"   🔒 Material Gate Locked: {mode_label}")

        # PASS 2: Primary Focal Center (Pure Open Description)
        center_report = ask_vlm(model, processor, pil_focal, prompt_center, max_tokens=240)

        # PASS 3: Side Structure (Pure Open Description)
        if has_side:
            pil_side = extract_upscaled_pil(img, mask, side_pt, box_ratio=0.42)
            side_report = ask_vlm(model, processor, pil_side, prompt_side, max_tokens=240)
        else:
            side_report = "Single compact focal piece (No separate side band structure)."

        print("\n🏛️ FINAL DYNAMIC JEWELRY REPORT:")
        print("---------------------------------------------------------------------")
        print(f"🌐 [GLOBAL OVERVIEW]\n{global_report}\n")
        print(f"🎯 [FOCAL CENTER CLOSE-UP]\n{center_report}\n")
        print(f"🔗 [SIDE BAND / STRUCTURE CLOSE-UP]\n{side_report}")
        print("=====================================================================\n")

if __name__ == "__main__":
    main()