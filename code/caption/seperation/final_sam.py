import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import cv2
import numpy as np
import re
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
# 1. ALL 5 TARGET IMAGES & MODEL PATHS
# ==========================================
TARGET_IMAGES = [
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod38_Angle1_1200x1200_8477ef_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod24_Angle1_1200x1200_7030ff_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\Candere\Rings\Candere_Rings_Prod1_Angle12_1100x1100_21c7ae_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\Candere\Earrings\Candere_Earrings_Prod163_Angle1_1100x1100_71a1d4_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\CC_Earrings\CC_Earrings_Prod2_Angle12_1200x1200_bd49e8_clean_black_bg.png"
]

LOCAL_MODEL_PATH = r"D:\gen jewels\web scraping\models\Llama-3.2-11B-Vision-Instruct"
SAM_MODEL_DIR = Path(r"D:\gen jewels\web scraping\models\sam2.1-hiera-large")
SAM_PT_PATH = SAM_MODEL_DIR / "sam2.1_hiera_large.pt"
SAM_YAML_NAME = "sam2.1_hiera_l.yaml"

NATIVE_VLM_RES = (1120, 1120)
MACRO_PAD_RATIO = 0.06
MIN_CONTEXT_WINDOW = 380
PAIR_MERGE_DISTANCE = 155

# ==========================================
# 2. VLM REPETITION PENALTY & GPU INFERENCE
# ==========================================
class GeneratedOnlyRepetitionPenalty(LogitsProcessor):
    """Applies repetition penalty strictly to newly generated tokens, ignoring prompt & <|image|> tokens."""
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

def ask_vlm(model, processor, pil_image, prompt, max_tokens=320):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
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
# 3. SAM-2 GPU ENGINE LOADER
# ==========================================
class UniversalSam2GpuGenerator:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.generator = None
        self.mode = None
        self._load_gpu_model()

    def _load_gpu_model(self):
        try:
            import hydra
            from hydra.core.global_hydra import GlobalHydra
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            GlobalHydra.instance().clear()
            hydra.initialize_config_dir(config_dir=str(self.model_dir.resolve()), version_base=None)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam2_model = build_sam2(
                config_file=SAM_YAML_NAME,
                ckpt_path=str(SAM_PT_PATH.resolve()),
                device=device,
                mode="eval"
            )
            self.generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=32,
                points_per_batch=64,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.88,
                crop_n_layers=1,
                min_mask_region_area=120
            )
            self.mode = "native_sam2"
            print("✅ SAM-2.1 Loaded on GPU (Native Meta CUDA Engine)!")
        except ImportError:
            from transformers import pipeline
            self.generator = pipeline(
                "mask-generation",
                model=str(self.model_dir),
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16
            )
            self.mode = "hf_pipeline"
            print("✅ SAM-2.1 Loaded on GPU (HuggingFace Pipeline)!")

    def generate_masks(self, rgb_np: np.ndarray):
        if self.mode == "native_sam2":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                anns = self.generator.generate(rgb_np)
            return [ann["segmentation"].astype(np.uint8) for ann in anns]
        else:
            outputs = self.generator(
                Image.fromarray(rgb_np),
                points_per_batch=64,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.88,
                crops_n_layers=1
            )
            return [np.array(m).astype(np.uint8) for m in outputs.get("masks", [])]

# ==========================================
# 4. SMART CATEGORY & GEOMETRY EXTRACTION
# ==========================================
def detect_category(path_str: str) -> str:
    p = path_str.lower()
    if "necklace" in p or "haram" in p or "choker" in p:
        return "Necklace"
    if "earring" in p or "jhumka" in p or "stud" in p:
        return "Earring"
    if "ring" in p:
        return "Ring"
    if "bangle" in p or "bracelet" in p or "kada" in p:
        return "Bangle"
    if "pendant" in p or "locket" in p:
        return "Pendant"
    return "Jewelry"

def upscale_to_square_pil(cropped_bgr):
    sh, sw = cropped_bgr.shape[:2]
    side = max(sh, sw)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    yo, xo = (side - sh) // 2, (side - sw) // 2
    canvas[yo:yo + sh, xo:xo + sw] = cropped_bgr
    upscaled = cv2.resize(canvas, NATIVE_VLM_RES, interpolation=cv2.INTER_LANCZOS4)
    return upscaled, Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))

def isolate_single_earring_if_pair(image_bgr, mask):
    """
    If an Earring image contains a pair of two disconnected earrings side-by-side,
    isolates the largest single complete earring for Macro/Micro inspection so SAM-2
    doesn't generate duplicate crops across the left and right earring.
    """
    h, w = mask.shape
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

    valid_blobs = []
    total_fg = np.sum(closed > 0)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > (total_fg * 0.18):
            valid_blobs.append((area, lbl))

    # If 2 distinct earrings exist, crop tightly around the primary one
    if len(valid_blobs) >= 2:
        valid_blobs.sort(reverse=True)
        main_lbl = valid_blobs[0][1]
        x = stats[main_lbl, cv2.CC_STAT_LEFT]
        y = stats[main_lbl, cv2.CC_STAT_TOP]
        bw = stats[main_lbl, cv2.CC_STAT_WIDTH]
        bh = stats[main_lbl, cv2.CC_STAT_HEIGHT]
        pad = int(max(bw, bh) * 0.08)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        return upscale_to_square_pil(image_bgr[y1:y2, x1:x2])

    return get_macro_bgr_crop(image_bgr, mask, center_pt=None)

def find_necklace_macro_zones(mask):
    h, w = mask.shape
    max_dim = max(h, w)
    k_size = max(15, int(max_dim * 0.04))
    if k_size % 2 == 0:
        k_size += 1
    closed_mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    )

    dist_1 = cv2.distanceTransform(closed_mask, cv2.DIST_L2, 5)
    _, _, _, focal_pt = cv2.minMaxLoc(dist_1)

    struct_mask = closed_mask.copy()
    cv2.circle(struct_mask, focal_pt, int(max_dim * 0.28), 0, -1)
    dist_2 = cv2.distanceTransform(struct_mask, cv2.DIST_L2, 5)
    _, max_val_2, _, side_pt = cv2.minMaxLoc(dist_2)
    return focal_pt, side_pt, max_val_2 > (max_dim * 0.015)

def get_macro_bgr_crop(image_bgr, mask, center_pt, box_ratio=0.45):
    h, w = image_bgr.shape[:2]
    if center_pt is None:
        coords = np.argwhere(mask > 0)
        if coords.size > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
        else:
            x1, y1, x2, y2 = 0, 0, w, h
    else:
        cx, cy = center_pt
        half = int(max(h, w) * box_ratio) // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        local_coords = np.argwhere(mask[y1:y2, x1:x2] > 0)
        if local_coords.size > 0:
            sy1, sx1 = local_coords.min(axis=0)
            sy2, sx2 = local_coords.max(axis=0)
            x1, y1, x2, y2 = x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2

    pad = int(max(x2 - x1, y2 - y1) * MACRO_PAD_RATIO)
    x1_f, y1_f = max(0, x1 - pad), max(0, y1 - pad)
    x2_f, y2_f = min(w, x2 + pad), min(h, y2 + pad)

    return upscale_to_square_pil(image_bgr[y1_f:y2_f, x1_f:x2_f])

# ==========================================
# 5. SAM-2 PAIR CLUSTERING & VISUAL DEDUPLICATION
# ==========================================
def compute_box_iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(a1 + a2 - inter)

def compute_visual_fingerprint(bgr_patch):
    """Computes an HSV + Edge density fingerprint to reject identical repeating chain links."""
    resized = cv2.resize(bgr_patch, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_paired_sam_crops_in_memory(sam_engine, macro_bgr, max_parts):
    H, W = macro_bgr.shape[:2]
    total_area = H * W
    raw_masks = sam_engine.generate_masks(cv2.cvtColor(macro_bgr, cv2.COLOR_BGR2RGB))

    candidates = []
    for mask_np in raw_masks:
        ys, xs = np.where(mask_np > 0)
        if ys.size == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bw, bh = x2 - x1, y2 - y1
        if ys.size < (total_area * 0.005) or (bw * bh) > (total_area * 0.62):
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 5.5:
            continue
        if macro_bgr[mask_np > 0].mean() < 25:
            continue
        candidates.append({
            "raw_box": (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "area": int(ys.size)
        })

    candidates.sort(key=lambda c: c["area"], reverse=True)

    # Merge adjacent twin motifs into Paired Units
    merged_units = []
    used = set()
    for i in range(len(candidates)):
        if i in used:
            continue
        c1 = candidates[i]
        curr_box = list(c1["raw_box"])
        is_pair = False
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            c2 = candidates[j]
            dist = np.hypot(c1["center"][0] - c2["center"][0], c1["center"][1] - c2["center"][1])
            if dist < PAIR_MERGE_DISTANCE:
                curr_box[0] = min(curr_box[0], c2["raw_box"][0])
                curr_box[1] = min(curr_box[1], c2["raw_box"][1])
                curr_box[2] = max(curr_box[2], c2["raw_box"][2])
                curr_box[3] = max(curr_box[3], c2["raw_box"][3])
                used.add(j)
                is_pair = True
        used.add(i)
        mx1, my1, mx2, my2 = curr_box
        merged_units.append({
            "raw_box": tuple(curr_box),
            "center": ((mx1 + mx2) // 2, (my1 + my2) // 2),
            "is_pair": is_pair
        })

    unique_units = []
    for unit in merged_units:
        cx, cy = unit["center"]
        bx1, by1, bx2, by2 = unit["raw_box"]
        crop_dim = max(MIN_CONTEXT_WINDOW, int(max(bx2 - bx1, by2 - by1) * 1.55))
        half = crop_dim // 2
        px1, py1 = max(0, cx - half), max(0, cy - half)
        px2, py2 = min(W, cx + half), min(H, cy + half)
        ctx_box = (px1, py1, px2, py2)

        patch = macro_bgr[py1:py2, px1:px2]
        fp = compute_visual_fingerprint(patch)

        is_dup = False
        for kept in unique_units:
            center_dist = np.hypot(cx - kept["center"][0], cy - kept["center"][1])
            iou = compute_box_iou(ctx_box, kept["ctx_box"])
            # Reject spatially overlapping boxes OR nearly identical repeating links
            hist_sim = float(np.dot(fp, kept["fp"]))
            if center_dist < 140 or iou > 0.42 or (hist_sim > 0.965 and center_dist < 350):
                is_dup = True
                break

        if not is_dup:
            unit["ctx_box"] = ctx_box
            unit["fp"] = fp
            unique_units.append(unit)
            if len(unique_units) >= max_parts:
                break

    pil_crops = []
    for unit in unique_units:
        px1, py1, px2, py2 = unit["ctx_box"]
        _, pil_crop = upscale_to_square_pil(macro_bgr[py1:py2, px1:px2])
        pil_crops.append((pil_crop, unit["is_pair"]))
    return pil_crops

# ==========================================
# 6. DEEP MULTI-LINE CATEGORY & MATERIAL PROMPT ENGINE
# ==========================================
CATEGORY_ANATOMY_BLUEPRINTS = {
    "Necklace": {
        "global": (
            "You are an expert jewelry cataloger examining a full Necklace.\n"
            "Provide a precise breakdown covering:\n"
            "1. Necklace Architecture & Silhouette: Specify the style (collar, bib, choker, Y-drop, or long chain) and how the central focal zone connects to the side chains.\n"
            "2. Metal Tone & Surface Craftsmanship: Identify the metal color (yellow, white, rose, or dual-tone gold) and finish (high-polish, matte antique, filigree, or embossed relief).\n"
            "3. Gemstone Inventory: List the exact colors and cuts of all visible embedded stones, OR explicitly write 'No visible embedded stones' if the piece is 100% solid metal."
        ),
        "macro_center": (
            "Examine this high-resolution macro view of the Necklace's Primary Central Section.\n"
            "Describe its physical construction in detail:\n"
            "- Central Medallion / Focal Motif: Describe the primary centerpiece shape and its surface relief (state whether it is geometric lattice, floral scrollwork, stone-clustered, or high-relief figurative sculpting; if a sculpted deity, human figure, or bird/animal is physically visible, state its exact identity, posture, and attire).\n"
            "- Surrounding Frame & Shoulder Transitions: Describe the concentric borders or flanking shapes framing the center.\n"
            "- Lower Suspension / Bottom Edge: Describe any hanging drops, spherical beads, bell domes, or fringes suspended along the lower rim."
        ),
        "macro_side": (
            "Examine this high-resolution macro view of the Necklace's Side Arm / Chain Section.\n"
            "- Structural Linkage: State whether the side arm is a continuous rigid sculpted band or a series of articulated repeating links, and name the exact link geometry.\n"
            "- Interior Surface Work: Describe the exact relief pattern, sculpted figures, or settings inside the band segments.\n"
            "- Perimeter & Edge Detailing: Describe the top and bottom borders (smooth rim, beaded wire edge, or suspended drops)."
        )
    },
    "Earring": {
        "global": (
            "You are an expert jewelry cataloger examining an Earring asset.\n"
            "Provide a precise breakdown covering:\n"
            "1. Earring Type & Silhouette: Identify the exact structural type (stud, drop/dangle, chandelier, hoop, or tiered bell/jhumka) and overall contour.\n"
            "2. Metal Tone & Surface Craftsmanship: Identify the metal color (yellow, white, rose, or dual-tone gold) and surface finish (high-polish, laser-cut filigree, granulation, or sculpted relief).\n"
            "3. Gemstone Inventory: List the exact colors and cuts of all visible embedded stones, OR explicitly write 'No visible embedded stones' if the piece is 100% solid metal."
        ),
        "macro_full": (
            "Examine this high-resolution macro view of the Earring's complete vertical anatomy from top to bottom:\n"
            "- Upper Stud / Ear Post Section: Describe the geometric shape, relief work, or stone arrangement of the top ear-lobe section.\n"
            "- Mid-Section / Connector Mechanism: Describe any intermediate bridge, floral link, or suspension bar connecting the top stud to the lower body.\n"
            "- Lower Body / Bell / Drop Fringe: Describe the bottom hanging structure (such as a hemispherical dome, cage, teardrop dangle, or cluster of hanging micro-beads)."
        )
    },
    "Ring": {
        "global": (
            "You are an expert jewelry cataloger examining a Ring.\n"
            "Provide a precise breakdown covering:\n"
            "1. Ring Style & Profile: Identify the ring category (solitaire, halo cluster, cocktail statement, signet, band, or traditional motif ring) and top-head geometry.\n"
            "2. Metal Tone & Shank Finish: State the metal color (yellow, white, rose, or dual-tone gold) and whether the band is smooth, textured, split-shank, or openwork.\n"
            "3. Gemstone Inventory: List the exact colors and cuts of all visible embedded stones, OR explicitly write 'No visible embedded stones' if the piece is 100% solid metal."
        ),
        "macro_full": (
            "Examine this high-resolution macro view of the Ring's complete structure:\n"
            "- Center Head / Crown Setting: Describe the primary focal centerpiece on top of the ring, including its exact shape, prong/bezel/halo architecture, or central sculpted relief.\n"
            "- Gallery & Under-Bezel Bridge: Describe the side profile support work, cutouts, or decorative arches beneath the center head.\n"
            "- Shoulders & Shank (Band): Describe how the upper shoulders taper into the circular band and any surface patterns or accent settings along the shank."
        )
    },
    "Bangle": {
        "global": (
            "You are an expert jewelry cataloger examining a Bangle / Bracelet.\n"
            "Provide a precise breakdown covering:\n"
            "1. Silhouette & Band Width: State whether it is a rigid circular bangle, hinged kada, or flexible link bracelet, and describe its profile width.\n"
            "2. Metal Tone & Surface Craftsmanship: State the metal color and surface technique (high-polish, openwork filigree, raised relief, or stone-studded channel).\n"
            "3. Gemstone Inventory: List the exact colors and cuts of all visible embedded stones, OR explicitly write 'No visible embedded stones' if the piece is 100% solid metal."
        ),
        "macro_full": (
            "Examine this high-resolution macro view of the Bangle's structural band:\n"
            "- Central / Repeating Band Pattern: Describe the repeating geometric modules, floral scrollwork, or sculpted figures running along the outer circumference.\n"
            "- Side Borders & Rims: Describe the flanking edges on both sides of the band.\n"
            "- Clasp / Terminal Heads: Describe any visible hinge, screw clasp, or sculpted terminal ends."
        )
    },
    "Pendant": {
        "global": (
            "You are an expert jewelry cataloger examining a Pendant.\n"
            "Provide a precise breakdown covering:\n"
            "1. Silhouette & Frame Geometry: Identify the outer contour shape and overall design style.\n"
            "2. Metal Tone & Surface Craftsmanship: State the metal color and metalwork finish.\n"
            "3. Gemstone Inventory: List the exact colors and cuts of all visible embedded stones, OR explicitly write 'No visible embedded stones' if the piece is 100% solid metal."
        ),
        "macro_full": (
            "Examine this high-resolution macro view of the Pendant:\n"
            "- Top Bail (Suspension Loop): Describe the top loop used for threading a chain.\n"
            "- Central Body & Relief: Describe the core centerpiece motif, openness/filigree, or figurative carving.\n"
            "- Outer Border & Bottom Drops: Describe the perimeter frame and any dangling elements at the base."
        )
    }
}

def build_versatile_prompts(category: str, global_report: str = ""):
    blueprint = CATEGORY_ANATOMY_BLUEPRINTS.get(category, CATEGORY_ANATOMY_BLUEPRINTS["Pendant"])
    p_global = blueprint["global"]

    if not global_report:
        return p_global, "", "", "", False

    lower_rep = global_report.lower()
    is_plain_metal = bool(re.search(
        r"(no\s+visible\s+(embedded\s+)?(gem)?stones|no\s+(embedded\s+)?(gem)?stones|without\s+(precious\s+)?(gem)?stones|100%\s+solid|solid\s+metal)",
        lower_rep
    ))

    if is_plain_metal:
        material_directive = (
            "STRICT MATERIAL RULE: This piece is confirmed 100% solid metal with ZERO stones. "
            "Do NOT mention gemstones, diamonds, crystals, or stone settings anywhere in your response. "
            "Focus 100% on the metal sculpting, surface textures, bevels, wirework, and relief geometry."
        )
        micro_detail_instructions = (
            "- Surface & Relief Taxonomy: State whether this component is smooth polished metal, beaded granulation, hollow wire filigree, botanical/floral embossing, or a high-relief sculpted figure.\n"
            "- Figurative Identification (Only if present): If a sculpted deity, human figure, or bird/animal is physically present in this crop, name it and describe its exact posture and attire; otherwise describe only the geometric or floral metalwork.\n"
            "- Structural Connections & Borders: Describe the surrounding rim, loops, hinges, or hanging metal drops."
        )
    else:
        material_directive = (
            "STRICT MATERIAL RULE: This piece is metal set with gemstones. "
            "For every stone visible, state its exact color, facet/cabochon cut shape, and how it is mounted (prong, bezel, pavé, channel, or halo)."
        )
        micro_detail_instructions = (
            "- Gemstone & Mounting Architecture: Describe the exact stone colors, cut shapes, and metal setting style (such as a central stone encircled by a pavé halo, bezel-set drops, or channel-set rows) if stones appear in this crop.\n"
            "- Metalwork & Relief Taxonomy: Describe the surrounding gold/metal craftsmanship (smooth polished struts, floral scrollwork, beaded rims, or high-relief figurative carvings; if a sculpted human figure, deity, or bird/animal is physically visible, state its exact posture and details).\n"
            "- Edge & Linkage Details: Describe how this component connects to adjacent elements or any hanging drops attached to its edge."
        )

    macro_center_base = blueprint.get("macro_center", blueprint.get("macro_full"))
    macro_side_base = blueprint.get("macro_side", blueprint.get("macro_full"))

    p_macro_center = (
        f"{macro_center_base}\n{material_directive}\n"
        "Be specific and factual. Do not mention the black background."
    )
    p_macro_side = (
        f"{macro_side_base}\n{material_directive}\n"
        "Be specific and factual. Do not mention the black background."
    )
    p_micro = (
        f"Examine this magnified high-context close-up of a specific structural section of the {category}.\n"
        f"{material_directive}\n"
        "Describe only what is physically visible inside this frame:\n"
        f"{micro_detail_instructions}\n"
        "Do not guess unseen parts and do not mention the black background."
    )

    return p_global, p_macro_center, p_macro_side, p_micro, is_plain_metal

# ==========================================
# 7. MAIN GPU PIPELINE EXECUTION
# ==========================================
def main():
    print("🚀 [1/2] Loading Llama-3.2 Vision Model (4-bit NF4) onto GPU...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH, device_map="auto", quantization_config=bnb_config, local_files_only=True
    )

    print("🚀 [2/2] Loading SAM-2.1 Vision Engine onto GPU...")
    sam_engine = UniversalSam2GpuGenerator(SAM_MODEL_DIR)

    for img_idx, img_path in enumerate(TARGET_IMAGES, start=1):
        path_obj = Path(img_path)
        if not path_obj.exists():
            print(f"\n❌ File not found: {img_path}")
            continue

        category = detect_category(str(path_obj))
        img_bgr = cv2.imread(str(path_obj))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Full image for Pass 1 Global Overview
        _, pil_full = get_macro_bgr_crop(img_bgr, mask, center_pt=None)

        print("\n=====================================================================")
        print(f"💎 ASSET [{img_idx}/{len(TARGET_IMAGES)}]: {path_obj.name}")
        print(f"🏷️  DETECTED CATEGORY: {category.upper()}")

        # PASS 1: GLOBAL OVERVIEW
        p_global, _, _, _, _ = build_versatile_prompts(category, "")
        global_report = ask_vlm(model, processor, pil_full, p_global, max_tokens=240)
        _, p_macro_center, p_macro_side, p_micro, is_plain = build_versatile_prompts(category, global_report)

        print(f"\n🌐 [GLOBAL OVERVIEW] -> (Material Lock: {'100% PLAIN METAL' if is_plain else 'GEMSTONE + METAL'})")
        print(f"   {global_report}")

        # DYNAMIC CATEGORY ROUTING
        if category == "Necklace":
            print(f"\n   🧭 Routing: [Necklace Dual-Zone Architecture (Center 4 + Side 4)]")
            focal_pt, side_pt, has_side = find_necklace_macro_zones(mask)
            c_bgr, c_pil = get_macro_bgr_crop(img_bgr, mask, focal_pt, 0.45)
            zones = [("🎯 FOCAL CENTER MACRO", c_bgr, c_pil, p_macro_center, 4)]
            if has_side:
                s_bgr, s_pil = get_macro_bgr_crop(img_bgr, mask, side_pt, 0.42)
                zones.append(("🔗 SIDE STRUCTURE MACRO", s_bgr, s_pil, p_macro_side, 4))
        elif category == "Earring":
            print(f"\n   🧭 Routing: [Single-Earring Pair-Deduplicated Full Architecture (Top 6 Units)]")
            e_bgr, e_pil = isolate_single_earring_if_pair(img_bgr, mask)
            zones = [("🎯 PRIMARY EARRING ANATOMY MACRO", e_bgr, e_pil, p_macro_center, 6)]
        else:
            print(f"\n   🧭 Routing: [{category} Compact Full-Anatomy Architecture (Top 6 Units)]")
            f_bgr, f_pil = get_macro_bgr_crop(img_bgr, mask, center_pt=None)
            zones = [(f"🎯 {category.upper()} ANATOMY MACRO", f_bgr, f_pil, p_macro_center, 6)]

        # PASS 2 & 3: MACRO + SAM-2 MICRO INSPECTIONS
        for zone_title, macro_bgr, pil_macro, macro_prompt, max_parts in zones:
            print(f"\n{zone_title}:")
            macro_report = ask_vlm(model, processor, pil_macro, macro_prompt, max_tokens=280)
            print(f"   {macro_report}")

            print(f"\n   --> Deploying GPU SAM-2 Pair-Clustering & Visual Deduplication (Max {max_parts} Unique Units)...")
            micro_items = extract_paired_sam_crops_in_memory(sam_engine, macro_bgr, max_parts)

            if not micro_items:
                print("       (No distinct micro-units isolated)")

            for i, (pil_micro, is_pair) in enumerate(micro_items, start=1):
                tag = "PAIRED UNIT" if is_pair else "SINGLE UNIT"
                micro_report = ask_vlm(model, processor, pil_micro, p_micro, max_tokens=260)
                print(f"\n       🔎 [Micro-Detail {i} ({tag})]:\n       {micro_report}")

        print("\n=====================================================================")

if __name__ == "__main__":
    main()