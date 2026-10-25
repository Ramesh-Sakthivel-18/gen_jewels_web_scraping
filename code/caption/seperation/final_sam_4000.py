import os
import sys
import io
import gc
import re
import cv2
import torch
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList
)

# ==============================================================================
# --- CLEAN TERMINAL & UTF-8 ENCODING ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ==============================================================================

# ==========================================
# 1. SYSTEM PATH CONFIGURATION
# ==========================================
if Path(r"D:\ramesh\jewelry ipa\training\dataset_1024_framing").exists():
    DATASET_DIR = Path(r"D:\ramesh\jewelry ipa\training\dataset_1024_framing")
    LOCAL_MODEL_PATH = r"D:\ramesh\jewelry ipa\training\model\Llama-3.2-11B-Vision-Instruct"
    SAM_MODEL_DIR = Path(r"D:\ramesh\jewelry ipa\training\model\sam2.1-hiera-large")
else:
    DATASET_DIR = Path(r"D:\gen jewels\web scraping\dataset_1024_framing")
    LOCAL_MODEL_PATH = r"D:\gen jewels\web scraping\models\Llama-3.2-11B-Vision-Instruct"
    SAM_MODEL_DIR = Path(r"D:\gen jewels\web scraping\models\sam2.1-hiera-large")

SAM_PT_PATH = SAM_MODEL_DIR / "sam2.1_hiera_large.pt"
SAM_YAML_NAME = "sam2.1_hiera_l.yaml"

# Save CSV inside D:\gen jewels\web scraping\dataset_1024_framing\csv\llama3.2_sam_detailes_csv.csv
CSV_DIR = DATASET_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = CSV_DIR / "llama3.2_sam_detailes_csv.csv"

# Save SAM outputs inside D:\gen jewels\web scraping\dataset_1024_framing\output_with_sam\
SAM_OUTPUT_ROOT = DATASET_DIR / "output_with_sam"
SAM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

NATIVE_VLM_RES = (1120, 1120)
MACRO_PAD_RATIO = 0.06
MIN_CONTEXT_WINDOW = 380
PAIR_MERGE_DISTANCE = 155

# ==========================================
# 2. VLM REPETITION PENALTY & GPU INFERENCE
# ==========================================
class GeneratedOnlyRepetitionPenalty(LogitsProcessor):
    """Applies repetition penalty strictly to generated tokens, ignoring prompt & <|image|> tokens."""
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

def ask_vlm(model, processor, pil_image, prompt, max_tokens=300):
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
            print("✅ [SAM] Loaded on GPU (Native Meta CUDA Engine)!", flush=True)
        except ImportError:
            from transformers import pipeline
            self.generator = pipeline(
                "mask-generation",
                model=str(self.model_dir),
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16
            )
            self.mode = "hf_pipeline"
            print("✅ [SAM] Loaded on GPU (HuggingFace Pipeline)!", flush=True)

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

def get_clean_product_folder_name(img_path: Path) -> str:
    stem = img_path.stem
    cleaned = re.sub(r"_(1024x1024|1100x1100|1200x1200)", "", stem)
    cleaned = re.sub(r"_(clean_black_bg|black_bg)", "", cleaned)
    return cleaned

def upscale_to_square_pil(cropped_bgr):
    sh, sw = cropped_bgr.shape[:2]
    side = max(sh, sw)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    yo, xo = (side - sh) // 2, (side - sw) // 2
    canvas[yo:yo + sh, xo:xo + sw] = cropped_bgr
    upscaled = cv2.resize(canvas, NATIVE_VLM_RES, interpolation=cv2.INTER_LANCZOS4)
    return upscaled, Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))

def isolate_single_earring_if_pair(image_bgr, mask):
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
# 5. SAM-2 SEGMENTATION, DISK SAVING & RAM EXTRACTION
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
    resized = cv2.resize(bgr_patch, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_and_save_sam_crops(sam_engine, macro_bgr, zone_dir: Path, zone_folder_name: str, max_parts: int):
    zone_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(zone_dir / f"00_{zone_folder_name}_MACRO_1120x1120.png"), macro_bgr)

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
            "mask": mask_np,
            "area": int(ys.size)
        })

    candidates.sort(key=lambda c: c["area"], reverse=True)

    merged_units = []
    used = set()
    for i in range(len(candidates)):
        if i in used:
            continue
        c1 = candidates[i]
        curr_box = list(c1["raw_box"])
        combined_mask = c1["mask"].copy()
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
                combined_mask = np.maximum(combined_mask, c2["mask"])
                used.add(j)
                is_pair = True
        used.add(i)
        mx1, my1, mx2, my2 = curr_box
        merged_units.append({
            "raw_box": tuple(curr_box),
            "center": ((mx1 + mx2) // 2, (my1 + my2) // 2),
            "mask": combined_mask,
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

    annotated_map = macro_bgr.copy()
    pil_crops = []

    for idx, unit in enumerate(unique_units, start=1):
        x1, y1, x2, y2 = unit["raw_box"]
        px1, py1, px2, py2 = unit["ctx_box"]
        mask_np = unit["mask"]
        label = f"#{idx} PAIR" if unit["is_pair"] else f"#{idx}"

        cv2.rectangle(annotated_map, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(annotated_map, (px1, py1), (px2, py2), (255, 255, 0), 1)
        cv2.putText(
            annotated_map, label, (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        context_crop = macro_bgr[py1:py2, px1:px2]
        isolated_full = cv2.bitwise_and(macro_bgr, macro_bgr, mask=mask_np * 255)
        cutout_crop = isolated_full[py1:py2, px1:px2]

        upscaled_context_bgr, pil_context = upscale_to_square_pil(context_crop)
        upscaled_cutout_bgr, _ = upscale_to_square_pil(cutout_crop)

        cv2.imwrite(str(zone_dir / f"part_{idx:02d}_context_1120x1120.png"), upscaled_context_bgr)
        cv2.imwrite(str(zone_dir / f"part_{idx:02d}_cutout_1120x1120.png"), upscaled_cutout_bgr)

        pil_crops.append((pil_context, unit["is_pair"]))

    cv2.imwrite(str(zone_dir / f"01_{zone_folder_name}_SAM_DETECTED_MAP.png"), annotated_map)
    return pil_crops

# ==========================================
# 6. DEEP MULTI-LINE CATEGORY & MATERIAL PROMPT ENGINE
# ==========================================
CATEGORY_ANATOMY_BLUEPRINTS = {
    "Necklace": {
        "global": (
            "You are a Senior Gemologist and Indian Fine Jewelry Appraiser examining a full Necklace.\n"
            "Provide an authoritative breakdown covering:\n"
            "1. Necklace Architecture & Silhouette: Specify the style (collar, bib, choker, haram, or Y-drop) and how the central focal zone connects to the side chains.\n"
            "2. Metal Tone & Surface Craftsmanship: Identify the metal color (yellow, white, or rose gold) and finish (high-polish, matte antique, filigree, or embossed relief).\n"
            "3. Gemstone & Enamel Inventory: Inspect the entire piece carefully for any faceted diamonds/white stones, red rubies/kemp cabochons, green emeralds, or colored enamel. List their exact colors and cuts, or write 'SOLID_GOLD_ONLY' if the entire piece has zero stones."
        ),
        "macro_center": (
            "Examine this high-resolution macro view of the Necklace's Primary Central Section.\n"
            "Describe its physical construction with authoritative precision:\n"
            "- Central Medallion / Focal Motif: Describe the centerpiece geometry and surface relief (state whether it is geometric lattice, floral lotus scrollwork, stone-clustered, or high-relief figurative sculpting; if an Indian heritage motif such as seated Goddess Lakshmi, classical dancers, peacocks, or yali/elephants is physically sculpted, state its exact identity, posture, and attire).\n"
            "- Surrounding Frame & Shoulder Transitions: Describe the concentric borders or flanking shapes framing the center.\n"
            "- Lower Suspension / Bottom Edge: Describe any hanging drops, spherical gold balls, ribbed drum beads, or bell domes suspended along the lower rim."
        ),
        "macro_side": (
            "Examine this high-resolution macro view of the Necklace's Side Arm / Chain Section.\n"
            "- Structural Linkage: State whether the side arm is a continuous rigid sculpted band or a series of articulated repeating links, and name the exact link geometry.\n"
            "- Interior Surface Work: Describe the exact relief pattern, sculpted heritage figures (such as classical dancers, seated Lakshmi medallions, or peacocks), or stone settings inside the band segments.\n"
            "- Perimeter & Edge Detailing: Describe the top and bottom borders (smooth rim, beaded wire edge, or suspended drops)."
        )
    },
    "Earring": {
        "global": (
            "You are a Senior Gemologist and Indian Fine Jewelry Appraiser examining an Earring asset.\n"
            "Provide an authoritative breakdown covering:\n"
            "1. Earring Type & Silhouette: Identify the exact structural type (stud, drop/dangle, chandelier, chandbali crescent, or tiered bell/jhumka) and overall contour.\n"
            "2. Metal Tone & Surface Craftsmanship: Identify the metal color (yellow, white, or rose gold) and surface finish (high-polish, openwork filigree, granulation beading, or sculpted temple relief).\n"
            "3. Gemstone & Enamel Inventory: Inspect carefully for any white diamonds, red ruby/kemp cabochons, pink/green stones, or colored enamel. List their exact colors and cuts, or write 'SOLID_GOLD_ONLY' if the entire earring has zero stones."
        ),
        "macro_full": (
            "Examine this high-resolution macro view of the Earring's complete vertical anatomy from top to bottom:\n"
            "- Upper Stud / Ear Post Section: Describe the geometric shape, sculpted motif (such as a floral rosette, heart, or peacock stud), or stone setting of the top ear-lobe section.\n"
            "- Mid-Section / Connector Mechanism: Describe the intermediate bridge, twin-bird/peacock crescent, chain links, or suspension bar connecting the top stud to the lower body.\n"
            "- Lower Body / Bell / Drop Fringe: Describe the bottom hanging structure (such as a hemispherical jhumka dome, chandbali crescent fringe, bezel-set stone drop, or cluster of hanging gold balls)."
        )
    },
    "Ring": {
        "global": (
            "You are a Senior Gemologist and Fine Jewelry Appraiser examining a Ring.\n"
            "Provide an authoritative breakdown covering:\n"
            "1. Ring Style & Profile: Identify the ring category (solitaire, marquise/oval cluster, halo, cocktail statement, or band) and top-head geometry.\n"
            "2. Metal Tone & Shank Finish: State the metal color (yellow, white, or rose gold) and whether the circular band is smooth, textured, split-shank, or openwork filigree scrollwork.\n"
            "3. Gemstone Inventory: Inspect the top head and shoulders carefully for any pavé diamonds, solitaire stones, or colored gemstones. List their exact colors, cuts, and placement, or write 'SOLID_GOLD_ONLY' if the entire ring has zero stones."
        ),
        "macro_full": (
            "Examine this high-resolution macro view of the Ring's complete structure:\n"
            "- Center Head / Crown Setting: Describe the primary focal centerpiece on top of the ring, including its exact contour (marquise, round, oval, square), stone cluster/prong/bezel arrangement, or beaded rim.\n"
            "- Gallery & Under-Bezel Bridge: Describe the side profile support work, cutouts, or decorative arches beneath the center head.\n"
            "- Shoulders & Shank (Band): Describe how the upper shoulders taper into the circular band and detail the openwork scrollwork, filigree curls, or channel settings along the shank."
        )
    },
    "Bangle": {
        "global": (
            "You are a Senior Gemologist and Fine Jewelry Appraiser examining a Bangle / Bracelet.\n"
            "Provide an authoritative breakdown covering:\n"
            "1. Silhouette & Band Width: State whether it is a rigid circular bangle, hinged kada, or articulated link bracelet.\n"
            "2. Metal Tone & Surface Craftsmanship: State the metal color and surface technique (high-polish, openwork filigree, raised temple relief, or stone-studded channel).\n"
            "3. Gemstone Inventory: Inspect carefully for any embedded diamonds, rubies, emeralds, or colored stones. List their exact colors and cuts, or write 'SOLID_GOLD_ONLY' if the entire bangle has zero stones."
        ),
        "macro_full": (
            "Examine this high-resolution macro view of the Bangle's structural band:\n"
            "- Central / Repeating Band Pattern: Describe the repeating geometric modules, floral scrollwork, or sculpted motifs running along the outer circumference.\n"
            "- Side Borders & Rims: Describe the flanking edges on both sides of the band.\n"
            "- Clasp / Terminal Heads: Describe any visible hinge, screw clasp, or sculpted terminal ends."
        )
    },
    "Pendant": {
        "global": (
            "You are a Senior Gemologist and Fine Jewelry Appraiser examining a Pendant.\n"
            "Provide an authoritative breakdown covering:\n"
            "1. Silhouette & Frame Geometry: Identify the outer contour shape and overall design style.\n"
            "2. Metal Tone & Surface Craftsmanship: State the metal color and metalwork finish.\n"
            "3. Gemstone Inventory: List the exact colors and cuts of all visible embedded stones, or write 'SOLID_GOLD_ONLY' if the entire pendant has zero stones."
        ),
        "macro_full": (
            "Examine this high-resolution macro view of the Pendant:\n"
            "- Top Bail (Suspension Loop): Describe the top loop used for threading a chain.\n"
            "- Central Body & Relief: Describe the core centerpiece motif, openness/filigree, or figurative carving.\n"
            "- Outer Border & Bottom Drops: Describe the perimeter frame and any dangling elements at the base."
        )
    }
}

def determine_if_plain_metal(global_report: str) -> bool:
    low = global_report.lower()
    if "solid_gold_only" in low:
        return True
    positive_stone_patterns = [
        r"\b(diamond|diamonds|emerald|emeralds|ruby|rubies|sapphire|sapphires|rhodochrosite|garnet|garnets|topaz|amethyst|tanzanite|kemp|polki|kundan)\b",
        r"\b(pink\s+stone|red\s+stone|green\s+stone|white\s+stone|clear\s+stone|purple\s+stone|blue\s+stone)\b",
        r"\b(pav[eé]|marquise-cut\s+diamond|prong-set|bezel-set\s+stone)\b"
    ]
    if any(re.search(pat, low) for pat in positive_stone_patterns):
        return False
    no_stone_patterns = [
        r"no\s+visible\s+(embedded\s+)?(gem)?stones",
        r"no\s+(embedded\s+)?(gem)?stones",
        r"zero\s+(gem)?stones",
        r"100%\s+solid\s+(yellow\s+|rose\s+|white\s+)?(gold|metal)"
    ]
    return any(re.search(pat, low) for pat in no_stone_patterns)

def build_versatile_prompts(category: str, global_report: str = ""):
    blueprint = CATEGORY_ANATOMY_BLUEPRINTS.get(category, CATEGORY_ANATOMY_BLUEPRINTS["Pendant"])
    p_global = blueprint["global"]

    if not global_report:
        return p_global, "", "", "", False

    is_plain_metal = determine_if_plain_metal(global_report)

    anti_hallucination_rule = (
        "STRICT VISUAL ACCURACY RULES:\n"
        "- Never use uncertain guessing words ('possibly', 'maybe', 'likely'). State only what is physically visible.\n"
        "- Do NOT hallucinate human faces, queens, eagles, or lions inside abstract filigree scrollwork or diamond clusters.\n"
        "- If a traditional Indian heritage carving is physically sculpted, identify it accurately (such as seated Goddess Lakshmi on a lotus, classical dancers in pleated skirts, peacocks with curved necks and plumage, swans/hamsa, or yali)."
    )

    if is_plain_metal:
        material_directive = (
            "STRICT MATERIAL RULE: This piece is confirmed 100% solid metal with ZERO gemstones. "
            "Do NOT mention diamonds, crystals, or stone settings anywhere. "
            "Focus 100% on the gold sculpting, surface textures, bevels, wirework, granulation beads, and relief geometry."
        )
        micro_detail_instructions = (
            "- Surface & Relief Taxonomy: State whether this section is smooth polished gold, beaded granulation, hollow wire filigree scrollwork, botanical/floral embossing, or a high-relief sculpted heritage figure.\n"
            "- Motif Identification: Describe the exact geometric curls, lotus petals, ribbed beads, or traditional sculpted figures (such as Goddess Lakshmi, classical dancers, or peacocks) physically present in this frame.\n"
            "- Structural Connections & Borders: Describe the surrounding rim, loops, hinges, or hanging gold drops."
        )
    else:
        material_directive = (
            "STRICT MATERIAL RULE: This piece is metal set with gemstones. "
            "For every stone visible in the frame, state its exact color (white/clear diamond, red ruby/kemp, green emerald, pink stone), cut shape (round, marquise, pear/teardrop, oval, cabochon), and mounting style (prong, bezel, pavé cluster, or halo)."
        )
        micro_detail_instructions = (
            "- Gemstone & Mounting Architecture: Describe the exact stone colors, cut shapes, and metal setting style (such as a pavé cluster, central stone with a halo, or bezel-set red/green cabochons) visible in this frame.\n"
            "- Metalwork & Relief Taxonomy: Describe the surrounding metal craftsmanship (openwork filigree curls, smooth polished struts, floral scrollwork, beaded rims, or high-relief Indian heritage carvings such as classical dancers, seated Goddess Lakshmi, or peacocks).\n"
            "- Edge & Linkage Details: Describe how this component connects to adjacent elements or any hanging drops attached to its edge."
        )

    macro_center_base = blueprint.get("macro_center", blueprint.get("macro_full"))
    macro_side_base = blueprint.get("macro_side", blueprint.get("macro_full"))

    p_macro_center = (
        f"{macro_center_base}\n\n{material_directive}\n{anti_hallucination_rule}\n"
        "Do not mention the black background."
    )
    p_macro_side = (
        f"{macro_side_base}\n\n{material_directive}\n{anti_hallucination_rule}\n"
        "Do not mention the black background."
    )
    p_micro = (
        f"Examine this magnified high-context close-up of a specific structural section of the {category}.\n"
        f"{material_directive}\n{anti_hallucination_rule}\n"
        "Describe only what is physically visible inside this frame:\n"
        f"{micro_detail_instructions}\n"
        "Do not mention the black background."
    )

    return p_global, p_macro_center, p_macro_side, p_micro, is_plain_metal

# ==========================================
# 7. MAIN BULK PIPELINE EXECUTION
# ==========================================
def main():
    print("=" * 85, flush=True)
    print("🚀 [1/4] Loading Llama-3.2-11B-Vision-Instruct (4-bit NF4) onto GPU...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        local_files_only=True
    ).eval()
    print("✅ [VLM] Llama-3.2-11B-Vision Loaded Successfully!", flush=True)

    print("\n🚀 [2/4] Loading SAM-2.1 Vision Engine onto GPU...", flush=True)
    sam_engine = UniversalSam2GpuGenerator(SAM_MODEL_DIR)

    print(f"\n📁 [3/4] Scanning Dataset Directory: {DATASET_DIR}", flush=True)
    all_image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in DATASET_DIR.rglob(ext):
            if "output_with_sam" in p.parts or "csv" in p.parts:
                continue
            all_image_paths.append(p)

    all_image_paths.sort()
    total_images = len(all_image_paths)
    print(f"✅ Found {total_images} dataset images.", flush=True)
    print(f"📂 SAM-2 Segmented Images Root: {SAM_OUTPUT_ROOT}", flush=True)
    print(f"📄 Output CSV Path:             {OUTPUT_CSV}", flush=True)
    print("=" * 85, flush=True)

    # Auto-Resume support
    results = []
    processed_paths = set()
    if OUTPUT_CSV.exists():
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            results = existing_df.to_dict("records")
            processed_paths = set(existing_df["Image_Path"].astype(str).tolist())
            print(f"🔄 Resuming: {len(processed_paths)} images already in CSV. Skipping processed items.\n", flush=True)
        except Exception as e:
            print(f"⚠️ Could not read existing CSV ({e}). Starting fresh.\n", flush=True)

    for idx, img_path in enumerate(all_image_paths, start=1):
        img_str_path = str(img_path)
        sub_folder = img_path.parent.name
        main_folder = img_path.parent.parent.name
        short_id = f"{main_folder}/{sub_folder}/{img_path.name}"

        if img_str_path in processed_paths:
            print(f"[{idx}/{total_images}][SKIP] {short_id}", flush=True)
            continue

        prod_folder_name = get_clean_product_folder_name(img_path)
        image_sam_dir = SAM_OUTPUT_ROOT / main_folder / sub_folder / prod_folder_name
        image_sam_dir.mkdir(parents=True, exist_ok=True)

        category = detect_category(f"{sub_folder}_{img_path.name}")

        try:
            img_bgr = cv2.imread(img_str_path)
            if img_bgr is None:
                print(f"[{idx}/{total_images}][ERROR] {short_id} (Unreadable image)", flush=True)
                continue

            cv2.imwrite(str(image_sam_dir / "00_ORIGINAL_FULL.png"), img_bgr)

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

            _, pil_full = get_macro_bgr_crop(img_bgr, mask, center_pt=None)
            raw_output_sections = []

            # PASS 1: VLM GLOBAL OVERVIEW
            print(f"[{idx}/{total_images}][VLM] {short_id} | Global", flush=True)
            p_global, _, _, _, _ = build_versatile_prompts(category, "")
            global_report = ask_vlm(model, processor, pil_full, p_global, max_tokens=240)
            _, p_macro_center, p_macro_side, p_micro, is_plain = build_versatile_prompts(category, global_report)

            mat_lock_str = "100% PLAIN METAL" if is_plain else "GEMSTONE + METAL"
            raw_output_sections.append(
                f"=== GLOBAL OVERVIEW (Category: {sub_folder} | Material Lock: {mat_lock_str}) ===\n{global_report}"
            )

            # DYNAMIC SHAPE ROUTING
            if category == "Necklace":
                focal_pt, side_pt, has_side = find_necklace_macro_zones(mask)
                c_bgr, c_pil = get_macro_bgr_crop(img_bgr, mask, focal_pt, 0.45)
                zones = [("01_FOCAL_CENTER", "FOCAL_CENTER", "🎯 FOCAL CENTER MACRO", c_bgr, c_pil, p_macro_center, 4)]
                if has_side:
                    s_bgr, s_pil = get_macro_bgr_crop(img_bgr, mask, side_pt, 0.42)
                    zones.append(("02_SIDE_STRUCTURE", "SIDE_STRUCTURE", "🔗 SIDE STRUCTURE MACRO", s_bgr, s_pil, p_macro_side, 4))
            elif category == "Earring":
                e_bgr, e_pil = isolate_single_earring_if_pair(img_bgr, mask)
                zones = [("01_PRIMARY_EARRING", "PRIMARY_EARRING", "🎯 PRIMARY EARRING ANATOMY MACRO", e_bgr, e_pil, p_macro_center, 6)]
            else:
                f_bgr, f_pil = get_macro_bgr_crop(img_bgr, mask, center_pt=None)
                zones = [("01_FULL_ITEM", "FULL_ITEM", f"🎯 {category.upper()} ANATOMY MACRO", f_bgr, f_pil, p_macro_center, 6)]

            # PASS 2 & 3: MACRO VLM + SAM-2 SEGMENTATION + MICRO VLM
            for subfolder_tag, zone_code, zone_title, macro_bgr, pil_macro, macro_prompt, max_parts in zones:
                print(f"[{idx}/{total_images}][VLM] {short_id} | Macro ({zone_code})", flush=True)
                macro_report = ask_vlm(model, processor, pil_macro, macro_prompt, max_tokens=280)
                raw_output_sections.append(f"\n=== {zone_title} ===\n{macro_report}")

                zone_save_dir = image_sam_dir / subfolder_tag
                print(f"[{idx}/{total_images}][SAM] {short_id} | Segmenting ({zone_code})", flush=True)
                micro_items = extract_and_save_sam_crops(
                    sam_engine, macro_bgr, zone_save_dir, zone_code, max_parts
                )

                for i, (pil_micro, is_pair) in enumerate(micro_items, start=1):
                    unit_tag = "PAIRED UNIT" if is_pair else "SINGLE UNIT"
                    print(f"[{idx}/{total_images}][VLM] {short_id} | Micro {i}/{len(micro_items)} ({zone_code})", flush=True)
                    micro_report = ask_vlm(model, processor, pil_micro, p_micro, max_tokens=260)

                    raw_output_sections.append(
                        f"\n--- [{zone_code} | Micro-Detail {i} ({unit_tag})] ---\n{micro_report}"
                    )

            full_raw_output = "\n".join(raw_output_sections).strip()

            results.append({
                "Image_Path": img_str_path,
                "Brand_Name": main_folder,
                "Category": sub_folder,
                "Full_Raw_Output": full_raw_output
            })

            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, quoting=1)
            print(f"[{idx}/{total_images}][CSV] {short_id} | Saved ({len(results)} total)\n", flush=True)

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"[{idx}/{total_images}][ERROR] {short_id} | {e}\n", flush=True)

    print("=" * 85, flush=True)
    print(f"🎉 ALL DONE! Processed {len(results)} images.", flush=True)
    print(f"📂 SAM-2 Segmented Dataset Folders: {SAM_OUTPUT_ROOT}", flush=True)
    print(f"📄 Final CSV Saved to:              {OUTPUT_CSV}", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    main()