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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# ==============================================================================
# --- CLEAN TERMINAL & UTF-8 ENCODING ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ==============================================================================

# ==========================================
# 1. SYSTEM PATHS & FULL DATASET CONFIGURATION
# ==========================================
MODEL_DIR = r"D:\ramesh\jewelry ipa\training\model\InternVL2-8B"
SAM_MODEL_DIR = Path(r"D:\ramesh\jewelry ipa\training\model\SAM")
SAM_PT_PATH = SAM_MODEL_DIR / "sam2.1_hiera_large.pt"

DATASET_DIR = Path(r"D:\ramesh\jewelry ipa\training\dataset_1024_framing")
CSV_DIR = DATASET_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = CSV_DIR / "intern_vl2_sam_newprompt.csv"

SAM_OUTPUT_ROOT = DATASET_DIR / "output_with_sam"
SAM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Full Dataset mode enabled with auto-resume support
RUN_FULL_DATASET = True
USE_CACHED_SAM_CROPS = True
PRINT_VLM_TEXT_IN_TERMINAL = False  # Single-line progress logging for full dataset run

NATIVE_CROP_RES = (1120, 1120)
MACRO_PAD_RATIO = 0.06
MIN_CONTEXT_WINDOW = 380
PAIR_MERGE_DISTANCE = 155

# ==========================================
# 2. INTERNVL2 DYNAMIC HIGH-RES PREPROCESSING
# ==========================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=4, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def pil_to_pixel_values(pil_image: Image.Image, input_size=448, max_num=4):
    image = pil_image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(patch) for patch in images]
    pixel_values = torch.stack(pixel_values).to(torch.float16).cuda()
    return pixel_values

def clean_vlm_response(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    banned_line_patterns = [
        r"^(note|strict\s+material\s+rule|strict\s+visual\s+accuracy|important\s+rule)\b",
        r"\b(as\s+per\s+strict\s+rules|per\s+strict\s+rules|according\s+to\s+strict\s+rules|adheres\s+strictly|without\s+making\s+assumptions)\b",
        r"^this\s+(detailed\s+|high-resolution\s+)?(breakdown|description|examination|analysis)\s+(covers|ensures|adheres|provides|captures|highlights|focuses)\b"
    ]
    for line in lines:
        s = line.strip().lower()
        if any(re.search(pat, s) for pat in banned_line_patterns):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def ask_internvl(model, tokenizer, pil_image: Image.Image, prompt: str, max_tokens=480, max_num=4):
    pixel_values = pil_to_pixel_values(pil_image, input_size=448, max_num=max_num)
    safe_prompt = "<image>\n" + prompt
    generation_config = dict(
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.12
    )
    with torch.no_grad():
        response = model.chat(
            tokenizer,
            pixel_values,
            safe_prompt,
            generation_config,
            history=None,
            return_history=False
        )
    return clean_vlm_response(response)

# ==========================================
# 3. SAM-2 GPU ENGINE LOADER (ON-DEMAND)
# ==========================================
class UniversalSam2GpuGenerator:
    def __init__(self, pt_path: Path):
        self.pt_path = pt_path
        self.model_dir = pt_path.parent
        self.generator = None

    def ensure_loaded(self):
        if self.generator is not None:
            return
        print("🚀 Loading SAM-2.1 Vision Engine onto GPU...", flush=True)
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = None

        for cfg_name in ["configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_l.yaml"]:
            try:
                sam2_model = build_sam2(
                    config_file=cfg_name,
                    ckpt_path=str(self.pt_path.resolve()),
                    device=device,
                    mode="eval"
                )
                break
            except Exception:
                continue

        if sam2_model is None:
            import hydra
            from hydra.core.global_hydra import GlobalHydra
            GlobalHydra.instance().clear()
            hydra.initialize_config_dir(config_dir=str(self.model_dir.resolve()), version_base=None)
            sam2_model = build_sam2(
                config_file="sam2.1_hiera_l.yaml",
                ckpt_path=str(self.pt_path.resolve()),
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
        print("✅ [SAM] Loaded on GPU!", flush=True)

    def generate_masks(self, rgb_np: np.ndarray):
        self.ensure_loaded()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            anns = self.generator.generate(rgb_np)
        return [ann["segmentation"].astype(np.uint8) for ann in anns]

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
    upscaled = cv2.resize(canvas, NATIVE_CROP_RES, interpolation=cv2.INTER_LANCZOS4)
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
# 5. SAM-2 CROP LOADER / GENERATOR
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

def get_or_create_sam_crops(sam_engine, macro_bgr, zone_dir: Path, zone_folder_name: str, max_parts: int):
    if USE_CACHED_SAM_CROPS and zone_dir.exists():
        existing_part_files = sorted(zone_dir.glob("part_*_context_1120x1120.png"))
        if len(existing_part_files) > 0:
            pil_crops = []
            for p_file in existing_part_files[:max_parts]:
                bgr = cv2.imread(str(p_file))
                if bgr is not None:
                    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    pil_crops.append((pil_img, False))
            if len(pil_crops) > 0:
                return pil_crops, True

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
    return pil_crops, False

# ==========================================
# 6. ZERO-BIAS UNIVERSAL OBSERVATIONAL PROMPTS
# ==========================================
GLOBAL_PROMPTS = {
    "Bangle": (
        "You are a Senior Gemologist examining a Bangle / Bracelet.\n"
        "Observe the image carefully and provide a factual 3-part breakdown:\n"
        "1. Silhouette & Structural Architecture: State whether this is (a) a wide/broad cuff kada built from multiple parallel transverse cylindrical bars/tubes bridged across two circular rims, OR (b) a single-band circular bangle with repeating floral clusters or geometric links. Note any visible hinge or screw clasp.\n"
        "2. Metal Tone & Surface Craftsmanship: State the gold color (yellow, antique, white, or rose gold) and describe the physical metal texture visible (such as finely ridged/fluted cylindrical gold rods, engraved side rims, smooth gold bezels/prongs, or openwork struts).\n"
        "3. Gemstones, Coral Cylinders & Beads: If the piece is 100% bare metal, write ONLY 'NONE - 100% SOLID METAL'. Otherwise, describe ONLY the stones, tubes, or beads physically present—stating their exact color (red, pink-red, green, pale mint-green, or white) and exact 3D shape (cylindrical red coral/stone tubes, carved/fluted green melon beads, pear/marquise-cut floral petal clusters, round stones, or tiny white diamond spacer rings). Do NOT mention stones that are absent."
    ),
    "Necklace": (
        "You are a Senior Gemologist examining a Necklace.\n"
        "Observe the image carefully and provide a factual 3-part breakdown:\n"
        "1. Style & Design Aesthetic: State clearly whether this is (a) a Modern Minimalist Fine-Chain / Y-Lariat Pendant Necklace, OR (b) a Traditional Ornate / Sculpted Heritage / Bib / Haram Necklace. Describe the chain structure and central drop/pendant silhouette.\n"
        "2. Metal Tone & Surface Craftsmanship: Identify the metal color and describe the actual metalwork (such as a thin delicate cable/box chain, a small stone-set O-ring connector, smooth bezel frames, openwork filigree, or 3D sculpted relief).\n"
        "3. Gemstones, Enamel & Beads: If the necklace is 100% bare gold, write ONLY 'NONE - 100% SOLID METAL'. Otherwise, list ONLY the non-metal elements physically visible—such as small round white pavé diamonds/stones, flat pastel colored enamel inlay sections (specifying exact pastel colors like mint-green, pink, or yellow), red/green cabochons, or hanging pearl beads. Do NOT list absent stones."
    ),
    "Earring": (
        "You are a Senior Gemologist examining an Earring asset.\n"
        "Observe the image carefully and provide a factual 3-part breakdown:\n"
        "1. Earring Type & Silhouette: State clearly whether this is (a) a Compact Stud (sitting flush on the ear with no hanging drop), (b) a Minimalist Linear Open-Link / Ball-Drop Dangle Earring, OR (c) a Traditional Ornate Jhumka Bell / Chandbali Drop Earring.\n"
        "2. Metal Tone & Structural Anatomy: Identify the gold color and describe the exact physical components from top to bottom (such as a curved top ear-wire hook or stud post, elongated oval paperclip chain links, a small beaded connector ring, a smooth polished spherical gold ball drop, or a flared bell dome).\n"
        "3. Gemstones & Enamel Inventory: Inspect carefully: if the earring is 100% bare gold with zero gemstones, write ONLY 'NONE - 100% SOLID METAL' under this section (do NOT list absent stones like emeralds or diamonds). If stones are present, state their exact color, shape, and setting."
    ),
    "Ring": (
        "You are a Senior Gemologist examining a Ring.\n"
        "Observe the image carefully and provide a factual 3-part breakdown:\n"
        "1. Ring Style & Top-Head Silhouette: Identify the contour of the top head (such as a curved leaf/marquise silhouette, round/oval halo cluster, solitaire, or band) and how the centerpiece is structured.\n"
        "2. Metal Tone, Head Lattice & Band (Shank) Finish: State the gold color. Distinguish clearly between the top head's metalwork (such as an openwork mesh/grid lattice frame) and the circular finger band/shank (stating clearly whether the band itself is smooth high-polish solid gold or textured).\n"
        "3. Gemstone Inventory: If 100% bare gold, write ONLY 'NONE - 100% SOLID METAL'. Otherwise, describe the exact color, cut shape, count/arrangement (such as a single diagonal row of round white diamonds/stones down the center of the leaf head), and prong/channel mounting."
    ),
    "Pendant": (
        "You are a Senior Gemologist examining a Pendant.\n"
        "Observe the image carefully and provide a factual 3-part breakdown:\n"
        "1. Silhouette & Frame Geometry: Identify the outer contour shape and design style.\n"
        "2. Metal Tone & Surface Craftsmanship: State the gold color and physical metalwork finish.\n"
        "3. Gemstones & Enamel Inventory: If 100% bare gold, write ONLY 'NONE - 100% SOLID METAL'. Otherwise, list the exact colors, cuts, and settings of all visible stones or enamel."
    )
}

def determine_if_plain_metal(text: str) -> bool:
    low = text.lower()
    if "none - 100% solid metal" in low or "100% bare metal" in low or "100% solid metal" in low:
        return True

    cleaned = re.sub(
        r"(no\s+visible|absence\s+of|without\s+any|do\s+not\s+appear|aren't\s+any|lack\s+any|no\s+emeralds|no\s+diamonds|no\s+rubies|none\s+were\s+observed)[^.\n]*",
        " ",
        low
    )

    positive_patterns = [
        r"\b(diamond|diamonds|emerald|emeralds|ruby|rubies|sapphire|sapphires|coral|enamel|peridot|kemp|polki|kundan|pearl|pearls)\b",
        r"\b(red\s+cylindrical|red\s+tube|green\s+bead|melon\s+bead|pink-red|white\s+stone|white\s+stones|green\s+stone|pastel\s+green|mint-green)\b"
    ]
    if any(re.search(pat, cleaned) for pat in positive_patterns):
        return False

    return True

def detect_sub_style(category: str, global_report: str, crop_bgr: np.ndarray) -> str:
    low = global_report.lower()
    if category == "Bangle":
        if re.search(r"\b(floral\s+cluster|petal|petals|marquise-cut|pear-cut)\b", low):
            return "Bangle_Single_Band"
        if re.search(r"\b(cylindrical\s+bars|parallel\s+cylindrical|coral\s+tubes|cuff\s+kada)\b", low):
            return "Bangle_Broad_Cylindrical_Kada"
        return "Bangle_Single_Band"

    if category == "Necklace":
        if re.search(r"\b(minimalist|lariat|y-drop|y-lariat|enamel|o-ring|delicate\s+cable|thin\s+chain|modern)\b", low):
            return "Necklace_Modern_Minimalist"
        return "Necklace_Traditional_Ornate"

    if category == "Earring":
        cleaned = re.sub(r"(without\s+any|with\s+no|no\s+hanging|no\s+dangling|not\s+a|rather\s+than)[^.\n]*", " ", low)
        if re.search(r"compact\s+stud|stud\s+earring", cleaned) and not re.search(r"\b(drop|dangle|jhumka|chandbali|hook|paperclip)\b", cleaned):
            return "Earring_Compact_Stud"
        if re.search(r"\b(jhumka|chandbali|bell\s+dome)\b", cleaned):
            return "Earring_Traditional_Jhumka"
        return "Earring_Linear_Drop"

    return f"{category}_Standard"

def build_macro_prompts(category: str, sub_style: str, is_plain_metal: bool, global_report: str):
    stone_rule = (
        "STRICT MATERIAL RULE: This piece is confirmed 100% bare solid gold with ZERO gemstones or enamel. Describe ONLY the solid gold components."
        if is_plain_metal else
        "STRICT MATERIAL RULE: Describe ONLY the exact stone/bead/enamel colors and shapes physically visible in this image. Do NOT invent rectangular step-cut stones, jhumka domes, or temple figures if they are not physically present."
    )

    if category == "Bangle":
        if sub_style == "Bangle_Broad_Cylindrical_Kada":
            p_macro = (
                "Examine this high-resolution macro view of the Broad Multi-Row Cylindrical Cuff Kada.\n"
                f"{stone_rule}\n"
                "Provide a factual 3-point breakdown of its physical construction:\n"
                "- Parallel Cylindrical Bars & Red Tubes: Describe the alternating transverse cylinders running across the width of the cuff (specifically the finely ridged/fluted yellow-gold cylindrical bars alternating with smooth glossy red cylindrical coral/stone tubes).\n"
                "- Terminal Beads & Diamond Collar Rings: Describe the carved/fluted pale-green melon beads capping both ends of the red tubes and the small white-stone-studded spacer rings/collars between the red tubes and green end-beads.\n"
                "- Side Rims, Inner Cage & Clasp: Describe the engraved circular gold side rims, the small gold beads between the bars, the interior flat gold support struts, and any visible side screw/hinge clasp."
            )
        else:
            p_macro = (
                "Examine this high-resolution macro view of the Circular Bangle.\n"
                f"{stone_rule}\n"
                "Provide a factual 3-point breakdown of its physical construction:\n"
                "- Repeating Gemstone / Metal Modules: Describe the exact repeating pattern along the outer band (such as 4-petal floral/butterfly clusters of pear/marquise-cut pink-red stones, or geometric modules) and any small round green accent stones along the top and bottom edges.\n"
                "- Mounting & Frame Architecture: Describe how each stone is held (smooth gold bezels, shared prongs, or raised settings) and the openwork gold frame connecting the modules.\n"
                "- Inner Band & Side Edges: Describe the side rims and interior gold band."
            )
        return p_macro, p_macro

    if category == "Necklace":
        if sub_style == "Necklace_Modern_Minimalist":
            p_center = (
                "Examine this high-resolution macro view of the Modern Minimalist Necklace's Central Drop / Pendant Section.\n"
                f"{stone_rule}\n"
                "Describe ONLY what is physically visible in this modern piece (do NOT mention temple motifs, peacocks, yali, or kemp stones):\n"
                "- Upper Connector Ring: Describe the small circular O-ring connector where the upper chains meet, noting the tiny round white pavé stones set into the ring.\n"
                "- Vertical Drop Chain: Describe the delicate vertical gold chain link suspending the lower pendant from the O-ring.\n"
                "- Lower Enamel / Pendant Terminal: Describe the oval bezel-set terminal drop at the bottom and its geometric color-blocked pastel enamel sections (mint-green, soft pink, and yellow)."
            )
            p_side = (
                "Examine this high-resolution macro view of the Modern Necklace's Upper Chain & Clasp Section.\n"
                f"{stone_rule}\n"
                "Describe ONLY the actual chain and clasp components visible:\n"
                "- Chain Link Geometry: Describe the thin, delicate yellow-gold chain links.\n"
                "- Clasp & Extension Mechanism: Describe the lobster-claw clasp, jump rings, and the small hexagonal/floral gold tag at the end of the chain."
            )
        else:
            p_center = (
                "Examine this high-resolution macro view of the Necklace's Central Focal Section.\n"
                f"{stone_rule}\n"
                "Provide a factual 3-point breakdown of what is physically visible:\n"
                "- Central Motif & Relief: Describe the centerpiece geometry, surface metalwork, or sculpted motifs physically present.\n"
                "- Surrounding Frame & Stone/Bead Accents: Describe the framing borders, granulation beading, or stone settings.\n"
                "- Lower Edge & Suspended Drops: Describe any hanging drops, beads, or fringe along the bottom rim."
            )
            p_side = (
                "Examine this high-resolution macro view of the Necklace's Side Chain Section.\n"
                f"{stone_rule}\n"
                "Provide a factual 3-point breakdown:\n"
                "- Repeating Link Structure: Describe the shape and relief work of the repeating side chain segments.\n"
                "- Stone or Metal Accents: Describe any stones or metalwork details positioned on or between the links.\n"
                "- Outer & Inner Borders: Describe the perimeter edges and connecting rings."
            )
        return p_center, p_side

    if category == "Earring":
        if sub_style == "Earring_Compact_Stud":
            p_macro = (
                "Examine this high-resolution macro view of the Compact Stud Earring.\n"
                f"{stone_rule}\n"
                "Provide a factual 3-point breakdown of its compact structure (no hanging drop):\n"
                "- Overall Stud Silhouette: Describe the outer shape and layout of the stud.\n"
                "- Metalwork Section: Describe the solid gold bar, frame, or surface texture.\n"
                "- Gemstone Arrangement: Describe the count, rows/columns, color, cut, and prong/bezel mounting of any stones present."
            )
        else:
            p_macro = (
                "Examine this high-resolution macro view of the Linear Drop / Dangle Earring from top to bottom.\n"
                f"{stone_rule}\n"
                "Describe ONLY the exact physical components visible from top to bottom (do NOT invent a jhumka bell dome if the bottom is a spherical ball or simple drop):\n"
                "- Upper Ear-Wire Hook / Top Section: Describe the long curved gold wire hook or top post at the upper end.\n"
                "- Middle Chain & Ring Links: Describe the suspended vertical elongated oval paperclip links and the small circular beaded gold ring in the mid-section.\n"
                "- Bottom Terminal Drop: Describe the smooth, high-polish solid gold spherical ball drop hanging from the bottom loop (or the exact bottom element physically visible)."
            )
        return p_macro, p_macro

    if category == "Ring":
        p_macro = (
            "Examine this high-resolution macro view of the Ring.\n"
            f"{stone_rule}\n"
            "Provide a factual 3-point breakdown of its physical construction:\n"
            "- Top Head Silhouette & Gemstone Row: Describe the curved leaf/marquise-shaped top head and the central diagonal row of round white stones set in gold prongs/channel.\n"
            "- Openwork Mesh Frame & Gallery: Describe the perforated openwork honeycomb/grid mesh lattice flanking both sides of the central stone row on the head.\n"
            "- Shoulders & Circular Band (Shank): Describe how the pointed ends of the leaf head join the smooth, solid, high-polish yellow-gold circular band."
        )
        return p_macro, p_macro

    p_macro = (
        f"Examine this high-resolution macro view of the {category}.\n"
        f"{stone_rule}\n"
        "Describe its top section, central body, and outer border based strictly on what is physically visible."
    )
    return p_macro, p_macro

def build_micro_prompt(category: str, sub_style: str, is_plain_metal: bool) -> str:
    if is_plain_metal:
        return (
            f"Examine this magnified close-up crop of the 100% solid-gold {category}.\n"
            "This piece contains ZERO gemstones. Write ONLY 3 concise numbered points describing what is physically visible inside this crop:\n"
            "1. Material Confirmation: State '100% solid yellow-gold metalwork; no gemstones'.\n"
            "2. Physical Component & Shape: Identify the exact gold part visible in this crop (such as a curved ear-wire hook, elongated oval paperclip link, small beaded gold ring, or smooth spherical gold ball drop).\n"
            "3. Surface Finish & Joint: Describe the high-polish smooth or beaded gold surface and how the loops interlock."
        )

    if sub_style == "Bangle_Broad_Cylindrical_Kada":
        return (
            "Examine this magnified close-up crop of the Broad Cylindrical Cuff Kada.\n"
            "Write ONLY 3 concise numbered points describing what is physically visible inside this crop:\n"
            "1. Cylindrical Tubes, Beads & Stones: Describe the glossy red cylindrical coral/stone tubes, carved pale-green melon end-beads, or tiny white-stone collar rings visible in this crop (or state 'Solid gold metalwork only in this crop' if only gold bars appear).\n"
            "2. Ridged Gold Bars & Frame: Describe the parallel finely ridged/fluted cylindrical yellow-gold bars, small gold spacer balls, or engraved side rim.\n"
            "3. Structural Assembly: Describe how the cylindrical tubes and ridged gold rods are mounted to the circular frame and inner backing."
        )

    if sub_style == "Bangle_Single_Band":
        return (
            "Examine this magnified close-up crop of the Gemstone Bangle.\n"
            "Write ONLY 3 concise numbered points describing what is physically visible inside this crop:\n"
            "1. Gemstones & Cuts: State the exact color and cut shape of the stones visible in this crop (such as pear/marquise-cut pink-red rubies arranged in a 4-petal floral cluster, and small round green emerald accents along the side edge).\n"
            "2. Gold Settings & Openwork Frame: Describe the gold bezel/prong settings holding each petal stone and the openwork gold cutouts between the clusters.\n"
            "3. Band Alignment: Describe how the floral stone clusters and green edge stones align along the curved gold band."
        )

    if sub_style == "Necklace_Modern_Minimalist":
        return (
            "Examine this magnified close-up crop of the Modern Minimalist Chain Necklace.\n"
            "Do NOT mention temple motifs, peacocks, yali, or red kemp stones. Write ONLY 3 concise numbered points describing what is physically visible inside this crop:\n"
            "1. Stones or Enamel Inlay: Describe the small round white pavé stones on the circular O-ring OR the flat geometric pastel enamel sections (mint-green, pink, yellow) inside the oval drop, OR state 'Solid gold chain/clasp metalwork only in this crop' if no stones or enamel appear in this view.\n"
            "2. Chain & Frame Metalwork: Describe the delicate gold cable chain links, circular connector ring, lobster clasp, or smooth oval gold bezel frame visible in this crop.\n"
            "3. Linkage & Suspension: Describe how the fine gold chain attaches to the ring, clasp, or pendant drop."
        )

    if category == "Ring":
        return (
            "Examine this magnified close-up crop of the Ring.\n"
            "Write ONLY 3 concise numbered points describing what is physically visible inside this crop:\n"
            "1. Gemstones & Mounting: Describe the round faceted white diamonds/stones set in a diagonal gold channel/prong row down the center (or state 'Solid gold metalwork only in this crop' if no stones appear).\n"
            "2. Openwork Mesh & Gold Finish: Describe the perforated openwork grid/honeycomb gold lattice surrounding the central stone row and the smooth polished outer gold rim.\n"
            "3. Head-to-Shank Transition: Describe how the curved leaf-shaped head frame tapers into the smooth solid gold band."
        )

    return (
        f"Examine this magnified close-up crop of the {category}.\n"
        "Write ONLY 3 concise numbered points describing what is physically visible inside this crop:\n"
        "1. Gemstones, Beads, or Enamel: State the exact color, shape, and setting of any stones, beads, or enamel physically visible in this crop, or state 'Solid gold metalwork only in this crop'.\n"
        "2. Metalwork Geometry & Finish: Describe the exact gold shapes and surface textures visible inside this frame.\n"
        "3. Structural Connections: Describe how the elements in this crop connect to adjacent parts."
    )

# ==========================================
# 7. MAIN EXECUTION (FULL DATASET, AUTO-RESUME)
# ==========================================
def main():
    print("=" * 90, flush=True)
    print("🚀 [1/4] Loading InternVL2-8B strictly to GPU 0 (4-bit NF4 Quantized)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()
    print("✅ [VLM] InternVL2-8B Loaded Successfully!", flush=True)

    sam_engine = UniversalSam2GpuGenerator(SAM_PT_PATH)

    if RUN_FULL_DATASET:
        print(f"\n📁 [2/4] Scanning Full Dataset Directory: {DATASET_DIR}", flush=True)
        all_image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for p in DATASET_DIR.rglob(ext):
                if "output_with_sam" in p.parts or "csv" in p.parts:
                    continue
                all_image_paths.append(p)
        all_image_paths.sort()
    else:
        all_image_paths = []

    total_images = len(all_image_paths)
    print(f"✅ Ready to process full dataset: {total_images} images.", flush=True)
    print(f"📂 SAM-2 Segmented Images Root: {SAM_OUTPUT_ROOT}", flush=True)
    print(f"📄 Output CSV Path:             {OUTPUT_CSV}", flush=True)
    print("=" * 90, flush=True)

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
        if not img_path.exists():
            continue

        sub_folder = img_path.parent.name
        main_folder = img_path.parent.parent.name
        short_id = f"{main_folder}/{sub_folder}/{img_path.name}"

        if img_str_path in processed_paths:
            continue

        prod_folder_name = get_clean_product_folder_name(img_path)
        image_sam_dir = SAM_OUTPUT_ROOT / main_folder / sub_folder / prod_folder_name
        image_sam_dir.mkdir(parents=True, exist_ok=True)

        category = detect_category(f"{sub_folder}_{img_path.name}")

        try:
            img_bgr = cv2.imread(img_str_path)
            if img_bgr is None:
                continue

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            _, pil_full = get_macro_bgr_crop(img_bgr, mask, center_pt=None)
            raw_output_sections = []

            # PASS 1: GLOBAL OVERVIEW
            p_global = GLOBAL_PROMPTS.get(category, GLOBAL_PROMPTS["Pendant"])
            global_report = ask_internvl(model, tokenizer, pil_full, p_global, max_tokens=420)

            is_plain = determine_if_plain_metal(global_report)
            sub_style = detect_sub_style(category, global_report, img_bgr)
            lock_str = "100% PLAIN METAL" if is_plain else "GEMSTONE / ENAMEL + METAL"

            raw_output_sections.append(
                f"=== GLOBAL OVERVIEW (Category: {sub_folder} | Sub-Style: {sub_style} | Lock: {lock_str}) ===\n{global_report}"
            )

            p_macro_center, p_macro_side = build_macro_prompts(category, sub_style, is_plain, global_report)
            p_micro = build_micro_prompt(category, sub_style, is_plain)

            if category == "Necklace":
                focal_pt, side_pt, has_side = find_necklace_macro_zones(mask)
                c_bgr, c_pil = get_macro_bgr_crop(img_bgr, mask, focal_pt, 0.45)
                zones = [("01_FOCAL_CENTER", "FOCAL_CENTER", "🎯 FOCAL CENTER MACRO", c_bgr, c_pil, p_macro_center, 4)]
                if has_side:
                    s_bgr, s_pil = get_macro_bgr_crop(img_bgr, mask, side_pt, 0.42)
                    zones.append(("02_SIDE_STRUCTURE", "SIDE_STRUCTURE", "🔗 SIDE STRUCTURE MACRO", s_bgr, s_pil, p_macro_side, 4))
            elif category == "Earring":
                e_bgr, e_pil = isolate_single_earring_if_pair(img_bgr, mask)
                zones = [("01_PRIMARY_EARRING", "PRIMARY_EARRING", f"🎯 {sub_style.upper()} MACRO", e_bgr, e_pil, p_macro_center, 6)]
            else:
                f_bgr, f_pil = get_macro_bgr_crop(img_bgr, mask, center_pt=None)
                zones = [("01_FULL_ITEM", "FULL_ITEM", f"🎯 {sub_style.upper()} MACRO", f_bgr, f_pil, p_macro_center, 6)]

            # PASS 2 & 3: MACRO + MICRO
            for subfolder_tag, zone_code, zone_title, macro_bgr, pil_macro, macro_prompt, max_parts in zones:
                macro_report = ask_internvl(model, tokenizer, pil_macro, macro_prompt, max_tokens=420)
                raw_output_sections.append(f"\n=== {zone_title} (Material Lock: {lock_str}) ===\n{macro_report}")

                zone_save_dir = image_sam_dir / subfolder_tag
                micro_items, _ = get_or_create_sam_crops(
                    sam_engine, macro_bgr, zone_save_dir, zone_code, max_parts
                )

                for i, (pil_micro, is_pair) in enumerate(micro_items, start=1):
                    micro_report = ask_internvl(model, tokenizer, pil_micro, p_micro, max_tokens=320)
                    raw_output_sections.append(
                        f"\n--- [{zone_code} | Micro-Detail {i}/{len(micro_items)}] ---\n{micro_report}"
                    )

            full_raw_output = "\n".join(raw_output_sections).strip()

            results.append({
                "Image_Path": img_str_path,
                "Brand_Name": main_folder,
                "Category": sub_folder,
                "Sub_Style": sub_style,
                "Full_Raw_Output": full_raw_output
            })

            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, quoting=1)
            print(f"[{idx}/{total_images}][SAVED] {short_id} ({len(results)} total)", flush=True)

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"[{idx}/{total_images}][ERROR] {short_id} | {e}", flush=True)

    print("=" * 90, flush=True)
    print(f"🎉 FULL DATASET COMPLETE! Processed {len(results)} images.", flush=True)
    print(f"📄 Saved to: {OUTPUT_CSV}", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()