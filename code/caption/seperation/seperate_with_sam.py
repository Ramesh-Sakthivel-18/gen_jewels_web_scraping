import os
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
TARGET_IMAGES = [
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod38_Angle1_1200x1200_8477ef_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod24_Angle1_1200x1200_7030ff_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\Candere\Rings\Candere_Rings_Prod1_Angle12_1100x1100_21c7ae_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\Candere\Earrings\Candere_Earrings_Prod163_Angle1_1100x1100_71a1d4_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\CC_Earrings\CC_Earrings_Prod2_Angle12_1200x1200_bd49e8_clean_black_bg.png"
]

SAM_MODEL_DIR = Path(r"D:\gen jewels\web scraping\models\sam2.1-hiera-large")
SAM_PT_PATH = SAM_MODEL_DIR / "sam2.1_hiera_large.pt"
SAM_YAML_NAME = "sam2.1_hiera_l.yaml"

BASE_OUTPUT_DIR = Path(r"D:\gen jewels\web scraping\code\caption\seperation\output_with_sam")

UPSCALE_RES = (1120, 1120)
MACRO_PAD_RATIO = 0.05
MIN_CONTEXT_WINDOW = 400       # Preserves full arches, pedestals, and frames around motifs
PAIR_MERGE_DISTANCE = 160      # Merges adjacent twin figures/motifs into a single paired crop

# ==========================================
# 2. SAM-2 GPU ENGINE LOADER
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
                config_file=SAM_YAML_NAME, ckpt_path=str(SAM_PT_PATH.resolve()), device=device, mode="eval"
            )
            self.generator = SAM2AutomaticMaskGenerator(
                model=sam2_model, points_per_side=32, points_per_batch=64, 
                pred_iou_thresh=0.86, stability_score_thresh=0.88, crop_n_layers=1, min_mask_region_area=120
            )
            self.mode = "native_sam2"
            print("✅ SAM-2.1 Loaded on GPU (Native Meta CUDA Engine)!")
        except ImportError:
            from transformers import pipeline
            self.generator = pipeline(
                "mask-generation", model=str(self.model_dir), device=0 if torch.cuda.is_available() else -1, torch_dtype=torch.float16
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
                Image.fromarray(rgb_np), points_per_batch=64, pred_iou_thresh=0.86, 
                stability_score_thresh=0.88, crops_n_layers=1
            )
            return [np.array(m).astype(np.uint8) for m in outputs.get("masks", [])]

# ==========================================
# 3. UTILITIES & OPENCV MACRO ZONING
# ==========================================
def create_next_run_folder(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()]
    next_num = 1 if not existing_runs else max(int(d.name[4:]) for d in existing_runs) + 1
    run_dir = base_dir / f"run_{next_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def compute_box_iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    a1, a2 = (b1[2] - b1[0]) * (b1[3] - b1[1]), (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(a1 + a2 - inter)

def find_necklace_macro_zones(mask):
    """Finds exact focal center and side structure for Necklaces."""
    h, w = mask.shape
    max_dim = max(h, w)
    k_size = max(15, int(max_dim * 0.04))
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    dist_1 = cv2.distanceTransform(closed_mask, cv2.DIST_L2, 5)
    _, _, _, focal_pt = cv2.minMaxLoc(dist_1)

    struct_mask = closed_mask.copy()
    cv2.circle(struct_mask, focal_pt, int(max_dim * 0.28), 0, -1)
    dist_2 = cv2.distanceTransform(struct_mask, cv2.DIST_L2, 5)
    _, max_val_2, _, side_pt = cv2.minMaxLoc(dist_2)

    return focal_pt, side_pt, max_val_2 > (max_dim * 0.015)

def get_macro_bgr_crop(image_bgr, mask, center_pt, box_ratio=0.45):
    h, w = image_bgr.shape[:2]
    if center_pt is None: # For full object bounding box (compact jewelry)
        coords = np.argwhere(mask > 0)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
    else:
        cx, cy = center_pt
        half_box = int(max(h, w) * box_ratio) // 2
        x1, y1 = max(0, cx - half_box), max(0, cy - half_box)
        x2, y2 = min(w, cx + half_box), min(h, cy + half_box)

    pad = int(max(x2 - x1, y2 - y1) * MACRO_PAD_RATIO)
    x1_f, y1_f = max(0, x1 - pad), max(0, y1 - pad)
    x2_f, y2_f = min(w, x2 + pad), min(h, y2 + pad)

    cropped = image_bgr[y1_f:y2_f, x1_f:x2_f]
    side = max(cropped.shape[0], cropped.shape[1])
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y_off, x_off = (side - cropped.shape[0]) // 2, (side - cropped.shape[1]) // 2
    canvas[y_off:y_off + cropped.shape[0], x_off:x_off + cropped.shape[1]] = cropped
    return cv2.resize(canvas, UPSCALE_RES, interpolation=cv2.INTER_LANCZOS4)

# ==========================================
# 4. MICRO EXTRACTION & SAVING
# ==========================================
def extract_and_save_sam_crops(sam_engine, macro_bgr, zone_dir: Path, zone_name: str, max_parts: int):
    """Runs SAM, merges pairs, creates context crops, and saves them to disk."""
    zone_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(zone_dir / f"00_{zone_name}_MACRO.png"), macro_bgr)

    H, W = macro_bgr.shape[:2]
    total_area = H * W
    rgb_macro = cv2.cvtColor(macro_bgr, cv2.COLOR_BGR2RGB)

    print(f"      ⚙️ Running SAM-2 GPU inference on {zone_name}...")
    raw_masks = sam_engine.generate_masks(rgb_macro)

    candidates = []
    for mask_np in raw_masks:
        ys, xs = np.where(mask_np > 0)
        if ys.size == 0: continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bw, bh = x2 - x1, y2 - y1
        box_area, mask_area = bw * bh, int(ys.size)

        if mask_area < (total_area * 0.005) or box_area > (total_area * 0.60): continue
        if max(bw, bh) / max(1, min(bw, bh)) > 5.5: continue
        if macro_bgr[mask_np > 0].mean() < 25: continue

        candidates.append({
            "raw_box": (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "mask": mask_np, "area": mask_area
        })

    candidates.sort(key=lambda c: c["area"], reverse=True)

    # Cluster Pairs
    merged_units = []
    used = set()
    for i in range(len(candidates)):
        if i in used: continue
        c1 = candidates[i]
        curr_box = list(c1["raw_box"])
        combined_mask = c1["mask"].copy()
        is_pair = False
        for j in range(i + 1, len(candidates)):
            if j in used: continue
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
            "raw_box": tuple(curr_box), "center": ((mx1 + mx2) // 2, (my1 + my2) // 2),
            "mask": combined_mask, "is_pair": is_pair
        })

    # Spatial Deduplication & Context Expanding
    unique_units = []
    for unit in merged_units:
        cx, cy = unit["center"]
        bx1, by1, bx2, by2 = unit["raw_box"]
        bw, bh = bx2 - bx1, by2 - by1

        crop_dim = max(MIN_CONTEXT_WINDOW, int(max(bw, bh) * 1.55))
        half_dim = crop_dim // 2

        px1, py1 = max(0, cx - half_dim), max(0, cy - half_dim)
        px2, py2 = min(W, cx + half_dim), min(H, cy + half_dim)
        ctx_box = (px1, py1, px2, py2)

        is_dup = any(np.hypot(cx - k["center"][0], cy - k["center"][1]) < 140 or 
                     compute_box_iou(ctx_box, k["ctx_box"]) > 0.45 for k in unique_units)
        
        if not is_dup:
            unit["ctx_box"] = ctx_box
            unique_units.append(unit)
            if len(unique_units) >= max_parts: break

    # Draw Map & Save Files
    annotated_map = macro_bgr.copy()
    for idx, unit in enumerate(unique_units, start=1):
        x1, y1, x2, y2 = unit["raw_box"]
        px1, py1, px2, py2 = unit["ctx_box"]
        label = f"#{idx} PAIR" if unit["is_pair"] else f"#{idx}"

        cv2.rectangle(annotated_map, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(annotated_map, (px1, py1), (px2, py2), (255, 255, 0), 1)
        cv2.putText(annotated_map, label, (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        context_crop = macro_bgr[py1:py2, px1:px2]
        cutout_crop = cv2.bitwise_and(macro_bgr, macro_bgr, mask=unit["mask"] * 255)[py1:py2, px1:px2]

        for suffix, sub_img in [("context", context_crop), ("cutout", cutout_crop)]:
            sh, sw = sub_img.shape[:2]
            side = max(sh, sw)
            canvas = np.zeros((side, side, 3), dtype=np.uint8)
            yo, xo = (side - sh) // 2, (side - sw) // 2
            canvas[yo:yo + sh, xo:xo + sw] = sub_img
            cv2.imwrite(str(zone_dir / f"part_{idx:02d}_{suffix}_1120x1120.png"), cv2.resize(canvas, UPSCALE_RES, interpolation=cv2.INTER_LANCZOS4))

    cv2.imwrite(str(zone_dir / f"01_{zone_name}_SAM_MAP.png"), annotated_map)
    return len(unique_units)

# ==========================================
# 5. MAIN EXECUTION (DYNAMIC ROUTING)
# ==========================================
def main():
    run_dir = create_next_run_folder(BASE_OUTPUT_DIR)
    print(f"📁 Saving crops to: {run_dir}")
    print("🚀 Loading SAM-2.1 on GPU...")
    sam_engine = UniversalSam2GpuGenerator(SAM_MODEL_DIR)

    for img_idx, img_path in enumerate(TARGET_IMAGES, start=1):
        path_obj = Path(img_path)
        if not path_obj.exists(): continue
        
        # Check if the folder or file name contains 'Necklace'
        is_necklace = "Necklace" in path_obj.parts or "Necklace" in path_obj.name
        
        photo_dir = run_dir / f"{img_idx:02d}_{path_obj.stem.split('_1200x1200')[0]}"
        photo_dir.mkdir(parents=True, exist_ok=True)
        img_bgr = cv2.imread(str(path_obj))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(photo_dir / "00_ORIGINAL_FULL.png"), img_bgr)

        print(f"\n🖼️ Processing [{img_idx}/{len(TARGET_IMAGES)}]: {path_obj.name}")
        
        if is_necklace:
            print("   🔗 Detected [Necklace]: Splitting into Center (Top 4) and Side (Top 4).")
            focal_pt, side_pt, has_side = find_necklace_macro_zones(mask)
            
            macro_center_bgr = get_macro_bgr_crop(img_bgr, mask, focal_pt, 0.45)
            n_center = extract_and_save_sam_crops(sam_engine, macro_center_bgr, photo_dir / "01_FOCAL_CENTER", "CENTER", max_parts=4)
            print(f"   ✅ Saved {n_center} parts in CENTER.")
            
            if has_side:
                macro_side_bgr = get_macro_bgr_crop(img_bgr, mask, side_pt, 0.42)
                n_side = extract_and_save_sam_crops(sam_engine, macro_side_bgr, photo_dir / "02_SIDE_STRUCTURE", "SIDE", max_parts=4)
                print(f"   ✅ Saved {n_side} parts in SIDE.")
        else:
            print("   💍 Detected [Compact Jewelry] (Earring/Ring/Bangle): Processing Full Frame (Top 8).")
            macro_full_bgr = get_macro_bgr_crop(img_bgr, mask, center_pt=None)
            n_parts = extract_and_save_sam_crops(sam_engine, macro_full_bgr, photo_dir / "01_FULL_ITEM", "FULL_ITEM", max_parts=8)
            print(f"   ✅ Saved {n_parts} parts from FULL ITEM.")

    print(f"\n🎉 Done! Inspect your dynamic crops at: {run_dir}")

if __name__ == "__main__":
    main()