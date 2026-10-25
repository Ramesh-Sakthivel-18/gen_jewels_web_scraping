import cv2
import numpy as np
import os
from pathlib import Path

# ==========================================
# 1. EXACT INPUT IMAGES & DIRECTORY CONFIG
# ==========================================
TARGET_IMAGES = [
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod38_Angle1_1200x1200_8477ef_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod24_Angle1_1200x1200_7030ff_clean_black_bg.png"
]

# Saving to a new folder so it doesn't mix with your upscaled ones
BASE_OUTPUT_DIR = r"D:\gen jewels\web scraping\code\caption\seperation\output_raw_crops"
PADDING_RATIO = 0.15

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. CORE MATHEMATICS (Follow the Gold)
# ==========================================
def find_focal_and_structural_centers(mask):
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, focal_center = cv2.minMaxLoc(dist_transform)

    h, w = mask.shape
    exclude_radius = int(max(w, h) * 0.25) 
    structural_mask = mask.copy()
    cv2.circle(structural_mask, focal_center, exclude_radius, 0, -1)

    y_coords, x_coords = np.where(structural_mask > 0)
    
    if len(x_coords) > 100:
        Z = np.column_stack((x_coords, y_coords)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        struct_1 = (int(centers[0][0]), int(centers[0][1]))
        struct_2 = (int(centers[1][0]), int(centers[1][1]))
    else:
        struct_1, struct_2 = focal_center, focal_center

    return focal_center, struct_1, struct_2

# ==========================================
# 3. RAW CROP ONLY (NO UPSCALING)
# ==========================================
def process_raw_crop(image, mask, center_pt, box_size):
    h, w = image.shape[:2]
    cx, cy = center_pt
    half_box = box_size // 2

    x1, y1 = max(0, cx - half_box), max(0, cy - half_box)
    x2, y2 = min(w, cx + half_box), min(h, cy + half_box)

    local_mask = mask[y1:y2, x1:x2]
    coords = np.argwhere(local_mask > 0)
    
    if coords.size > 0:
        sy_min, sx_min = coords.min(axis=0)
        sy_max, sx_max = coords.max(axis=0)
        x1_tight, y1_tight = x1 + sx_min, y1 + sy_min
        x2_tight, y2_tight = x1 + sx_max, y1 + sy_max
    else:
        x1_tight, y1_tight, x2_tight, y2_tight = x1, y1, x2, y2

    bw, bh = x2_tight - x1_tight, y2_tight - y1_tight
    pad = int(max(bw, bh) * PADDING_RATIO)
    x1_final, y1_final = max(0, x1_tight - pad), max(0, y1_tight - pad)
    x2_final, y2_final = min(w, x2_tight + pad), min(h, y2_tight + pad)
    
    cropped_img = image[y1_final:y2_final, x1_final:x2_final]
    
    cw, ch = cropped_img.shape[1], cropped_img.shape[0]
    side = max(cw, ch)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    
    x_offset, y_offset = (side - cw) // 2, (side - ch) // 2
    canvas[y_offset:y_offset+ch, x_offset:x_offset+cw] = cropped_img
    
    # ❌ NO CV2.RESIZE HERE. Returning the exact raw pixel dimensions!
    return canvas

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    for img_path in TARGET_IMAGES:
        if not os.path.exists(img_path):
            print(f"❌ Could not find: {img_path}")
            continue
            
        filename = Path(img_path).name
        stem_name = Path(img_path).stem
        print(f"\n⚙️ Processing: {filename}")
        
        image_out_dir = os.path.join(BASE_OUTPUT_DIR, stem_name)
        os.makedirs(image_out_dir, exist_ok=True)
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        focal_pt, struct_1, struct_2 = find_focal_and_structural_centers(mask)
        box_size = int(max(w, h) * 0.45)
        
        # Extract RAW crops
        img_focal = process_raw_crop(img, mask, focal_pt, box_size)
        img_struct_1 = process_raw_crop(img, mask, struct_1, box_size)
        img_struct_2 = process_raw_crop(img, mask, struct_2, box_size)
        
        structs = sorted([(struct_1, img_struct_1), (struct_2, img_struct_2)], key=lambda item: item[0][0])
        
        # Save RAW crops
        cv2.imwrite(os.path.join(image_out_dir, f"1_Focal_Center.png"), img_focal)
        cv2.imwrite(os.path.join(image_out_dir, f"2_Structure_Left.png"), structs[0][1])
        cv2.imwrite(os.path.join(image_out_dir, f"3_Structure_Right.png"), structs[1][1])
        
        debug_img = img.copy()
        cv2.circle(debug_img, focal_pt, 15, (0, 0, 255), -1)
        cv2.circle(debug_img, structs[0][0], 15, (0, 255, 0), -1)
        cv2.circle(debug_img, structs[1][0], 15, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(image_out_dir, f"0_Debug_Map.png"), debug_img)
        
        print(f"✅ Separated and saved raw sizes to: {image_out_dir}")

if __name__ == "__main__":
    main()