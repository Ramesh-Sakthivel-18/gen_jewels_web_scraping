import cv2
import numpy as np
import os
from pathlib import Path

# ==========================================
# CONFIGURATION & EXACT INPUT IMAGES
# ==========================================
TARGET_IMAGES = [
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod38_Angle1_1200x1200_8477ef_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod24_Angle1_1200x1200_7030ff_clean_black_bg.png"
]

# Auto-create the new upscaled output folder
BASE_OUTPUT_DIR = r"D:\gen jewels\web scraping\code\caption\seperation\output_upscale"
UPSCALE_RESOLUTION = (896, 896) # Target High-Res size for VLM
PADDING_RATIO = 0.15

# ==========================================
# 1. CORE MATHEMATICS: FINDING THE POINTS
# ==========================================
def find_focal_and_structural_centers(mask):
    # FIND FOCAL CENTER (Thickest part of the jewelry)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
    focal_center = max_loc 

    # ISOLATE ARMS/BANDS 
    h, w = mask.shape
    exclude_radius = int(max(w, h) * 0.25) 
    structural_mask = mask.copy()
    cv2.circle(structural_mask, focal_center, exclude_radius, 0, -1)

    y_coords, x_coords = np.where(structural_mask > 0)
    
    # FIND STRUCTURAL CENTERS (Left/Right)
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
# 2. EXTRACT RAW CROPS (No Resizing Yet)
# ==========================================
def get_raw_square_crop(image, mask, center_pt, box_size):
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
    
    # Paste onto a raw black square canvas
    cw, ch = cropped_img.shape[1], cropped_img.shape[0]
    side = max(cw, ch)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    
    x_offset, y_offset = (side - cw) // 2, (side - ch) // 2
    canvas[y_offset:y_offset+ch, x_offset:x_offset+cw] = cropped_img
    
    return canvas

# ==========================================
# 3. MATHEMATICAL UPSCALING ALGORITHM
# ==========================================
def upscale_image(raw_image, target_resolution):
    """
    Applies Lanczos4 interpolation to mathematically calculate and 
    upscale missing pixels without hallucinating fake details.
    """
    return cv2.resize(raw_image, target_resolution, interpolation=cv2.INTER_LANCZOS4)

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def process_images():
    # Auto-create the output folder
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Get current highest numbered folder to continue sequentially (1, 2, 3...)
    existing_folders = [int(d) for d in os.listdir(BASE_OUTPUT_DIR) if d.isdigit()]
    start_idx = max(existing_folders) + 1 if existing_folders else 1

    for i, img_path in enumerate(TARGET_IMAGES):
        if not os.path.exists(img_path):
            print(f"❌ Could not find: {img_path}")
            continue
            
        filename = Path(img_path).name
        print(f"\n⚙️ Processing: {filename}")
        
        # Create numbered output folder
        current_out_dir = os.path.join(BASE_OUTPUT_DIR, str(start_idx + i))
        os.makedirs(current_out_dir, exist_ok=True)
        
        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Create Gold Mask
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        # Get Mathematical Centers
        print("   -> Finding structural centers...")
        focal_pt, struct_1, struct_2 = find_focal_and_structural_centers(mask)
        box_size = int(max(w, h) * 0.45)
        
        # Extract RAW crops (No resizing yet)
        print("   -> Extracting raw tight crops...")
        raw_focal = get_raw_square_crop(img, mask, focal_pt, box_size)
        raw_struct_1 = get_raw_square_crop(img, mask, struct_1, box_size)
        raw_struct_2 = get_raw_square_crop(img, mask, struct_2, box_size)
        
        # Sort structurally (Left vs Right)
        struct_pts_and_raw = sorted([(struct_1, raw_struct_1), (struct_2, raw_struct_2)], key=lambda item: item[0][0])
        
        # Upscale the RAW crops to High-Res VLM format
        print("   -> Upscaling images using Lanczos4 Algorithm...")
        upscaled_focal = upscale_image(raw_focal, UPSCALE_RESOLUTION)
        upscaled_left = upscale_image(struct_pts_and_raw[0][1], UPSCALE_RESOLUTION)
        upscaled_right = upscale_image(struct_pts_and_raw[1][1], UPSCALE_RESOLUTION)
        
        # Save ONLY the Upscaled outputs
        cv2.imwrite(os.path.join(current_out_dir, f"1_Focal_Center_Upscaled.png"), upscaled_focal)
        cv2.imwrite(os.path.join(current_out_dir, f"2_Structure_Left_Upscaled.png"), upscaled_left)
        cv2.imwrite(os.path.join(current_out_dir, f"3_Structure_Right_Upscaled.png"), upscaled_right)
        
        # Save Debug Map to verify where the math looked
        debug_img = img.copy()
        cv2.circle(debug_img, focal_pt, 15, (0, 0, 255), -1)
        cv2.circle(debug_img, struct_pts_and_raw[0][0], 15, (0, 255, 0), -1) 
        cv2.circle(debug_img, struct_pts_and_raw[1][0], 15, (255, 0, 0), -1) 
        cv2.imwrite(os.path.join(current_out_dir, f"0_Debug_Map.png"), debug_img)
        
        print(f"✅ Saved perfectly separated and upscaled crops to: {current_out_dir}")

if __name__ == "__main__":
    process_images()