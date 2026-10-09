import os
import numpy as np
from PIL import Image
import concurrent.futures

# --- 1. PATH CONFIGURATION ---
input_base_dir = r"D:\gen jewels\web scraping\dataset finetuned"
# We create a new folder so we don't accidentally overwrite your raw extractions!
output_base_dir = r"D:\gen jewels\web scraping\dataset_1024_framing"

# Target SDXL Canvas Size
CANVAS_SIZE = 1024
# The jewelry should take up 80% of the canvas (about 819 pixels max), leaving a 10% black border
MAX_JEWELRY_SIZE = int(CANVAS_SIZE * 0.8)

def process_single_image(img_path, output_path):
    """Math-based crop and center (Zero stretching, No AI needed)"""
    try:
        # 1. Open the image and convert to numpy array for fast math
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_array = np.array(img)
            
            # 2. Find the Bounding Box (Ignore all pure black #000000 pixels)
            # Create a mask of all pixels that are NOT pure black
            mask = img_array > 0
            # Find the exact X and Y coordinates where the gold exists
            coords = np.argwhere(mask[:, :, 0]) # Check red channel (works for gold/white)
            
            if coords.size == 0:
                print(f" [WARNING] Completely black image found, skipping: {img_path}")
                return
                
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 3. Crop tightly around the jewelry
            cropped_img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
            
            # 4. Calculate Resize Ratio (Never stretch, and never enlarge past original size to prevent blur)
            crop_width, crop_height = cropped_img.size
            
            # Find which side is the longest (to fit within our 819px max limit)
            longest_side = max(crop_width, crop_height)
            
            # If the jewelry is larger than 819px, shrink it. If it's smaller, keep it its original size!
            ratio = min(1.0, MAX_JEWELRY_SIZE / longest_side)
            
            new_width = int(crop_width * ratio)
            new_height = int(crop_height * ratio)
            
            # 5. Resize using High-Quality Lanczos Filter
            resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 6. Create the perfectly pure black 1024x1024 Canvas
            final_canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0))
            
            # 7. Calculate exact center coordinates for pasting
            paste_x = (CANVAS_SIZE - new_width) // 2
            paste_y = (CANVAS_SIZE - new_height) // 2
            
            # Paste the jewelry dead-center
            final_canvas.paste(resized_img, (paste_x, paste_y))
            
            # 8. Save
            final_canvas.save(output_path, "PNG")
            
    except Exception as e:
        print(f" [ERROR] Failed on {os.path.basename(img_path)}: {str(e)}")

def run_normalization():
    print(f"🚀 Starting Phase 2: Canvas Normalization (1024x1024)...")
    
    # Gather all images
    tasks = []
    for root, dirs, files in os.walk(input_base_dir):
        relative_path = os.path.relpath(root, input_base_dir)
        target_output_dir = os.path.join(output_base_dir, relative_path)
        os.makedirs(target_output_dir, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                output_path = os.path.join(target_output_dir, file)
                
                # Smart resume (skip if already exists)
                if not os.path.exists(output_path):
                    tasks.append((img_path, output_path))
                else:
                    print(f" [SKIPPED] {file}")

    print(f"Found {len(tasks)} images to process. Firing up CPU Multi-Threading...")

    # MULTI-THREADING: This acts like a GPU by using all your CPU cores at once!
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_image, img_path, out_path) for img_path, out_path in tasks]
        
        # Show progress
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if count % 100 == 0:
                print(f"✅ Processed {count} / {len(tasks)} images...")

    print("\n🎉 PHASE 2 COMPLETE! All images are 1024x1024, centered, and ready for SDXL!")

if __name__ == "__main__":
    run_normalization()