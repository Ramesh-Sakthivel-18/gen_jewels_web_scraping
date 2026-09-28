import os
import glob
from PIL import Image
from rembg import remove, new_session

# --- PATH CONFIGURATION ---
input_base_dir = r"D:\gen jewels\web scraping\Dataset\DarJewellery"
output_base_dir = r"D:\gen jewels\web scraping\Dataset\dar_jewelry_watermark_removed"

folders_to_process = [
    "CC_Bangles",
    "CC_Earrings",
    "CC_Ladies_Rings",
    "CC_Necklace",
    "Diamond_Necklace",
    "Gold_Necklace"
]

# --- INITIALIZE AI MODEL (Prioritizing your RTX 3060 GPU) ---
print("Loading BiRefNet Model on GPU...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def process_images():
    for folder_name in folders_to_process:
        input_folder = os.path.join(input_base_dir, folder_name)
        output_folder = os.path.join(output_base_dir, folder_name)
        
        # 1. Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nProcessing Folder: {folder_name}")
        
        # 2. Gather ALL images in the folder
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp')
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
            image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
            
        image_paths = sorted(list(set(image_paths)))
        
        if not image_paths:
            print(f"No images found in {input_folder}")
            continue

        print(f"Found {len(image_paths)} images in {folder_name}. Starting extraction...")

        # 3. Process ALL images (No 10-image limit)
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            
            # CHANGED: Now saving as .png instead of .jpg
            clean_img_name = f"{os.path.splitext(img_name)[0]}_clean.png"
            output_path = os.path.join(output_folder, clean_img_name)
            
            # Skip if already processed (so you can pause and resume later without starting over)
            if os.path.exists(output_path):
                print(f" [SKIPPED] Already exists -> {clean_img_name}")
                continue
            
            try:
                # Open original image
                with Image.open(img_path) as orig_img:
                    orig_img = orig_img.convert("RGBA")
                    
                    # AI EXTRACTION: Flawlessly cut out the jewelry
                    extracted_img = remove(orig_img, session=session)
                    
                    # RE-COMPOSITING: Create a pure black canvas (#000000)
                    black_canvas = Image.new("RGBA", extracted_img.size, (0, 0, 0, 255))
                    
                    # Paste the isolated jewelry onto the black canvas
                    black_canvas.paste(extracted_img, (0, 0), extracted_img)
                    
                    # Convert to RGB (standard for opaque images) and save as Lossless PNG
                    final_img = black_canvas.convert("RGB")
                    final_img.save(output_path, "PNG")
                    
                print(f" [SUCCESS] Saved -> {clean_img_name}")
                
            except Exception as e:
                print(f" [ERROR] Failed on {img_name}: {str(e)}")

if __name__ == "__main__":
    process_images()
    print("\n🎉 Full dataset processing complete! All files saved as lossless PNGs.")