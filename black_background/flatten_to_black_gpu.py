import os
from PIL import Image
from rembg import remove, new_session

# --- 1. PATH CONFIGURATION ---
# Your main dataset folder containing all the brands
input_base_dir = r"D:\gen jewels\web scraping\Dataset"

# The NEW folder outside the dataset for your final SDXL images
output_base_dir = r"D:\gen jewels\web scraping\dataset finetuned"

# The specific brand folders you want to process
brands_to_process = [
    "BhimaJewellery",
    "Candere",
    "Tanishq",
    # (If you want to re-process the Dar ones into this new folder too, you can include them here)
    "dar_jewelry_watermark_removed", 
    "DarJewellery" 
]

# --- 2. INITIALIZE AI MODEL (Prioritizing RTX 3060 GPU) ---
print("Loading BiRefNet Model on GPU...")
# The 'CUDAExecutionProvider' guarantees it uses the cuDNN fuel we set up earlier!
session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def process_entire_dataset():
    # Loop through each brand folder
    for brand in brands_to_process:
        brand_input_path = os.path.join(input_base_dir, brand)
        
        # Check if the brand folder actually exists before processing
        if not os.path.exists(brand_input_path):
            continue
            
        print(f"\n{'='*50}\n🚀 Starting Brand: {brand}\n{'='*50}")
        
        # os.walk automatically discovers all subfolders (Bangles, Earrings, Pendants, etc.)
        for root, dirs, files in os.walk(brand_input_path):
            
            # Find the relative path (e.g., "Tanishq\Bangles")
            relative_path = os.path.relpath(root, input_base_dir)
            
            # Create the exact same folder structure in the new 'dataset finetuned' directory
            target_output_dir = os.path.join(output_base_dir, relative_path)
            os.makedirs(target_output_dir, exist_ok=True)
            
            # Process every file inside this specific subfolder
            for file in files:
                # Make sure it's an image file
                ext = file.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg', 'png', 'webp']:
                    
                    img_path = os.path.join(root, file)
                    
                    # Create the new filename (forcing .png)
                    clean_img_name = f"{os.path.splitext(file)[0]}_black_bg.png"
                    output_path = os.path.join(target_output_dir, clean_img_name)
                    
                    # Smart Resume: Skip if this image is already processed
                    if os.path.exists(output_path):
                        print(f" [SKIPPED] Already exists -> {relative_path}\\{clean_img_name}")
                        continue
                        
                    try:
                        # Open original image
                        with Image.open(img_path) as orig_img:
                            orig_img = orig_img.convert("RGBA")
                            
                            # AI EXTRACTION (Runs on GPU)
                            extracted_img = remove(orig_img, session=session)
                            
                            # RE-COMPOSITING: Pillow creates the pure black background
                            black_canvas = Image.new("RGBA", extracted_img.size, (0, 0, 0, 255))
                            black_canvas.paste(extracted_img, (0, 0), extracted_img)
                            
                            # Convert to RGB (Opaque) and Save as Lossless PNG
                            final_img = black_canvas.convert("RGB")
                            final_img.save(output_path, "PNG")
                            
                        print(f" [SUCCESS] Saved -> {relative_path}\\{clean_img_name}")
                        
                    except Exception as e:
                        print(f" [ERROR] Failed on {file}: {str(e)}")

if __name__ == "__main__":
    process_entire_dataset()
    print("\n🎉 ALL BRANDS PROCESSED! Check the new 'dataset finetuned' folder.")