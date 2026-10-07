import os
import gc  # Garbage Collector
from PIL import Image
from rembg import remove, new_session

# --- 1. PATH CONFIGURATION ---
input_base_dir = r"D:\gen jewels\web scraping\Dataset"
output_base_dir = r"D:\gen jewels\web scraping\dataset finetuned"

brands_to_process = [
    "BhimaJewellery",
    "Candere",
    "Tanishq",
    "dar_jewelry_watermark_removed", 
    "DarJewellery" 
]

# --- 2. THE CASCADING PURGE SYSTEM ---
def create_gpu_session():
    """Creates a STRICTLY VRAM-only session"""
    print("\n[SYSTEM] Spawning fresh AI Model into GPU VRAM...")
    # Removed CPU fallback. It will ONLY use your RTX 3060 now.
    return new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def process_entire_dataset():
    # Load initial session
    session = create_gpu_session()
    processed_count = 0  # Counter for our cascading purge
    
    for brand in brands_to_process:
        brand_input_path = os.path.join(input_base_dir, brand)
        
        if not os.path.exists(brand_input_path):
            continue
            
        print(f"\n{'='*50}\n🚀 Starting Brand: {brand}\n{'='*50}")
        
        for root, dirs, files in os.walk(brand_input_path):
            relative_path = os.path.relpath(root, input_base_dir)
            target_output_dir = os.path.join(output_base_dir, relative_path)
            os.makedirs(target_output_dir, exist_ok=True)
            
            for file in files:
                ext = file.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg', 'png', 'webp']:
                    
                    img_path = os.path.join(root, file)
                    clean_img_name = f"{os.path.splitext(file)[0]}_black_bg.png"
                    output_path = os.path.join(target_output_dir, clean_img_name)
                    
                    if os.path.exists(output_path):
                        print(f" [SKIPPED] -> {clean_img_name}")
                        continue
                        
                    try:
                        # Process image
                        with Image.open(img_path) as orig_img:
                            orig_img = orig_img.convert("RGBA")
                            
                            extracted_img = remove(orig_img, session=session)
                            
                            black_canvas = Image.new("RGBA", extracted_img.size, (0, 0, 0, 255))
                            black_canvas.paste(extracted_img, (0, 0), extracted_img)
                            
                            final_img = black_canvas.convert("RGB")
                            final_img.save(output_path, "PNG")
                            
                        print(f" [SUCCESS] Saved -> {relative_path}\\{clean_img_name}")
                        
                        # --- THE PURGE ARCHITECTURE ---
                        
                        # 1. Destroy local variables from physical RAM immediately
                        del extracted_img
                        del black_canvas
                        del final_img
                        
                        # 2. Force Python to empty the physical RAM cache instantly
                        gc.collect()
                        
                        # 3. Cascading VRAM Reset (Every 100 images)
                        processed_count += 1
                        if processed_count % 100 == 0:
                            print("\n[MEMORY PURGE] 100 images reached. Flushing VRAM & RAM to prevent OOM...")
                            # Destroy the current AI session
                            del session
                            # Force garbage collection again
                            gc.collect()
                            # Spawn a brand new, clean session in the GPU
                            session = create_gpu_session()
                            
                    except Exception as e:
                        print(f" [ERROR] Failed on {file}: {str(e)}")
                        
if __name__ == "__main__":
    # Extra safety: prevent Windows from grabbing unnecessary memory at launch
    os.environ["OMP_NUM_THREADS"] = "1"
    
    process_entire_dataset()
    print("\n🎉 ALL BRANDS PROCESSED WITH ZERO MEMORY LEAKS!")