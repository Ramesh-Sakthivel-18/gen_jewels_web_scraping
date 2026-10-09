import os
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from pathlib import Path

# --- Specific Local Paths ---
MODEL_DIR = r"D:\gen jewels\web scraping\models\InternVL2-8B"
DATASET_DIR = r"D:\gen jewels\web scraping\dataset_1024_framing"

# Automatically create the CSV inside the dataset folder
OUTPUT_CSV = os.path.join(DATASET_DIR, "jewelry_captions_raw.csv")

# --- The Perfected Prompt ---
SYSTEM_PROMPT = (
    "You are a Master Fine Jewelry Appraiser & Gemologist. Analyze this jewelry photograph with microscopic precision.\n"
    "Output your analysis strictly in the format below without any conversational filler.\n\n"
    
    "RULES (Mandatory Attribute Extraction):\n"
    "- Category: (e.g., Jhumka, Choker Necklace, Rigid Bangle, Solitaire Ring, Drop Earring)\n"
    "- Metal Tone & Karat: (e.g., 22k Antique Matte Yellow Gold, 18k High-Polish Rose Gold, Platinum)\n"
    "- Craftsmanship Style: (e.g., South Indian Temple Nakshi, Polki Jadau, Kundan, Modern Pavé Diamond)\n"
    "- Primary Motif / Carving: (e.g., Seated Goddess Lakshmi, Peacock, Filigree Floral, Geometric Halo)\n"
    "- Gemstone Types & Cuts: (e.g., Cabochon rubies, brilliant-cut diamonds, natural pearls, faceted emeralds)\n"
    "- Stone Setting & Prongs: (e.g., Bezel-set, micro-pavé, sharp 4-prong claw, flush-set, closed-back)\n"
    "- Hanging / Drop Accents: (e.g., Golden ghungroo bells, dangling drop pearls, beaded tassel, none)\n"
    "- Background & Studio Setup: (e.g., Solid black studio background, clean macro studio lighting)\n\n"
    
    "DEEP VISUAL OBSERVATION:\n"
    "Describe the physical architecture from the center outward in 3-4 dense, factual sentences. "
    "Focus on: metal thickness and filigree wirework, how tightly the stones are packed (gaps vs seamless), "
    "exact prong sharpness holding the stones, surface texture of the gold (matte, hammered, or mirror polish), "
    "and symmetry of the piece on its solid black background."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0).to(torch.float16).cuda()
    return pixel_values

def main():
    print("🚀 [1/4] Loading InternVL2-8B strictly to GPU 0...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
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
    )
    
    print("📁 [2/4] Scanning Dataset Directory for images...")
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(Path(DATASET_DIR).rglob(ext))
    
    total_images = len(image_paths)
    print(f"✅ Found {total_images} images. Starting processing...\n")
    print("=" * 80)
    
    results = []
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    for idx, img_path in enumerate(image_paths):
        img_str_path = str(img_path)
        
        # --- NEW FOLDER EXTRACTION LOGIC ---
        sub_folder = img_path.parent.name         # e.g., 'Bangles'
        main_folder = img_path.parent.parent.name # e.g., 'Candere' or 'BhimaJewellery'
        
        print(f"🔍 [{idx + 1}/{total_images}] Brand: '{main_folder}' | Category: '{sub_folder}' | Image: '{img_path.name}'")
        
        try:
            pixel_values = load_image(img_str_path)
            
            with torch.no_grad():
                response = model.chat(
                    tokenizer, 
                    pixel_values, 
                    SYSTEM_PROMPT, 
                    generation_config, 
                    history=None, 
                    return_history=False
                )
                
            # Parse the string into columns
            if "DEEP VISUAL OBSERVATION" in response:
                parts = response.split("DEEP VISUAL OBSERVATION")
                rules_text = parts[0].replace("RULES (Mandatory Attribute Extraction):", "").strip()
                deep_obs_text = parts[1].replace(":", "", 1).strip()
            else:
                rules_text = response.strip()
                deep_obs_text = "Parsing error: Deep Visual Observation section missing."

            # Append the data with Brand and Category columns
            results.append({
                "Image_Path": img_str_path,
                "Brand_Name": main_folder,
                "Category": sub_folder,
                "Rules": rules_text,
                "Deep_Visual_Observation": deep_obs_text
            })
            
            # Instant save after every single generation
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    print(f"\n🎉 ALL DONE! 100% complete. CSV successfully saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()