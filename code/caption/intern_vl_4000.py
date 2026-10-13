# this code is run in PICCC-8 system and and the files are linked to that system 


import os
import sys
import re
import torch
import warnings
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# ==============================================================================
# --- CLEAN TERMINAL ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ==============================================================================

# --- Specific Local Paths ---
MODEL_DIR = r"D:\ramesh\jewelry ipa\training\model\InternVL2-8B"
DATASET_DIR = r"D:\ramesh\jewelry ipa\training\dataset_1024_framing"

# This will automatically save the fully segmented dataset results here
OUTPUT_CSV = os.path.join(DATASET_DIR, "internvl2_segmented_dataset.csv")

# --- THE STRICT APPRAISER PROMPT ---
INTERNVL_PROMPT = """You are a Senior Gemologist and Master Indian Fine Jewelry Appraiser. Inspect this fine jewelry photograph on a solid black background with forensic optical accuracy.

Verified Catalog Category: {category}

STRICT FORENSIC RULES:
1. TRUNCATED CROPS vs. COMPLETE PIECES:
   - Inspect the outer edges of the jewelry. If the jewelry is sliced or truncated by the image borders (e.g. an isolated section of a chain or bracelet link with no clasp or main pendant), you MUST classify it as: "Macro structural detail of {category} segment". NEVER hallucinate a central pendant on a cut-off chain link.
   - If the complete silhouette with its focal centerpiece is visible, classify as: "Full {category} silhouette".
2. EXPLICIT GEMSTONE & COLOR DISCOVERY:
   - Scrutinize every stone for its specific hue and cut:
     * Lavender / violet / light purple stone = "light purple pear-cut gemstone (tanzanite)"
     * Vivid green stone = "green pear-cut emerald"
     * Clear sparkling facets = "brilliant-cut diamond pavé"
   - Look closely at earrings: identify spiral, twisted, or flowing drop contours.
3. STRICT POSITIVE STATEMENTS:
   - Never say 'no danglers' or 'without stones'. Describe only visible elements.
4. ZERO UNCERTAINTY:
   - Banned words: 'possibly', 'maybe', 'likely', 'or'. Be authoritative.
5. NO LABEL LEAKAGE IN SDXL CAPTION:
   - The Final SDXL Caption must be pure, continuous comma-separated descriptive text. 
   - NEVER output key-value labels like 'Central motif:', 'Metal tone:', or 'Stone settings:' inside the caption.

OUTPUT FORMAT:

1. Overall Structure & Form: [Full silhouette vs. Macro structural detail segment]
2. Main Focal Design: [Center motif or repeating chain pattern, specifying all colored stones]
3. Materials, Metal & Finish: [Rose gold, yellow gold, or white gold, high-polish]
4. Micro Craftsmanship & Settings: [Claw prongs, pavé cluster, or bezels]
5. Micro Mechanics & Articulation: [Articulated links or solid rigid frame]
6. Overall Aesthetic & Environment: Sharp macro focus, solid black studio background

Final SDXL Caption: gj_temple_jewel, {category}, [Structure or Macro], [Motif and colored stones], [Metal], [Settings], [Mechanics], solid black studio background"""

# ==============================================================================
# --- FORENSIC TEXT PARSER ---
# Extracts content strictly between boundaries and normalizes whitespace
# ==============================================================================
def parse_internvl_output(raw_text):
    parsed = {
        "Structure_and_Form": "",
        "Focal_Design": "",
        "Materials_Metal_Finish": "",
        "Craftsmanship_and_Settings": "",
        "Structural_Mechanics": "",
        "Aesthetic_and_Environment": "",
        "SDXL_Caption": "",
        "Full_Raw_Output": raw_text.strip()
    }

    # Helper function to extract text between two regex patterns
    def extract_segment(pattern, text):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # Replace internal newlines with spaces and strip extra whitespace
            return match.group(1).replace('\n', ' ').strip()
        return ""

    # Signpost Regex Boundaries (Made slightly flexible to handle missing colons)
    p1 = r"1\. Overall Structure & Form:?(.*?)2\. Main Focal Design:?"
    p2 = r"2\. Main Focal Design:?(.*?)3\. Materials, Metal & Finish:?"
    p3 = r"3\. Materials, Metal & Finish:?(.*?)4\. Micro Craftsmanship & Settings:?"
    p4 = r"4\. Micro Craftsmanship & Settings:?(.*?)5\. Micro Mechanics & Articulation:?"
    p5 = r"5\. Micro Mechanics & Articulation:?(.*?)6\. Overall Aesthetic & Environment:?"
    p6 = r"6\. Overall Aesthetic & Environment:?(.*?)Final SDXL Caption:?"

    parsed["Structure_and_Form"] = extract_segment(p1, raw_text)
    parsed["Focal_Design"] = extract_segment(p2, raw_text)
    parsed["Materials_Metal_Finish"] = extract_segment(p3, raw_text)
    parsed["Craftsmanship_and_Settings"] = extract_segment(p4, raw_text)
    parsed["Structural_Mechanics"] = extract_segment(p5, raw_text)
    parsed["Aesthetic_and_Environment"] = extract_segment(p6, raw_text)

    # SDXL Caption Extraction (Triggers exactly on 'gj_temple_jewel')
    sdxl_match = re.search(r"(gj_temple_jewel.*)", raw_text, re.DOTALL | re.IGNORECASE)
    if sdxl_match:
        parsed["SDXL_Caption"] = sdxl_match.group(1).replace('\n', ' ').strip()
    else:
        # Fallback if model missed the trigger word but still wrote the caption
        fallback = extract_segment(r"Final SDXL Caption:?(.*)", raw_text)
        parsed["SDXL_Caption"] = fallback

    return parsed

# ==============================================================================
# --- HIGH-RES DYNAMIC IMAGE PATCHING (InternVL2 Official Logic) ---
# ==============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
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
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num
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

def load_image(image_file, input_size=448, max_num=4):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(patch) for patch in images]
    pixel_values = torch.stack(pixel_values).to(torch.float16).cuda()
    return pixel_values
# ==============================================================================

def main():
    print("🚀 [1/4] Loading InternVL2-8B strictly to GPU 0 (4-bit Quantized, High-Res)...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False
    )
    
    assert hasattr(tokenizer, 'convert_tokens_to_ids'), \
        f"Tokenizer load FAILED -- got {type(tokenizer)} instead of a real tokenizer."

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

    print("📁 [2/4] Scanning Dataset Directory for images...")
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(Path(DATASET_DIR).rglob(ext))
    
    total_images = len(image_paths)
    print(f"✅ Found {total_images} images. Starting bulk processing...\n")
    print("=" * 80)

    generation_config = dict(max_new_tokens=1024, do_sample=False)
    results = []

    for idx, img_path in enumerate(image_paths):
        img_str_path = str(img_path)

        # Folder Logic
        sub_folder = img_path.parent.name         # e.g., 'Diamond_Necklace'
        main_folder = img_path.parent.parent.name # e.g., 'dar_jewelry'
        clean_category = sub_folder.replace("_", " ")

        print(f"🔍 [{idx + 1}/{total_images}] Brand: '{main_folder}' | Category: '{clean_category}' | Image: '{img_path.name}'")

        raw_prompt = INTERNVL_PROMPT.format(category=clean_category)
        safe_prompt = "<image>\n" + raw_prompt

        try:
            pixel_values = load_image(img_str_path, max_num=4)

            with torch.no_grad():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    safe_prompt,
                    generation_config,
                    history=None,
                    return_history=False
                )
            
            # --- PARSE THE RAW TEXT INTO SEGMENTS ---
            parsed_data = parse_internvl_output(response)
            
            # Show a brief preview of the extracted SDXL prompt
            print(f"  ➤ Extracted SDXL: {parsed_data['SDXL_Caption'][:80]}...")

            # --- BUILD ROW FOR EXACT 11-COLUMN SCHEMA ---
            results.append({
                "Image_Path": img_str_path,
                "Brand_Name": main_folder,
                "Category": sub_folder,
                "Structure_and_Form": parsed_data["Structure_and_Form"],
                "Focal_Design": parsed_data["Focal_Design"],
                "Materials_Metal_Finish": parsed_data["Materials_Metal_Finish"],
                "Craftsmanship_and_Settings": parsed_data["Craftsmanship_and_Settings"],
                "Structural_Mechanics": parsed_data["Structural_Mechanics"],
                "Aesthetic_and_Environment": parsed_data["Aesthetic_and_Environment"],
                "SDXL_Caption": parsed_data["SDXL_Caption"],
                "Full_Raw_Output": parsed_data["Full_Raw_Output"]
            })
            
            # Write safely using Pandas (handles comma escaping automatically)
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, quoting=1) # 1 = csv.QUOTE_ALL

        except Exception as e:
            import traceback
            print(f"❌ [ERROR] Processing {img_path.name}: {e}")

    print(f"\n🎉 ALL DONE! 100% complete. Clean 11-Column CSV successfully saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()