import os
import re
import csv
import warnings
import pandas as pd
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning)

# --- Specific Local Paths ---
MODEL_DIR = r"D:\gen jewels\web scraping\models\Qwen2-VL-2B-Instruct"
DATASET_DIR = r"D:\gen jewels\web scraping\dataset_1024_framing"
OUTPUT_CSV = os.path.join(DATASET_DIR, "qwen2_full_dataset_captions.csv")

# --- ROBUST PROMPT ---
QWEN_MASTER_PROMPT = """You are a structural jewelry vision parser inspecting fine jewelry on a solid black background.
Do not write conversational filler. Start directly with item 1. Keep each item under 12 words.

Verified Category: {category}

RULES:
- If edges are cut off by the frame with no main pendant or clasp, state "Macro chain segment". If fully visible, describe the full silhouette.
- Never say 'gold-plated' or 'silver'. Use fine metals: 22k antique gold, rose gold, white gold, or platinum.
- Never use uncertainty words like 'possibly', 'maybe', or 'or'.
- Never use negative words like 'no danglers' or 'none'. Omit absent features.

OUTPUT STRICTLY WITH THESE 4 LABELS:

1. Outer Silhouette:
2. Central Motif:
3. Metal and Main Stones:
4. Mechanical Articulation:"""

# --- PARSER UTILS ---
def is_negative_or_empty(text: str) -> bool:
    """Detects if a phrase is negative, absent, or empty."""
    t = text.lower().strip()
    if not t or t == "n/a":
        return True
    negative_triggers = ["none", "no ", "not ", "n/a", "without", "nil", "zero", "single piece"]
    for trigger in negative_triggers:
        if t.startswith(trigger) or f" {trigger}" in t:
            return True
    return False

def clean_extracted_value(raw_val: str) -> str:
    """Strips markdown asterisks, residual labels, colons, and leading dashes."""
    val = re.sub(r'^\s*(?:\*{0,2})?[A-Za-z\s&]+(?:\*{0,2})?[:\-]\s*', '', raw_val)
    val = val.replace('\n', ' ').replace('*', '').strip()
    val = re.sub(r'^[-\s:]+', '', val)
    return val.strip()

def parse_qwen_output(text: str, category: str):
    p1 = re.search(r'(?:^|\n)\s*\*{0,2}1[\.\)\:]\s*(.*?)(?=(?:^|\n)\s*\*{0,2}2[\.\)\:]|\Z)', text, re.DOTALL)
    p2 = re.search(r'(?:^|\n)\s*\*{0,2}2[\.\)\:]\s*(.*?)(?=(?:^|\n)\s*\*{0,2}3[\.\)\:]|\Z)', text, re.DOTALL)
    p3 = re.search(r'(?:^|\n)\s*\*{0,2}3[\.\)\:]\s*(.*?)(?=(?:^|\n)\s*\*{0,2}4[\.\)\:]|\Z)', text, re.DOTALL)
    p4 = re.search(r'(?:^|\n)\s*\*{0,2}4[\.\)\:]\s*(.*?)(?=\Z)', text, re.DOTALL)

    structure = clean_extracted_value(p1.group(1)) if p1 else "N/A"
    motif = clean_extracted_value(p2.group(1)) if p2 else "N/A"
    materials = clean_extracted_value(p3.group(1)) if p3 else "N/A"
    mechanics = clean_extracted_value(p4.group(1)) if p4 else "N/A"

    # Assemble clean caption (drops negative/empty parts automatically)
    clean_parts = [category]
    for part in [structure, motif, materials, mechanics]:
        if part and not is_negative_or_empty(part):
            clean_parts.append(part.rstrip('.,;'))

    return {
        "Outer_Structure": structure,
        "Central_Motif": motif,
        "Metal_and_Stones": materials,
        "Mechanics": mechanics,
        "SDXL_Caption": f"gj_temple_jewel, {', '.join(clean_parts)}, solid black studio background",
        "Full_Raw_Output": text.strip()
    }

def main():
    print("🚀 [1/3] Loading Qwen2-VL-2B-Instruct Native (Float16) onto GPU 0...")
    
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="cuda:0", 
    ).eval()
    
    # Suppress generation sampling warnings cleanly
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    
    print("\n📁 [2/3] Scanning Dataset & Checking Progress...")
    
    # Generate list of all images
    all_image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        all_image_paths.extend(Path(DATASET_DIR).rglob(ext))
    
    # CRASH-PROOF RESUME LOGIC
    processed_images = set()
    file_exists = os.path.isfile(OUTPUT_CSV)
    
    if file_exists:
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            if "Image_Path" in df_existing.columns:
                processed_images = set(df_existing["Image_Path"].tolist())
                print(f"🔄 Resume Active: Skipping {len(processed_images)} previously processed images.")
        except Exception as e:
            print(f"⚠️ Warning reading CSV for resume: {e}")
            
    images_to_process = [p for p in all_image_paths if str(p) not in processed_images]
    total_remaining = len(images_to_process)
    
    if total_remaining == 0:
        print("✅ All images have already been processed! CSV is fully populated.")
        return
        
    print(f"✅ Found {total_remaining} images ready to process.\n")
    print("=" * 80)
    
    # EXACT 9 CSV COLUMNS
    csv_headers = [
        "Image_Path", "Brand_Name", "Category", 
        "Outer_Structure", "Central_Motif", "Metal_and_Stones", 
        "Mechanics", "SDXL_Caption", "Full_Raw_Output"
    ]
    
    # If the CSV does not exist yet, create it and write ONLY the header row first
    if not file_exists:
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
            writer.writeheader()
    
    # Process images one by one
    for idx, img_path in enumerate(images_to_process):
        img_str_path = str(img_path)
        
        category_raw = img_path.parent.name
        brand_name = img_path.parent.parent.name
        clean_category = category_raw.replace("CC_", "").replace("CC ", "").replace("_", " ").strip()
        
        active_prompt = QWEN_MASTER_PROMPT.format(category=clean_category)
        
        print(f"\n📸 Image [{idx + 1}/{total_remaining}]")
        print(f"   Brand: {brand_name} | Category: {clean_category} | File: {img_path.name}")
        print("-" * 65)
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_str_path},
                        {"type": "text", "text": active_prompt},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda:0")
            
            # Lightning fast generation limited to 250 tokens
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=250, 
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            raw_output = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Run Regex parser
            parsed_data = parse_qwen_output(raw_output, clean_category)
            
            # Map exactly to 9 Column schema
            csv_row = {
                "Image_Path": img_str_path,
                "Brand_Name": brand_name,
                "Category": clean_category,
                "Outer_Structure": parsed_data["Outer_Structure"],
                "Central_Motif": parsed_data["Central_Motif"],
                "Metal_and_Stones": parsed_data["Metal_and_Stones"],
                "Mechanics": parsed_data["Mechanics"],
                "SDXL_Caption": parsed_data["SDXL_Caption"],
                "Full_Raw_Output": parsed_data["Full_Raw_Output"]
            }
            
            # --- ABSOLUTE SAFE SAVE (Open, Append, Close for every single image) ---
            with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
                writer.writerow(csv_row)
            
            # Terminal display
            print(f"   ➤ Final SDXL: {parsed_data['SDXL_Caption']}")
            print(f"   💾 Securely written to CSV.")
            
        except Exception as e:
            print(f"❌ Error processing image {img_path.name}: {e}")
            
        print("=" * 80)

    print(f"\n🎉 FULL RUN COMPLETE! Check your dataset folder for the final '{Path(OUTPUT_CSV).name}' file.")

if __name__ == "__main__":
    main()