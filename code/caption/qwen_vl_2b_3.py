import os
import re
import warnings
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning)

# --- Specific Local Paths ---
MODEL_DIR = r"D:\gen jewels\web scraping\models\Qwen2-VL-2B-Instruct"

TEST_IMAGES = [
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Diamond_Necklace\Diamond_Necklace_Prod2_Angle2_1200x1200_a8a9d6_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Diamond_Necklace\Diamond_Necklace_Prod1_Angle2_1200x1200_cbb949_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\CC_Earrings\CC_Earrings_Prod2_Angle2_1200x1200_86368f_clean_black_bg.png"
]

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
    # Strip any header label that was captured before the colon
    val = re.sub(r'^\s*(?:\*{0,2})?[A-Za-z\s&]+(?:\*{0,2})?[:\-]\s*', '', raw_val)
    # Remove asterisks and clean newlines
    val = val.replace('\n', ' ').replace('*', '').strip()
    val = re.sub(r'^[-\s:]+', '', val)
    return val.strip()

# --- NUMBER-ANCHOR PARSER (Indestructible against asterisks and format drift) ---
def parse_qwen_output(text: str, category: str):
    # Matches the numbers 1, 2, 3, 4 regardless of markdown asterisks (**, *, none)
    p1 = re.search(r'(?:^|\n)\s*\*{0,2}1[\.\)\:]\s*(.*?)(?=(?:^|\n)\s*\*{0,2}2[\.\)\:]|\Z)', text, re.DOTALL)
    p2 = re.search(r'(?:^|\n)\s*\*{0,2}2[\.\)\:]\s*(.*?)(?=(?:^|\n)\s*\*{0,2}3[\.\)\:]|\Z)', text, re.DOTALL)
    p3 = re.search(r'(?:^|\n)\s*\*{0,2}3[\.\)\:]\s*(.*?)(?=(?:^|\n)\s*\*{0,2}4[\.\)\:]|\Z)', text, re.DOTALL)
    p4 = re.search(r'(?:^|\n)\s*\*{0,2}4[\.\)\:]\s*(.*?)(?=\Z)', text, re.DOTALL)

    structure = clean_extracted_value(p1.group(1)) if p1 else "N/A"
    motif = clean_extracted_value(p2.group(1)) if p2 else "N/A"
    materials = clean_extracted_value(p3.group(1)) if p3 else "N/A"
    mechanics = clean_extracted_value(p4.group(1)) if p4 else "N/A"

    # Assemble caption while filtering out negatives ('none', 'single piece', etc.)
    clean_parts = [category]
    for part in [structure, motif, materials, mechanics]:
        if part and not is_negative_or_empty(part):
            clean_parts.append(part.rstrip('.,;'))

    parsed = {
        "Structure_and_Form": structure,
        "Focal_Design": motif,
        "Materials_Metal_Finish": materials,
        "Craftsmanship_and_Settings": "Detailed fine settings",
        "Structural_Mechanics": mechanics,
        "Aesthetic_and_Environment": "Sharp macro focus, solid black studio background",
        "Full_Raw_Output": text.strip(),
        "SDXL_Caption": f"gj_temple_jewel, {', '.join(clean_parts)}, solid black studio background"
    }

    return parsed

def main():
    print("🚀 Loading Qwen2-VL-2B-Instruct Native (Float16) onto GPU 0...")
    
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
    
    print("\n✅ Model Loaded. Starting Inference...\n")
    print("=" * 80)
    
    for idx, img_path_str in enumerate(TEST_IMAGES):
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"❌ Cannot find image: {img_path_str}")
            continue
            
        category_raw = img_path.parent.name
        brand_name = img_path.parent.parent.name
        clean_category = category_raw.replace("CC_", "").replace("CC ", "").replace("_", " ").strip()
        
        active_prompt = QWEN_MASTER_PROMPT.format(category=clean_category)
        
        print(f"\n📸 Image [{idx + 1}/3]:")
        print(f"   Brand: {brand_name} | Category: {clean_category} | File: {img_path.name}")
        print("-" * 65)
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path_str},
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
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=300, 
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            raw_output = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # --- Pass through NUMBER-ANCHOR Parser ---
            parsed_data = parse_qwen_output(raw_output, clean_category)
            
            # --- Terminal Display ---
            print("🔬 4-LABEL FACTUAL OUTPUT:")
            print(f"   ➤ Outer Silhouette: {parsed_data['Structure_and_Form']}")
            print(f"   ➤ Central Motif:    {parsed_data['Focal_Design']}")
            print(f"   ➤ Metal & Stones:   {parsed_data['Materials_Metal_Finish']}")
            print(f"   ➤ Mechanics:        {parsed_data['Structural_Mechanics']}")
                    
            print("\n🔥 ASSEMBLED SDXL CAPTION (Negative Filtered):")
            print(f"   {parsed_data['SDXL_Caption']}")
            
        except Exception as e:
            print(f"❌ Error processing image {img_path.name}: {e}")
            
        print("=" * 80)

if __name__ == "__main__":
    main()