import os
import re
import csv
import torch
from PIL import Image
from pathlib import Path
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# --- SAFEGUARD HOTFIX FOR TORCH/TRANSFORMERS ---
if not hasattr(torch, 'accelerator'):
    class DummyAccelerator:
        def current_accelerator(self): return None
    torch.accelerator = DummyAccelerator()
# -----------------------------------------------

# --- Paths ---
MODEL_DIR = r"D:\gen jewels\web scraping\models\Llama-3.2-11B-Vision-Instruct"
DATASET_DIR = r"D:\gen jewels\web scraping\dataset_1024_framing"
OUTPUT_CSV = os.path.join(DATASET_DIR, "jewelry_master_llama_3.2_11b_captions.csv")

# --- The Perfected Master Prompt ---
MASTER_PROMPT = """You are the Chief Senior Gemologist and Master Indian Jewelry Appraiser. Analyze this jewelry photograph on a solid black background. 

The user will provide a "Verified Category". You MUST use this category to ground your analysis.

CRITICAL RULES:
1. STRICT PHYSICAL ACCURACY (NO HALLUCINATIONS): Describe ONLY exactly what you see. DO NOT blindly copy the examples provided in the brackets below. They are just examples.
2. FULL PIECE vs. MACRO SEGMENT: 
   - If you see the complete piece (full necklace with pendant, complete pair of earrings, full bangle), describe its complete silhouette.
   - If the edges of the jewelry are physically cut off or sliced by the frame (e.g., only a chain section, a single branch, or no focal pendant visible), you MUST classify it as: "Macro structural detail of [Verified Category] chain segment". NEVER describe a cut-off segment as a full necklace or full earring.
3. CONFIDENT GEMOLOGICAL DECLARATIONS (ZERO UNCERTAINTY): 
   - BANNED WORDS: Never use 'possibly', 'maybe', 'appears to be', or indecisive 'or' (e.g., NEVER say 'ruby or red gemstone' or 'possibly diamonds'). 
   - State stones with absolute authority: identify clear faceted stones as diamonds, deep red stones as rubies, and green stones as emeralds.
4. MANDATORY FOCAL GEMSTONE INCLUSION: If a distinct contrasting colored stone is visible (e.g., pear-cut red ruby, emerald cabochon, or pearl drop), you MUST explicitly identify its cut, color, and placement in BOTH Part 1 and Part 2. Never omit the hero gemstone.
5. FINE JEWELRY METALS ONLY (NO SILVER): This is an authentic luxury fine jewelry catalog (18k/22k yellow gold, rose gold, white gold, and diamonds). NEVER classify white metallic shine or diamond clusters as 'silver' or 'costume'.
6. STRICT POSITIVE STATEMENTS & NO UNDERSCORES: Never say 'no stones' or 'no danglers'. If a feature is absent, simply omit it. Use natural English words separated by spaces and commas. NEVER use underscores ('_') anywhere except in the trigger token 'gj_temple_jewel'.
7. MANDATORY TWO-PART EXECUTION: You MUST generate BOTH Part 1 and Part 2 in full. Never stop after Part 1.

PART 1: FORENSIC AUDIT
1. Overall Structure & Form: State the Verified Category. Describe the exact physical shape. (State 'Macro structural detail of [Verified Category] segment' if the piece is cropped/cut off by the canvas edges; otherwise describe the full piece silhouette).
2. Main Focal Design: Describe the central motif or repeating pattern (e.g., teardrop pendant, floral cluster, repeating leafy chain motif).
3. Materials, Metal & Finish: Describe the exact metal tone (yellow gold, rose gold, white gold) and surface finish (high-polish, matte).
4. Micro Craftsmanship & Settings: State the exact gemstones (diamonds, rubies, emeralds) and how they are mounted (claw prongs, bezels, or pavé).
5. Structural Mechanics: Describe construction. State if it is a completely solid/rigid piece (like a stud earring) OR if it has flexible linking joints (like an articulated chain).
6. Overall Aesthetic & Environment: Describe the studio macro lighting and solid black background.

PART 2: FINAL SDXL CAPTION
Synthesize the audit into a single high-density prompt (Strictly 35 to 45 words max).
Format exactly as:
gj_temple_jewel, [Verified Category], [Structure or Macro Segment], [Motif & Focal Gemstones], [Materials], [Craftsmanship & Settings], [Mechanics], solid black studio background"""


def parse_llm_output(text):
    """
    4-Stage Barrier & Anchor Parsing System.
    Guarantees zero column mismatch regardless of LLM newlines or formatting.
    """
    data = {
        "Structure_and_Form": "",
        "Focal_Design": "",
        "Materials_Metal_Finish": "",
        "Craftsmanship_and_Settings": "",
        "Structural_Mechanics": "",
        "Aesthetic_and_Environment": "",
        "SDXL_Caption": "",
        "Raw_Output": text
    }
    
    # STAGE 1: The Master Split (Isolate Zone A and Zone B)
    if "PART 2:" in text:
        zone_a, zone_b = text.split("PART 2:", 1)
    else:
        zone_a, zone_b = text, ""

    # Pre-clean markdown bolding so boundary numbers (1. 2. 3.) are never obstructed
    zone_a = zone_a.replace("**", "")

    # Helper function to extract text between anchors and clean newlines
    def extract_between(text, start_marker, end_marker=None):
        if end_marker:
            pattern = rf"{start_marker}\.(.*?)(?={end_marker}\.)"
        else:
            pattern = rf"{start_marker}\.(.*)"
            
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1)
            # Remove the header prefix (e.g., "Overall Structure & Form: ")
            extracted = re.sub(r"^[A-Za-z &]+:\s*", "", extracted.strip())
            # Convert all newlines into single spaces to protect CSV integrity
            return " ".join(extracted.split())
        return ""

    # STAGE 2: The "Next-Number Boundary" Rule
    data["Structure_and_Form"]         = extract_between(zone_a, "1", "2")
    data["Focal_Design"]               = extract_between(zone_a, "2", "3")
    data["Materials_Metal_Finish"]     = extract_between(zone_a, "3", "4")
    data["Craftsmanship_and_Settings"] = extract_between(zone_a, "4", "5")
    data["Structural_Mechanics"]       = extract_between(zone_a, "5", "6")
    data["Aesthetic_and_Environment"]  = extract_between(zone_a, "6", None) # Reads to the end of Zone A

    # STAGE 3: The SDXL Caption Anchor Lock
    # Search for the trigger word across both zones as a fail-safe
    search_area = zone_b if "gj_temple_jewel" in zone_b else text
    
    if "gj_temple_jewel" in search_area:
        start_idx = search_area.find("gj_temple_jewel")
        raw_caption = search_area[start_idx:]
        # Remove bolding, newlines, and trailing spaces
        clean_caption = " ".join(raw_caption.replace("**", "").split())
        data["SDXL_Caption"] = clean_caption

    return data


def main():
    print("🚀 Loading Llama-3.2-11B-Vision-Instruct in 4-bit...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(Path(DATASET_DIR).rglob(ext))
    
    total_images = len(image_paths)
    if total_images == 0:
        print("❌ No images found!")
        return
        
    print(f"✅ Model loaded. Found {total_images} images. Starting Mass Processing...\n")
    print("=" * 80)
    
    csv_headers = [
        "Brand_Name", "Category", "Image_Path", 
        "Structure_and_Form", "Focal_Design", "Materials_Metal_Finish", 
        "Craftsmanship_and_Settings", "Structural_Mechanics", 
        "Aesthetic_and_Environment", "SDXL_Caption", "Raw_Output"
    ]
    
    # STAGE 4: CSV Integrity (csv.QUOTE_ALL wraps every field in double quotes)
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers, quoting=csv.QUOTE_ALL)
            writer.writeheader()

    for idx, img_path in enumerate(image_paths, 1):
        sub_folder = img_path.parent.name
        main_folder = img_path.parent.parent.name
        img_str_path = str(img_path)
        
        # Clean terminal output
        print(f"[{idx}/{total_images}] {main_folder} | {sub_folder} | {img_path.name}")
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Verified Category from Folder: {sub_folder}\n\n{MASTER_PROMPT}"}
                ]}
            ]
            
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs, 
                    max_new_tokens=600, 
                    do_sample=False, 
                    temperature=None, 
                    top_p=None
                )
            
            generated_text = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Map columns cleanly
            parsed_data = parse_llm_output(generated_text)
            
            row_data = {
                "Brand_Name": main_folder,
                "Category": sub_folder,
                "Image_Path": img_str_path,
                **parsed_data
            }
            
            # Instant append ensuring zero data loss
            with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers, quoting=csv.QUOTE_ALL)
                writer.writerow(row_data)
                
        except Exception as e:
            print(f"   ❌ Error processing {img_path.name}: {e}")

    print(f"\n🎉 ALL {total_images} IMAGES PROCESSED! Master CSV safely saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()