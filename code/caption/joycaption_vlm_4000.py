import os
import sys
import re
import warnings
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# Ensure clean UTF-8 terminal encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration & Paths ---
MODEL_DIR = r"D:\gen jewels\web scraping\models\llama-joycaption-beta-one-hf-llava"
DATASET_DIR = r"D:\gen jewels\web scraping\dataset_1024_framing"
OUTPUT_CSV = os.path.join(DATASET_DIR, "joycaption_full_dataset_captions.csv")

# Standardized 11-Column Schema
CSV_HEADERS = [
    "Image_Path",
    "Brand_Name",
    "Category",
    "Structure_and_Form",
    "Focal_Design",
    "Materials_Metal_Finish",
    "Craftsmanship_and_Settings",
    "Structural_Mechanics",
    "Aesthetic_and_Environment",
    "SDXL_Caption",
    "Full_Raw_Output"
]

# --- Master Jewelry Prompt for JoyCaption ---
JOYCAPTION_PROMPT_TEMPLATE = """You are a Senior Gemologist and Master Indian Jewelry Appraiser. Analyze this fine jewelry image on a solid black background.

Verified Category: {category}

MANDATORY INSTRUCTION: You MUST write out all 6 numbered points in exact order BEFORE generating the final caption. Do NOT skip any number.

CRITICAL RULES:
1. STRICT PHYSICAL ACCURACY (ZERO COLOR HALLUCINATION):
   - Report ONLY colors physically present in the photo.
   - Do NOT guess purple or tanzanite unless bright purple is unmistakably visible.
   - Identify green stones as green emerald, clear stones as diamonds, and white beads as seed pearls.
2. INDIAN TEMPLE JEWELRY RECOGNITION:
   - Identify antique gold relief carvings of deities (e.g., seated Goddess Lakshmi on a lotus, Lord Ganesha, peacocks).
   - Identify hanging seed pearls, ghungroo bells, or drop beads accurately.
3. MACRO vs. FULL SILHOUETTE:
   - If an image shows an isolated chain link, tennis necklace segment, or cropped piece with edges cut off by the canvas, state: "Macro structural detail of {category} segment".
   - ONLY state "Full {category} silhouette" if the complete piece is fully visible.
4. STRICT POSITIVE STATEMENTS ONLY:
   - FORBIDDEN: Never write 'no articulated links', 'without stones', 'no clasps', or 'no danglers'. State only what is present.
5. CONCISE FORMAT:
   - Each numbered layer MUST be strictly ONE single concise sentence (under 15 words).

FORMAT STRICTLY AS:

1. Overall Structure & Form: [Full silhouette or Macro segment, shape and layout]
2. Main Focal Design: [Center deity carving, floral stud, emerald drop, or repeating chain links]
3. Materials, Metal & Finish: [22k antique yellow gold or 18k gold, high-polish or matte Nakshi]
4. Micro Craftsmanship & Settings: [Prong settings, bezel settings, pavé, or rawa wirework]
5. Micro Mechanics & Articulation: [Articulated flexible links, hanging bead drops, or solid studs]
6. Overall Aesthetic & Environment: Sharp macro focus, solid black studio background

Final SDXL Caption: gj_temple_jewel, {category}, [Full silhouette or macro], [Motif & genuine stones], [Metal tone & finish], [Settings & mechanics], solid black studio background"""

# --- Bulletproof Parsing Function ---
def parse_joycaption_output(raw_text: str, category: str = ""):
    cat = category or "Jewelry"
    
    # Pre-clean markdown bolding to make headers uniform
    normalized_text = raw_text.replace("**", " ").replace("##", " ")
    
    # Flexible regex patterns for headers
    section_patterns = [
        ("Structure_and_Form", r"(?:1[.)\-:]?\s*(?:Overall\s+)?Structure(?:\s*(?:&|and)\s*Form)?[:\s\-]*)"),
        ("Focal_Design", r"(?:2[.)\-:]?\s*(?:Main\s+)?Focal(?:\s*Design)?[:\s\-]*)"),
        ("Materials_Metal_Finish", r"(?:3[.)\-:]?\s*Materials?,?(?:\s*Metal)?(?:\s*(?:&|and)\s*Finish)?[:\s\-]*)"),
        ("Craftsmanship_and_Settings", r"(?:4[.)\-:]?\s*(?:Micro\s+)?Craftsmanship(?:\s*(?:&|and)\s*Settings?)?[:\s\-]*)"),
        ("Structural_Mechanics", r"(?:5[.)\-:]?\s*(?:Micro\s+)?Mechanics(?:\s*(?:&|and)\s*Articulation)?[:\s\-]*)"),
        ("Aesthetic_and_Environment", r"(?:6[.)\-:]?\s*(?:Overall\s+)?Aesthetic(?:\s*(?:&|and)\s*Environment)?[:\s\-]*)"),
        ("SDXL_Caption", r"(?:(?:Final\s+)?SDXL\s+Caption[:\s\-]*|gj_temple_jewel)")
    ]

    found_spans = []
    for key, pattern in section_patterns:
        m = re.search(pattern, normalized_text, re.IGNORECASE)
        if m:
            found_spans.append((m.start(), m.end(), key))

    found_spans.sort(key=lambda x: x[0])

    parsed = {
        "Structure_and_Form": "N/A",
        "Focal_Design": "N/A",
        "Materials_Metal_Finish": "N/A",
        "Craftsmanship_and_Settings": "N/A",
        "Structural_Mechanics": "N/A",
        "Aesthetic_and_Environment": "Sharp macro focus, solid black studio background",
        "SDXL_Caption": "N/A"
    }

    # Slice each layer from span boundaries
    for idx, (start, end, key) in enumerate(found_spans):
        next_start = found_spans[idx + 1][0] if idx + 1 < len(found_spans) else len(normalized_text)
        val = normalized_text[end:next_start].replace("\n", " ").strip()
        val = re.sub(r"^[-*:\s]+", "", val).strip()

        # Keep ONLY the first concise sentence
        if key != "SDXL_Caption" and ". " in val:
            val = val.split(". ")[0].strip()
            if not val.endswith("."):
                val += "."

        # Strip negative phrasing leaks
        val = re.sub(r",?\s*no articulated links or clasps\.?", "", val, flags=re.IGNORECASE)
        val = re.sub(r",?\s*no articulated links\.?", "", val, flags=re.IGNORECASE)
        val = re.sub(r",?\s*no clasps\.?", "", val, flags=re.IGNORECASE)
        val = re.sub(r",?\s*no danglers\.?", "", val, flags=re.IGNORECASE)
        val = re.sub(r",?\s*without stones\.?", "", val, flags=re.IGNORECASE)
        val = re.sub(r"\s+", " ", val).strip(" ,;.-")

        if val:
            parsed[key] = val

    # Extract SDXL_Caption
    m_trigger = re.search(r"(gj_temple_jewel[^\n\r]+)", normalized_text, re.IGNORECASE)
    if m_trigger:
        parsed["SDXL_Caption"] = m_trigger.group(1).strip()
    elif parsed["SDXL_Caption"] == "N/A":
        # Fallback build from layers
        valid_parts = [
            parsed[k] for k in [
                "Structure_and_Form", "Focal_Design", "Materials_Metal_Finish",
                "Craftsmanship_and_Settings", "Structural_Mechanics"
            ] if parsed[k] != "N/A"
        ]
        parsed["SDXL_Caption"] = f"gj_temple_jewel, {cat}, {', '.join(valid_parts)}, solid black studio background"

    # Fail-safe safety net: unpack final caption if layers were skipped
    if parsed["Structure_and_Form"] == "N/A" and "gj_temple_jewel" in parsed["SDXL_Caption"]:
        clean_tokens = [t.strip() for t in parsed["SDXL_Caption"].split(",") if t.strip()]
        if len(clean_tokens) >= 6:
            parsed["Structure_and_Form"] = clean_tokens[2]
            parsed["Focal_Design"] = clean_tokens[3]
            parsed["Materials_Metal_Finish"] = clean_tokens[4]
            parsed["Craftsmanship_and_Settings"] = clean_tokens[5]
            if len(clean_tokens) >= 7:
                parsed["Structural_Mechanics"] = clean_tokens[6]
            else:
                parsed["Structural_Mechanics"] = "Solid structural construction"

    # Clean the SDXL Caption formatting
    caption = parsed["SDXL_Caption"]
    caption = re.sub(r"\b(Metal tone|Settings|Mechanics|Structure|Motif)\s*:\s*", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\b(?:metal tone(?:\s*&\s*finish)?|settings(?:\s*&\s*mechanics)?|motif(?:\s*&\s*genuine stones)?|full silhouette(?:\s*or\s*macro)?)\s*:\s*", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r",?\s*no articulated links or clasps", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r",?\s*no articulated links", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r",?\s*no clasps", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r",?\s*no danglers", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r",?\s*without stones", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\[.*?\]", "", caption).replace("[", "").replace("]", "")
    caption = re.sub(r",\s*,+", ",", caption)
    caption = re.sub(r"\s+", " ", caption).strip(" ,;.-")

    if not caption.lower().endswith("solid black studio background"):
        caption = re.sub(r"(?:,\s*)?solid black studio background.*$", "", caption, flags=re.IGNORECASE).strip(" ,;.-")
        caption = f"{caption}, solid black studio background"

    parsed["SDXL_Caption"] = caption
    return parsed

def main():
    print("🚀 [1/3] Loading JoyCaption Beta One (4-bit NF4) onto GPU 0...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["vision_tower"]
    )

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()

    print("\n📁 [2/3] Scanning Dataset Directory & Checking Resume Status...")
    
    # 1. Discover all images
    all_image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        all_image_paths.extend(Path(DATASET_DIR).rglob(ext))

    total_dataset_count = len(all_image_paths)
    print(f"📊 Total images found in dataset: {total_dataset_count}")

    # 2. Crash-Proof Resume Check
    processed_paths = set()
    file_exists = os.path.isfile(OUTPUT_CSV)
    
    if file_exists:
        try:
            df_existing = pd.read_csv(OUTPUT_CSV, usecols=["Image_Path"])
            processed_paths = set(df_existing["Image_Path"].dropna().tolist())
            print(f"🔄 Resume Active: Found {len(processed_paths)} previously processed images.")
        except Exception as e:
            print(f"⚠️ Warning reading existing CSV: {e}")

    # Initialize CSV with header row if brand new
    if not file_exists:
        pd.DataFrame(columns=CSV_HEADERS).to_csv(OUTPUT_CSV, index=False, quoting=1)

    # Filter out completed files
    images_to_process = [p for p in all_image_paths if str(p) not in processed_paths]
    remaining_count = len(images_to_process)
    
    if remaining_count == 0:
        print("🎉 All images in the dataset have already been processed! Exiting.")
        return

    print(f"✅ Ready to process remaining: {remaining_count} images.")
    print(f"💾 Saving row-by-row to: {OUTPUT_CSV}\n")
    print("=" * 80)

    # 3. Batch Processing Loop
    for idx, img_path in enumerate(images_to_process):
        img_str_path = str(img_path)

        # Folder taxonomy
        category_raw = img_path.parent.name
        brand_name = img_path.parent.parent.name
        clean_category = category_raw.replace("CC_", "").replace("CC ", "").replace("_", " ").strip()

        current_num = len(processed_paths) + idx + 1
        print(f"\n[{current_num}/{total_dataset_count}] Brand: {brand_name} | Category: {clean_category} | File: {img_path.name}")
        print("-" * 75)

        try:
            raw_image = Image.open(img_path).convert("RGB")
            raw_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            prompt_text = JOYCAPTION_PROMPT_TEMPLATE.format(category=clean_category)

            if hasattr(processor, "apply_chat_template") and processor.chat_template:
                convo = [
                    {
                        "role": "system",
                        "content": "You are a forensic jewelry appraiser. You always output a structured 6-point appraisal followed by a final caption.",
                    },
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ]
                formatted_prompt = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = f"<image>\n{prompt_text}"

            inputs = processor(text=formatted_prompt, images=raw_image, return_tensors="pt").to("cuda:0")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=550,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
                )

            input_token_len = inputs["input_ids"].shape[1]
            generated_ids_trimmed = generated_ids[:, input_token_len:]
            
            raw_output = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            parsed = parse_joycaption_output(raw_output, category=clean_category)

            # Build single row matching exact 11-column schema
            row_data = {
                "Image_Path": img_str_path,
                "Brand_Name": brand_name,
                "Category": clean_category,
                "Structure_and_Form": parsed["Structure_and_Form"],
                "Focal_Design": parsed["Focal_Design"],
                "Materials_Metal_Finish": parsed["Materials_Metal_Finish"],
                "Craftsmanship_and_Settings": parsed["Craftsmanship_and_Settings"],
                "Structural_Mechanics": parsed["Structural_Mechanics"],
                "Aesthetic_and_Environment": parsed["Aesthetic_and_Environment"],
                "SDXL_Caption": parsed["SDXL_Caption"],
                "Full_Raw_Output": raw_output
            }

            # --- INCREMENTAL WRITE (Row-by-Row, quoting=1) ---
            pd.DataFrame([row_data]).to_csv(OUTPUT_CSV, mode="a", header=False, index=False, quoting=1)

            print(f"   ➤ Final SDXL: {parsed['SDXL_Caption']}")
            print("   💾 Saved to CSV.")

        except Exception as e:
            print(f"❌ Error processing image {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 75)

    print(f"\n🎉 FULL RUN COMPLETE! All rows safely exported to:\n   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()