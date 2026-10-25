# import os
# import sys
# import re
# from pathlib import Path
# import pandas as pd
# from llama_cpp import Llama

# # =====================================================================
# # 1. PATH CONFIGURATION
# # =====================================================================
# MODEL_PATH = r"D:\gen jewels\web scraping\models\Qwen2.5-32B-Instruct-GGUF\Qwen2.5-32B-Instruct-Q4_K_M.gguf"

# CSV_PATHS = {
#     "florence2":  r"D:\gen jewels\web scraping\dataset_1024_framing\csv\florence2_full_dataset_captions.csv",
#     "internvl2":  r"D:\gen jewels\web scraping\dataset_1024_framing\csv\internvl2_segmented_dataset.csv",
#     "llama32":    r"D:\gen jewels\web scraping\dataset_1024_framing\csv\jewelry_master_llama_3.2_11b_captions.csv",
#     "joycaption": r"D:\gen jewels\web scraping\dataset_1024_framing\csv\joycaption_full_dataset_captions.csv",
#     "qwen2":      r"D:\gen jewels\web scraping\dataset_1024_framing\csv\qwen2_full_dataset_captions.csv",
# }

# DATASET_ROOTS = [
#     r"D:\gen jewels\web scraping\dataset_1024_framing\Tanishq",
#     r"D:\gen jewels\web scraping\dataset_1024_framing\BhimaJewellery",
#     r"D:\gen jewels\web scraping\dataset_1024_framing\Candere",
#     r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed",
# ]

# # =====================================================================
# # 2. IN-MEMORY HASH-MAP BUILDER (CPU / RAM)
# # =====================================================================
# def find_col(df, candidates):
#     cols_map = {col.lower().strip(): col for col in df.columns}
#     for cand in candidates:
#         if cand.lower() in cols_map:
#             return cols_map[cand.lower()]
#     return None

# def load_csv_to_ram(model_name, file_path):
#     print(f"[*] Preloading CSV: [{model_name}] -> {Path(file_path).name}")
#     if not os.path.exists(file_path):
#         print(f"    [!] File not found: {file_path}")
#         return {}
    
#     df = pd.read_csv(file_path, low_memory=False)
    
#     # Auto-detect column headers regardless of formatting variations
#     path_col  = find_col(df, ["image_path", "image", "filename", "file_name", "path"])
#     raw_col   = find_col(df, ["raw_output", "raw_caption", "caption", "full_appraisal_text", "description"])
#     sdxl_col  = find_col(df, ["sdxl_caption", "sdxl_prompt", "prompt"])
#     brand_col = find_col(df, ["brand_name", "brand", "folder_brand"])
#     cat_col   = find_col(df, ["category", "jewelry_type", "type"])
    
#     table = {}
#     for _, row in df.iterrows():
#         val = str(row[path_col]) if path_col and pd.notna(row[path_col]) else ""
#         if not val:
#             continue
        
#         # Primary lookup key is the base filename (lowercased)
#         key = Path(val).name.lower().strip()
#         table[key] = {
#             "raw": str(row[raw_col]).strip() if raw_col and pd.notna(row[raw_col]) else "",
#             "sdxl": str(row[sdxl_col]).strip() if sdxl_col and pd.notna(row[sdxl_col]) else "",
#             "brand": str(row[brand_col]).strip() if brand_col and pd.notna(row[brand_col]) else "",
#             "category": str(row[cat_col]).strip() if cat_col and pd.notna(row[cat_col]) else "",
#         }
#     print(f"    Loaded {len(table)} records.")
#     return table

# # =====================================================================
# # 3. QWEN2.5-32B GGUF INITIALIZATION
# # =====================================================================
# def load_qwen_gguf(model_path):
#     print(f"\n[*] Initializing Qwen2.5-32B Instruct (Q4_K_M)...")
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
#     # 24 layers offloaded to 12GB VRAM (~7.8 GB VRAM); remaining layers run on CPU RAM
#     llm = Llama(
#         model_path=model_path,
#         n_gpu_layers=24,      # Fits cleanly in 12GB RTX 3060 with context buffer
#         n_ctx=2560,           # Sufficient context length for 5 model reports
#         n_threads=6,          # Optimal CPU worker threads
#         verbose=False
#     )
#     print("[+] Qwen2.5-32B loaded successfully.\n")
#     return llm

# # =====================================================================
# # 4. SYNTHESIS ENGINE
# # =====================================================================
# def synthesize_sdxl_prompt(llm, brand, category, reports):
#     system_prompt = (
#         "You are the Supreme Master Gemologist and Chief Indian Jewelry Appraiser. "
#         "Cross-examine the 5 independent vision reports for the same luxury jewelry piece. "
#         "Eliminate all hallucinations, conflicting guesses, and marketing fluff. "
#         "Synthesize the physical reality into ONE concise, photo-accurate SDXL training prompt."
#     )
    
#     user_prompt = f"""Ground Truth Metadata:
# - Brand Origin: {brand}
# - Verified Category: {category}

# === 5 MODEL VISION REPORTS ===
# [Model 1 - Florence-2]: {reports.get('florence2', 'N/A')}
# [Model 2 - InternVL2]: {reports.get('internvl2', 'N/A')}
# [Model 3 - Llama-3.2]: {reports.get('llama32', 'N/A')}
# [Model 4 - JoyCaption]: {reports.get('joycaption', 'N/A')}
# [Model 5 - Qwen2-VL]: {reports.get('qwen2', 'N/A')}

# STRICT RULES:
# 1. Output MUST strictly start with: gj_temple_jewel, {category}, 
# 2. Output MUST strictly end with: , solid black studio background
# 3. Strictly 30 to 42 natural English words total.
# 4. Positive statements only (NO negative words like 'no', 'without', 'absence').
# 5. Use authentic temple jewelry terminology (e.g., 22k yellow gold, Nakshi carving, rawa granulation, cabochon rubies, basra pearls, screw-back post, jor clasp).
# 6. Output ONLY the raw final prompt. No intro, no quotes, no conversational filler."""

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]

#     response = llm.create_chat_completion(
#         messages=messages,
#         max_tokens=100,
#         temperature=0.1,  # Near-deterministic to enforce strict rules
#         top_p=0.9,
#     )
    
#     output = response["choices"][0]["message"]["content"].strip()
#     # Remove any markdown wrapping if the LLM output it
#     output = re.sub(r"^```[a-zA-Z]*\n?|```$", "", output).strip()
#     output = re.sub(r"^['\"]+|['\"]+$", "", output).strip()
#     return output

# # =====================================================================
# # 5. FIND 1 SAMPLE IMAGE PRESENT IN DATASET
# # =====================================================================
# def get_sample_image(roots, all_lookups):
#     image_exts = {".png", ".jpg", ".jpeg", ".webp"}
#     for root in roots:
#         if not os.path.exists(root):
#             continue
#         for folder_path, _, files in os.walk(root):
#             for file in files:
#                 if Path(file).suffix.lower() in image_exts:
#                     img_path = Path(folder_path) / file
#                     key = img_path.name.lower().strip()
#                     # Check if this image has at least 2 reports in the lookups
#                     matches = sum(1 for table in all_lookups.values() if key in table)
#                     if matches >= 2:
#                         return img_path
#     return None

# # =====================================================================
# # 6. MAIN EXECUTION
# # =====================================================================
# def main():
#     print("=" * 80)
#     print("  QWEN2.5-32B ENSEMBLE CONSENSUS SYNTHESIZER — SINGLE SAMPLE TEST")
#     print("=" * 80)

#     # Step 1: Preload all 5 CSVs into fast RAM tables
#     all_lookups = {}
#     for model_name, path in CSV_PATHS.items():
#         all_lookups[model_name] = load_csv_to_ram(model_name, path)

#     # Step 2: Find 1 valid test image that exists on disk
#     target_image = get_sample_image(DATASET_ROOTS, all_lookups)
#     if not target_image:
#         print("[!] No matching image found across roots. Please check folder paths.")
#         return

#     # Step 3: Load Qwen2.5-32B GGUF
#     llm = load_qwen_gguf(MODEL_PATH)

#     # Step 4: Extract the 5 reports for this target image
#     filename_key = target_image.name.lower().strip()
#     reports = {}
#     brand = "Indian Heritage Jeweller"
#     category = target_image.parent.name.replace("_", " ")

#     for model_name, lookup in all_lookups.items():
#         entry = lookup.get(filename_key, {})
#         # Use raw appraisal first; if empty, fallback to sdxl caption
#         content = entry.get("raw") or entry.get("sdxl") or "No description available."
#         reports[model_name] = content
#         if entry.get("brand"):
#             brand = entry["brand"]
#         if entry.get("category"):
#             category = entry["category"]

#     # Step 5: Print the 5 model inputs in the terminal
#     print("\n" + "=" * 80)
#     print(f" TARGET SAMPLE IMAGE: {target_image.name}")
#     print(f" Path: {target_image}")
#     print(f" Metadata Ground Truth -> Brand: {brand} | Category: {category}")
#     print("-" * 80)
#     print(" 5 INPUT CAPTIONS SENT TO QWEN2.5-32B:")
#     for model_name, text in reports.items():
#         print(f"\n [{model_name.upper()}]:\n {text}")

#     # Step 6: Run Synthesis
#     print("\n" + "-" * 80)
#     print("[*] Running Qwen2.5-32B consensus reasoning...")
#     final_caption = synthesize_sdxl_prompt(llm, brand, category, reports)

#     # Step 7: Print the Final SDXL Caption & Validation
#     words = final_caption.split()
#     word_count = len(words)
#     has_prefix = final_caption.startswith(f"gj_temple_jewel, {category}") or final_caption.startswith("gj_temple_jewel")
#     has_suffix = final_caption.endswith("solid black studio background")

#     print("\n" + "=" * 80)
#     print(">>> FINAL SYNTHESIZED SDXL PROMPT:")
#     print(final_caption)
#     print("-" * 80)
#     print(f"Word Count     : {word_count} words (Target: 30-42)")
#     print(f"Trigger Prefix : {'[OK]' if has_prefix else '[CHECK PREFIX]'}")
#     print(f"Studio Suffix  : {'[OK]' if has_suffix else '[CHECK SUFFIX]'}")
#     print("=" * 80 + "\n")

# if __name__ == "__main__":
#     main()



import os
import sys
import re
from pathlib import Path
import pandas as pd
from llama_cpp import Llama

# =====================================================================
# 1. PATH CONFIGURATIONS
# =====================================================================
# TARGET_IMAGE_PATH = r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\CC_Necklace\CC_Necklace_Prod2_Angle5_1200x1200_797196_clean_black_bg.png"
TARGET_IMAGE_PATH = r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Gold_Necklace\Gold_Necklace_Prod38_Angle1_1200x1200_8477ef_clean_black_bg.png"
MODEL_PATH = r"D:\gen jewels\web scraping\models\Qwen2.5-32B-Instruct-GGUF\Qwen2.5-32B-Instruct-Q4_K_M.gguf"

CSV_PATHS = {
    "florence2":  r"D:\gen jewels\web scraping\dataset_1024_framing\csv\florence2_full_dataset_captions.csv",
    "internvl2":  r"D:\gen jewels\web scraping\dataset_1024_framing\csv\internvl2_segmented_dataset.csv",
    "llama32":    r"D:\gen jewels\web scraping\dataset_1024_framing\csv\jewelry_master_llama_3.2_11b_captions.csv",
    "joycaption": r"D:\gen jewels\web scraping\dataset_1024_framing\csv\joycaption_full_dataset_captions.csv",
    "qwen2":      r"D:\gen jewels\web scraping\dataset_1024_framing\csv\qwen2_full_dataset_captions.csv",
}

# =====================================================================
# 2. HELPER FUNCTIONS FOR CSV SEARCH
# =====================================================================
def find_col(df, candidates):
    cols_map = {col.lower().strip(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in cols_map:
            return cols_map[cand.lower()]
    return None

def extract_report_from_csv(csv_path, target_filename):
    """Searches the CSV for the target filename with exact and stem fallback matching."""
    if not os.path.exists(csv_path):
        return None, f"File not found: {csv_path}"

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        return None, f"Error reading CSV: {e}"

    path_col  = find_col(df, ["image_path", "image", "filename", "file_name", "path"])
    raw_col   = find_col(df, ["raw_output", "raw_caption", "caption", "full_appraisal_text", "description"])
    sdxl_col  = find_col(df, ["sdxl_caption", "sdxl_prompt", "prompt"])
    brand_col = find_col(df, ["brand_name", "brand", "folder_brand"])
    cat_col   = find_col(df, ["category", "jewelry_type", "type"])

    if not path_col:
        return None, "Path column not found in CSV."

    target_clean = target_filename.lower().strip()
    target_stem = Path(target_filename).stem.lower().replace("_clean_black_bg", "")

    # 1. Exact filename match
    matched_row = None
    for _, row in df.iterrows():
        val = str(row[path_col]).strip()
        if not val or val.lower() == "nan":
            continue
        row_fname = Path(val).name.lower().strip()
        
        if row_fname == target_clean or target_clean in row_fname:
            matched_row = row
            break
        # Fallback to base stem match
        if target_stem in row_fname:
            matched_row = row
            break

    if matched_row is None:
        return None, "Image not found in this CSV."

    raw_text = str(matched_row[raw_col]).strip() if raw_col and pd.notna(matched_row[raw_col]) else ""
    sdxl_text = str(matched_row[sdxl_col]).strip() if sdxl_col and pd.notna(matched_row[sdxl_col]) else ""
    brand_text = str(matched_row[brand_col]).strip() if brand_col and pd.notna(matched_row[brand_col]) else "DAR Jewellery"
    cat_text = str(matched_row[cat_col]).strip() if cat_col and pd.notna(matched_row[cat_col]) else "Necklace"

    # Prioritize raw output; fallback to sdxl caption
    content = raw_text if len(raw_text) > 10 else sdxl_text
    return {
        "content": content,
        "brand": brand_text,
        "category": cat_text
    }, None

# =====================================================================
# 3. QWEN2.5-32B GGUF LOADER
# =====================================================================
def load_qwen_model(model_path):
    print(f"\n[*] Loading Qwen2.5-32B Instruct (Q4_K_M)...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # n_gpu_layers=24 offloads ~7.8GB to RTX 3060 (12GB), rest in system RAM
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=24,
        n_ctx=2560,
        n_threads=6,
        verbose=False
    )
    print("[+] Qwen2.5-32B loaded successfully.")
    return llm

# =====================================================================
# 4. MASTER SYNTHESIS CALL
# =====================================================================
def run_consensus(llm, brand, category, reports):
    system_prompt = (
        "You are the Supreme Master Gemologist and Chief Indian Jewelry Appraiser. "
        "Cross-examine the 5 independent vision reports for the same luxury jewelry piece. "
        "Eliminate all hallucinations, conflicting guesses, and marketing fluff. "
        "Synthesize the physical reality into ONE concise, photo-accurate SDXL training prompt."
    )
    
    user_prompt = f"""Ground Truth Metadata:
        - Brand Origin: {brand}
        - Verified Category: {category}

        === 5 MODEL VISION REPORTS ===
        [Model 1 - Florence-2]: {reports.get('florence2', 'N/A')}
        [Model 2 - InternVL2]: {reports.get('internvl2', 'N/A')}
        [Model 3 - Llama-3.2]: {reports.get('llama32', 'N/A')}
        [Model 4 - JoyCaption]: {reports.get('joycaption', 'N/A')}
        [Model 5 - Qwen2-VL]: {reports.get('qwen2', 'N/A')}

        STRICT RULES:
        1. SPECIFIC ICONOGRAPHY OVER GENERIC LABELS (CRITICAL):
        - If ANY model mentions a specific Hindu deity (e.g. Goddess Lakshmi, Lord Ganesha, Krishna) or traditional animal (peacock, elephant), you MUST use the exact name (e.g. 'seated Goddess Lakshmi carving on lotus', 'flanking carved peacocks') instead of generic words like 'deity', 'figure', or 'pattern'.
        2. NO HALLUCINATED DIAMONDS:
        - For antique 22k matte gold temple jewelry with rubies or ghungroos, do NOT add diamonds unless diamonds are unmistakably verified across multiple reports.
        3. WEARABLE MECHANICS & MOTIFS:
        - Clearly state the build: mention layered haram, open-cuff hinge, hanging gold ghungroo beads, or drop pearls when described.
        4. TOKEN BUDGET & FORMAT:
        - Output MUST strictly start with: gj_temple_jewel, {category}, 
        - Output MUST strictly end with: , solid black studio background
        - Strictly 30 to 42 natural English words total.
        - Positive statements only (NO 'no', 'without'). Avoid stuttering duplicate words.
        - Do NOT end with a period. Output ONLY the raw final prompt line with zero quotes or commentary."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=120,
        temperature=0.1,
        top_p=0.9,
    )
    
    output = response["choices"][0]["message"]["content"].strip()
    output = re.sub(r"^```[a-zA-Z]*\n?|```$", "", output).strip()
    output = re.sub(r"^['\"]+|['\"]+$", "", output).strip()
    output = output.rstrip(" .")  # Strips trailing period so [CHECK SUFFIX] passes
    return output

# =====================================================================
# 5. MAIN EXECUTION
# =====================================================================
def main():
    target_path = Path(TARGET_IMAGE_PATH)
    target_filename = target_path.name

    print("=" * 80)
    print(" TARGET IMAGE EXTRACTION & CONSENSUS")
    print(f" Target File: {target_filename}")
    print("=" * 80)

    # 1. Search all 5 CSVs for this exact image
    reports = {}
    brand = "DAR Jewellery"
    category = "Necklace"

    for model_key, csv_path in CSV_PATHS.items():
        data, err = extract_report_from_csv(csv_path, target_filename)
        if data:
            reports[model_key] = data["content"]
            if data["brand"] and data["brand"].lower() != "nan":
                brand = data["brand"]
            if data["category"] and data["category"].lower() != "nan":
                category = data["category"]
        else:
            reports[model_key] = f"Error/Missing: {err}"

    # Clean category name (e.g., CC_Necklace -> Necklace)
    category = category.replace("CC_", "").replace("_", " ").strip()

    # 2. Display the 5 VLM input captions
    print(f"\nMetadata: Brand = {brand} | Category = {category}")
    print("-" * 80)
    print("5 VLM RAW INPUT CAPTIONS FOR THIS IMAGE:")
    print("-" * 80)
    for model_key, text in reports.items():
        print(f"\n>>> [{model_key.upper()}]:")
        print(text)

    # 3. Load Qwen2.5-32B Model
    llm = load_qwen_model(MODEL_PATH)

    # 4. Generate the Synthesized SDXL Prompt
    print("\n" + "=" * 80)
    print("[*] Synthesizing reports through Qwen2.5-32B...")
    final_caption = run_consensus(llm, brand, category, reports)

    # 5. Validation Check & Final Output
    words = final_caption.split()
    word_count = len(words)
    has_prefix = final_caption.startswith(f"gj_temple_jewel, {category}") or final_caption.startswith("gj_temple_jewel")
    has_suffix = final_caption.endswith("solid black studio background")

    print("\n" + "=" * 80)
    print(">>> FINAL SYNTHESIZED SDXL PROMPT:")
    print(final_caption)
    print("=" * 80)
    print(f"Word Count     : {word_count} words (Target: 30-42 words)")
    print(f"Trigger Prefix : {'[OK]' if has_prefix else '[CHECK PREFIX]'}")
    print(f"Studio Suffix  : {'[OK]' if has_suffix else '[CHECK SUFFIX]'}")
    print("=" * 80)

if __name__ == "__main__":
    main()