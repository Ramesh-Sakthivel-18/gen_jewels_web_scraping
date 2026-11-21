import os
import re
import warnings
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from PIL import Image

# ==============================================================================
# --- CLEAN TERMINAL & HUGGINGFACE MONKEY PATCHES ---
# Ensures modern Transformers library doesn't crash on Florence-2
# ==============================================================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

nn.Module._supports_sdpa = False

import transformers
transformers.configuration_utils.PretrainedConfig.forced_bos_token_id = None
transformers.configuration_utils.PretrainedConfig.forced_eos_token_id = None

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, 'additional_special_tokens'):
    PreTrainedTokenizerBase.additional_special_tokens = property(lambda self: [])

# Cache Fix for Florence-2 memory loops
from transformers.cache_utils import EncoderDecoderCache, DynamicCache
EncoderDecoderCache.__getitem__ = lambda self, idx: (self.self_attention_cache, self.cross_attention_cache)[idx]
EncoderDecoderCache.__iter__ = lambda self: iter((self.self_attention_cache, self.cross_attention_cache))
EncoderDecoderCache.__len__ = lambda self: 2

def safe_dynamic_cache_getitem(self, idx):
    keys = getattr(self, 'key_cache', getattr(self, '_key_cache', []))
    vals = getattr(self, 'value_cache', getattr(self, '_value_cache', []))
    return (keys, vals)[idx]

DynamicCache.__getitem__ = safe_dynamic_cache_getitem
DynamicCache.__iter__ = lambda self: iter((
    getattr(self, 'key_cache', getattr(self, '_key_cache', [])),
    getattr(self, 'value_cache', getattr(self, '_value_cache', []))
))
DynamicCache.__len__ = lambda self: 2

from transformers import AutoProcessor, AutoModelForCausalLM

# ==============================================================================
# --- SPECIFIC LOCAL PATHS (FULL DATASET) ---
# ==============================================================================
MODEL_DIR = r"D:\ramesh\jewelry ipa\training\model\Florence-2-large"
DATASET_DIR = r"D:\ramesh\jewelry ipa\training\dataset_1024_framing"
# Final output file for the entire dataset
OUTPUT_CSV = os.path.join(DATASET_DIR, "florence2_full_dataset_captions.csv")

# ==============================================================================
# --- PIPELINE LOGIC ---
# ==============================================================================
def run_florence_task(model, processor, image, task_token):
    """Executes a pure native task on Florence-2 with ZERO text instructions."""
    inputs = processor(text=task_token, images=image, return_tensors="pt").to('cuda:0')
    
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=512,
            num_beams=1,
            do_sample=False
        )
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_token, 
        image_size=(image.width, image.height)
    )
    return parsed_answer[task_token]

def clean_detailed_caption(text, category):
    """
    Cleans Florence-2's output:
    1. Removes secondary marketing fluff sentences entirely.
    2. Strips prefixes ('A stunning', 'The image shows').
    3. Overrides category mistakes ('bracelet' -> 'necklace chain segment').
    4. Removes duplicate background mentions.
    """
    raw = str(text).strip()
    
    # Split into individual sentences
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    
    cleaned_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # Drop sentences that are purely marketing fluff
        if re.search(r'\b(unique design|sparkle and elegance|timeless elegance|captivating appearance|touch of elegance|striking visual)\b', s_clean, re.I):
            continue
        cleaned_sentences.append(s_clean)
        
    # If all sentences were filtered, retain the first sentence
    combined = " ".join(cleaned_sentences) if cleaned_sentences else (sentences[0].strip() if sentences else "")
    
    # Strip conversational prefixes
    prefixes = [
        r"^the image shows\s+", r"^this image shows\s+", r"^the photo depicts\s+", 
        r"^this is a picture of\s+", r"^a close-up of\s+", r"^a close up of\s+",
        r"^a stunning\s+", r"^a beautiful\s+", r"^an elegant\s+", r"^a magnificent\s+"
    ]
    for p in prefixes:
        combined = re.sub(p, "", combined, flags=re.IGNORECASE).strip()
        
    # Strip internal mentions of the background
    bg_patterns = [
        r",?\s*(?:set\s+)?against a (?:solid\s+)?black (?:studio\s+)?background\.?",
        r",?\s*(?:on|with) a (?:solid\s+)?black (?:studio\s+)?background\.?",
        r",?\s*making the (?:gold|jewelry|necklace|earrings|piece) stand out\.?"
    ]
    for bg in bg_patterns:
        combined = re.sub(bg, "", combined, flags=re.IGNORECASE).strip()
        
    # Category Ground Truth: If folder is a Necklace, replace 'bracelet' with 'necklace chain segment'
    if "necklace" in category.lower():
        combined = re.sub(r'\bbracelet\b', 'necklace chain segment', combined, flags=re.IGNORECASE)
        
    # Clean trailing punctuation and trailing conjunctions
    combined = re.sub(r'[,.\s]+$', '', combined).strip()
    combined = re.sub(r'\b(with|and|the)\s*$', '', combined, flags=re.IGNORECASE).strip()
    combined = re.sub(r'[,.\s]+$', '', combined).strip()
    
    # Capitalize the first letter
    if combined:
        combined = combined[0].upper() + combined[1:]
        
    return combined

def clean_dense_labels(raw_labels, category):
    """Deduplicates labels and removes category contradictions (e.g. 'ring' on a necklace)."""
    if not isinstance(raw_labels, list):
        return ""
        
    unique_labels = []
    cat_lower = category.lower()
    
    for label in raw_labels:
        clean_lbl = str(label).strip().lower()
        if not clean_lbl:
            continue
            
        words = set(clean_lbl.split())
        is_contradiction = False
        
        # If it says 'ring' but the category is NOT a ring
        if "ring" in words and "ring" not in cat_lower and "earring" in cat_lower:
            is_contradiction = True
        # If it says 'necklace' or 'pendant' on an earring
        if ("necklace" in words or "pendant" in words) and "earring" in cat_lower:
            is_contradiction = True
            
        if not is_contradiction and clean_lbl not in unique_labels:
            unique_labels.append(clean_lbl)
            
    return ", ".join(unique_labels)

def main():
    print("🚀 [1/3] Loading Florence-2-Large to GPU 0 in FULL PRECISION (float16)...")
    
    # No quantization! Model is loaded in pure float16 native precision.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda:0",
        attn_implementation="eager"
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    print("📁 [2/3] Scanning Dataset Directory for all images...")
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(Path(DATASET_DIR).rglob(ext))
    
    total_images = len(image_paths)
    print(f"✅ Found {total_images} images. Starting bulk sequence...\n")
    print("=" * 80)
    
    results = []
    
    for idx, img_path in enumerate(image_paths):
        img_str_path = str(img_path)
        
        # Extract Category and Brand
        sub_folder = img_path.parent.name
        main_folder = img_path.parent.parent.name
        clean_category = sub_folder.replace("_", " ")

        print(f"[{idx + 1}/{total_images}] {main_folder} | {clean_category} | {img_path.name}")

        try:
            image = Image.open(img_str_path).convert('RGB')
            # Protect VRAM from 2000x2000 spikes (does not compress model logic)
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # -------------------------------------------------------------
            # PASS 1: Native Detailed Caption (Both Raw and Cleaned)
            # -------------------------------------------------------------
            raw_detailed = run_florence_task(model, processor, image, "<DETAILED_CAPTION>")
            final_detailed = clean_detailed_caption(raw_detailed, clean_category)
            
            # -------------------------------------------------------------
            # PASS 2: Native Dense Region Labels (Saved for CSV audit only)
            # -------------------------------------------------------------
            raw_dense = run_florence_task(model, processor, image, "<DENSE_REGION_CAPTION>")
            labels_list = raw_dense.get('labels', []) if isinstance(raw_dense, dict) else []
            final_keywords = clean_dense_labels(labels_list, clean_category)
            
            # -------------------------------------------------------------
            # PASS 3: Clean, Token-Safe SDXL Caption Assembly
            # Notice: final_keywords is NOT included to prevent token overflow and contradictions!
            # -------------------------------------------------------------
            sdxl_caption = f"gj_temple_jewel, {clean_category}, {final_detailed}, solid black studio background"
            
            # Remove any accidental double commas or spaces
            sdxl_caption = re.sub(r',\s*,', ',', sdxl_caption).strip()

            # Terminal Output Logging
            print(f"  ➤ Raw Output:         {str(raw_detailed).strip()}")
            print(f"  ➤ Detailed Summary:   {final_detailed}")
            print(f"  ➤ Extracted Keywords: {final_keywords}")
            print(f"  ➤ Final SDXL Caption: {sdxl_caption}")
            print("-" * 80)

            # Append to Results (Now includes Raw_Detailed_Caption)
            results.append({
                "Image_Path": img_str_path,
                "Brand_Name": main_folder,
                "Category": sub_folder,
                "Detailed_Caption": final_detailed,
                "Component_Keywords": final_keywords,
                "SDXL_Caption": sdxl_caption,
                "Raw_Detailed_Caption": raw_detailed  # <--- Added Raw Text Here!
            })
            
            # Incremental Save to CSV (quoting=1 safely wraps fields with commas in double quotes)
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, quoting=1)
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    print(f"\n🎉 Process Complete! Clean, token-safe SDXL CSV generated at: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()