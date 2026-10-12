import os
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

# --- Local Model Path ---
MODEL_DIR = r"D:\gen jewels\web scraping\models\Llama-3.2-11B-Vision-Instruct"

# --- The 7 Specific Images You Requested ---
TEST_IMAGES = [
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Diamond_Necklace\Diamond_Necklace_Prod2_Angle2_1200x1200_6c4e16_clean_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Diamond_Necklace\Diamond_Necklace_Prod2_Angle3_1200x1200_6b9a44_clean_black_bg.png",
    # r"D:\gen jewels\web scraping\dataset_1024_framing\dar_jewelry_watermark_removed\Diamond_Necklace\Diamond_Necklace_Prod2_Angle1_1200x1200_f9aa92_clean_black_bg.png",
    # r"D:\gen jewels\web scraping\dataset_1024_framing\BhimaJewellery\Bangles\098645e5-364f-4d61-b5be-8524a69db15a-NGO-BGSB-COD-S-L-NA-25894466-additional-1_black_bg.png",
    # r"D:\gen jewels\web scraping\dataset_1024_framing\Tanishq\Rings\tanishq_2000x2000_3d6db72e_black_bg.png",
    # r"D:\gen jewels\web scraping\dataset_1024_framing\Tanishq\Rings\tanishq_2000x2000_e975ff86_black_bg.png",
    # r"D:\gen jewels\web scraping\dataset_1024_framing\Tanishq\Earrings\tanishq_2000x2000_94324468_black_bg.png",
    r"D:\gen jewels\web scraping\dataset_1024_framing\Tanishq\Earrings\tanishq_513x470_0a4d3a7a_black_bg.png"
]

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


def main():
    print("🚀 [1/3] Loading Llama-3.2-11B-Vision-Instruct in 4-bit to RTX 3060...")
    
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
    
    print(f"✅ Model loaded successfully. Starting inspection of {len(TEST_IMAGES)} specific images.\n")
    print("=" * 80)
    
    for idx, raw_path in enumerate(TEST_IMAGES):
        # Clean up path string in case it has hidden quotes
        img_path = Path(raw_path.strip('\"\''))
        
        # Extract folder names to inject the Verified Category!
        sub_folder = img_path.parent.name         # e.g., 'Diamond_Necklace' or 'Rings'
        main_folder = img_path.parent.parent.name # e.g., 'dar_jewelry_watermark_removed'
        
        print(f"\n🔍 INSPECTING {idx + 1}/{len(TEST_IMAGES)}")
        print(f"   Brand: {main_folder} | Category: {sub_folder}")
        print(f"   Image: {img_path.name}")
        print("-" * 50)
        
        try:
            if not img_path.exists():
                print(f"❌ ERROR: File not found at path: {img_path}")
                continue

            image = Image.open(img_path).convert('RGB')
            
            # Format the conversation specifically for Llama-3.2-Vision
            # Notice how we inject 'Verified Category from Folder: {sub_folder}'
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Verified Category from Folder: {sub_folder}\n\n{MASTER_PROMPT}"}
                ]}
            ]
            
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, return_tensors="pt").to(model.device)
            
            # Generate output with strict deterministic settings (warnings eliminated)
            with torch.no_grad():
                output = model.generate(
                    **inputs, 
                    max_new_tokens=600, 
                    do_sample=False, 
                    temperature=None, 
                    top_p=None
                )
            
            # Decode only the generated text
            generated_text = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            print("🟢 LLAMA-3.2 OUTPUT:\n")
            print(generated_text.strip())
            print("\n" + "=" * 80)
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    print("\n🎉 INSPECTION COMPLETE! Please review the terminal outputs for full & zoomed images.")

if __name__ == "__main__":
    main()