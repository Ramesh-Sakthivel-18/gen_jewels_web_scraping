# this code is run in PICCC-8 system and and the files are linked to that system 



import sys
import torch
import warnings
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

# The exact 3 image paths for testing
TEST_IMAGES = [
    r"D:\ramesh\jewelry ipa\training\dataset_1024_framing\dar_jewelry_watermark_removed\Diamond_Necklace\Diamond_Necklace_Prod3_Angle2_1200x1200_f5ba1e_clean_black_bg.png",
    r"D:\ramesh\jewelry ipa\training\dataset_1024_framing\dar_jewelry_watermark_removed\Diamond_Necklace\Diamond_Necklace_Prod3_Angle3_1200x1200_ac0c30_clean_black_bg.png",
    r"D:\ramesh\jewelry ipa\training\dataset_1024_framing\Tanishq\Earrings\tanishq_2000x2000_0f4a1821_black_bg.png"
]

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
# --- HIGH-RES DYNAMIC IMAGE PATCHING (InternVL2 Official Logic) ---
# This prevents 2000x2000 images from becoming blurry by intelligently 
# slicing them into crystal-clear 448x448 tiles.
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
    # Stack the multiple high-res patches into a single batch tensor
    pixel_values = torch.stack(pixel_values).to(torch.float16).cuda()
    return pixel_values
# ==============================================================================

def main():
    print("🚀 [1/3] Loading InternVL2-8B to GPU 0 (4-bit Quantization + Dynamic High-Res Patching)...")

    # use_fast=False is REQUIRED for InternVL2 to prevent the 'bool' tokenizer bug.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False
    )

    # Sanity-check: confirm we have a real tokenizer
    assert hasattr(tokenizer, 'convert_tokens_to_ids'), \
        f"Tokenizer load FAILED -- got {type(tokenizer)} instead of a real tokenizer."

    print(f"✅ Tokenizer loaded successfully: {type(tokenizer).__name__}")

    # 4-bit config to safely fit an 8B model into your 12GB VRAM.
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

    print("✅ Model loaded successfully. Starting 3-image inference...\n")
    print("=" * 100)

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    for idx, img_str_path in enumerate(TEST_IMAGES):
        path_obj = Path(img_str_path)

        # Check if file actually exists
        if not path_obj.exists():
            print(f"❌ [ERROR] Could not find image at {img_str_path}\n")
            continue

        # Extract category from the parent folder
        raw_category = path_obj.parent.name
        clean_category = raw_category.replace("_", " ")

        # Inject the category and prepend "<image>\n" required by InternVL
        raw_prompt = INTERNVL_PROMPT.format(category=clean_category)
        safe_prompt = "<image>\n" + raw_prompt

        print(f"🔍 [{idx + 1}/3] Testing Image - Category: '{clean_category}'")
        print(f"📁 File: {path_obj.name}")
        print("-" * 50)

        try:
            # Load and slice image into high-res tiles automatically
            pixel_values = load_image(img_str_path, max_num=4)

            # Pass arguments POSITIONALLY to bypass internal routing bugs.
            with torch.no_grad():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    safe_prompt,
                    generation_config,
                    history=None,
                    return_history=False
                )

            # Print the output strictly to the terminal
            print("\n🤖 [INTERNVL2-8B HIGH-RES APPRAISAL]\n")
            print(response.strip())
            print("\n" + "=" * 100 + "\n")

        except Exception as e:
            import traceback
            print(f"❌ [ERROR] Processing {path_obj.name}: {e}")
            traceback.print_exc()

    print("\n🎉 [DONE] Test complete! Dynamic patching generated high-res descriptions.")

if __name__ == "__main__":
    main()