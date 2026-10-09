import os
import random
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from pathlib import Path

# --- Specific Local Paths ---
MODEL_DIR = r"D:\gen jewels\web scraping\models\InternVL2-8B"
DATASET_DIR = r"D:\gen jewels\web scraping\dataset_1024_framing"

# --- The Ultimate Prompt ---
SYSTEM_PROMPT = (
    "You are a Master Indian & Diamond Jewelry Appraiser. Analyze this jewelry piece and output your exact findings. "
    "Do not include introductory filler.\n\n"
    
    "RULES (Mandatory Extractions):\n"
    "- Category: (Bangle, Necklace, Jhumka, Ring, etc.)\n"
    "- Metal Tone & Finish: (e.g., 22k Antique Gold, 18k White Gold, high-polish, oxidized)\n"
    "- Structural Style: (e.g., Temple, Kundan, Polki, Victorian, Modern Diamond)\n"
    "- Key Motifs: (e.g., Goddess Lakshmi, Peacock, Floral, Geometric)\n"
    "- Gemstone Types: (e.g., cabochon rubies, faceted emeralds, uncut Polki diamonds, brilliant-cut)\n"
    "- Setting Architecture: (e.g., bezel-set, sharp metal prongs, densely packed pavé, clustered halo, foil-backed)\n\n"
    
    "FREE PLAY (Deep Visual Observation):\n"
    "Visually scan the piece from the center focal point outwards. Describe the microscopic craftsmanship, "
    "focusing heavily on how the diamonds and gemstones are packed. Are they tightly clustered with zero "
    "metal showing (pavé/invisible), or spaced out? Are they raised on sharp jagged prongs or sitting "
    "flush/smooth in thick metal bezels? Describe the physical texture, the microscopic gaps between stones, "
    "the filigree wirework, and the 3D depth of the piece. Leave no detail unnoticed."
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
    print("\n[1/3] Loading InternVL2-8B in 4-bit Quantization (Strict Official OpenGVLab Method)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    # 4-bit quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load model — with bitsandbytes 4-bit quantization, the model is automatically
    # placed on the correct device. We use device_map="cuda:0" to pin directly to
    # GPU 0, which avoids accelerate's dispatch_model calling .to() on the quantized
    # model (the root cause of the ValueError).
    # IMPORTANT: Do NOT chain .eval() — it internally calls .to() which crashes
    # with quantized models. The model is already in eval-compatible state.
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("[2/3] Scanning Dataset Directory for images...")
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(Path(DATASET_DIR).rglob(ext))
    
    if len(image_paths) == 0:
        print("No images found in the directory! Please check your dataset path.")
        return
    elif len(image_paths) < 3:
        sample_images = image_paths
    else:
        sample_images = random.sample(image_paths, 3)
    
    print(f"[3/3] Selected {len(sample_images)} random images. Starting Inference on RTX 3060...\n")
    print("=" * 80)
    
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    for idx, img_path in enumerate(sample_images):
        img_str_path = str(img_path)
        print(f"\n--- IMAGE {idx + 1}: {img_path.name} ---")
        
        try:
            pixel_values = load_image(img_str_path)
            
            # model.chat() with return_history=True returns (response, history)
            # model.chat() with return_history=False returns just the response string
            with torch.no_grad():
                response = model.chat(
                    tokenizer, 
                    pixel_values, 
                    SYSTEM_PROMPT, 
                    generation_config, 
                    history=None, 
                    return_history=False
                )
            
            print("[GEM] MODEL OUTPUT:\n")
            print(response.strip())
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n" + "=" * 80)
        
if __name__ == "__main__":
    main()