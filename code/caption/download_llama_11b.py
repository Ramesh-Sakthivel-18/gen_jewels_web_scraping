import os
from huggingface_hub import snapshot_download

# --- Paths ---
# Your base models directory
BASE_MODEL_DIR = r"D:\gen jewels\web scraping\models"

# The specific folder for Llama 3.2 Vision
LOCAL_DIR = os.path.join(BASE_MODEL_DIR, "Llama-3.2-11B-Vision-Instruct")

# --- Hugging Face Details ---
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Set your HuggingFace token as an environment variable before running:
#   set HF_TOKEN=hf_your_token_here
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def main():
    # 1. Create the new folder if it doesn't exist
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"📁 Target Directory Set: {LOCAL_DIR}")
    
    print(f"🚀 Starting download of {MODEL_ID}...")
    print("⏳ This model is ~22GB. Grab a coffee, this might take a while depending on your internet.")

    # 2. Download the model
    # local_dir_use_symlinks=False forces the actual files to sit in your folder, 
    # not in the hidden C:\.cache\huggingface folder.
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False, 
            token=HF_TOKEN,
            # We ignore Original PyTorch (.bin/.pth) if safetensors exist to save bandwidth
            ignore_patterns=["*.pth", "*.bin", "*.msgpack"] 
        )
        print(f"\n🎉 SUCCESS! Model successfully downloaded and saved to:\n{LOCAL_DIR}")
        print("You are now ready to write the new captioning script!")
        
    except Exception as e:
        print(f"\n❌ ERROR downloading model: {e}")
        print("Did you paste your HF_TOKEN and accept the Meta agreement on the Hugging Face website?")

if __name__ == "__main__":
    main()