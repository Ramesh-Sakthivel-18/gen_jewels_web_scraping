import sys
import io
import pandas as pd
from pathlib import Path
from transformers import CLIPTokenizer

# Clean UTF-8 terminal output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ==========================================
# 1. PATHS & HARD TOKEN LIMIT
# ==========================================
LOCAL_CLIP_DIR = Path(r"D:\ramesh\jewelry ipa\training\model\clip_vision")
MASTER_CSV = Path(r"D:\ramesh\jewelry ipa\training\dataset_1024_framing\csv\mistral_2agent_sdxl_master.csv")
HARD_LIMIT = 77  # Exact SDXL CLIP max_length (includes <|startoftext|> and <|endoftext|>)

# ==========================================
# 2. LOAD LOCAL CLIP TOKENIZER (CPU ONLY)
# ==========================================
def load_local_clip_tokenizer(model_dir: Path) -> CLIPTokenizer:
    if not model_dir.exists():
        raise FileNotFoundError(f"Local CLIP folder not found: {model_dir}")
    try:
        return CLIPTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    except Exception:
        return CLIPTokenizer(
            vocab_file=str(model_dir / "vocab.json"),
            merges_file=str(model_dir / "merges.txt")
        )

def count_exact_clip_tokens(tokenizer: CLIPTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, truncation=False))

# ==========================================
# 3. AUDIT ALL 4,273 ROWS (CSV + .TXT FILES)
# ==========================================
def main():
    print("=" * 95, flush=True)
    print(f"⏳ Loading Local CLIP Tokenizer from: {LOCAL_CLIP_DIR}", flush=True)
    tokenizer = load_local_clip_tokenizer(LOCAL_CLIP_DIR)
    print(f"✅ Local CLIP Tokenizer Loaded (Vocab Size: {len(tokenizer)} | Hard Limit: <= {HARD_LIMIT} tokens)", flush=True)
    print("=" * 95, flush=True)

    if not MASTER_CSV.exists():
        print(f"❌ Master CSV not found: {MASTER_CSV}")
        sys.exit(1)

    df = pd.read_csv(MASTER_CSV)
    total_rows = len(df)

    csv_over_77 = []
    txt_over_77 = []
    missing_txt = []
    mismatched_txt = []
    csv_token_counts = []
    txt_token_counts = []

    for idx, row in df.iterrows():
        img_path = str(row["Image_Path"]).strip()
        txt_path_str = str(row["Txt_Path"]).strip() if "Txt_Path" in df.columns and pd.notna(row["Txt_Path"]) else str(Path(img_path).with_suffix(".txt"))
        txt_path = Path(txt_path_str)

        csv_prompt = str(row["Agent_2_SDXL_Prompt"]).strip() if pd.notna(row["Agent_2_SDXL_Prompt"]) else ""
        csv_tokens = count_exact_clip_tokens(tokenizer, csv_prompt)
        csv_token_counts.append(csv_tokens)

        if csv_tokens > HARD_LIMIT:
            csv_over_77.append((idx + 1, img_path, txt_path_str, csv_tokens, csv_prompt))

        if not txt_path.exists():
            missing_txt.append((idx + 1, img_path, txt_path_str))
            continue

        txt_prompt = txt_path.read_text(encoding="utf-8").strip()
        txt_tokens = count_exact_clip_tokens(tokenizer, txt_prompt)
        txt_token_counts.append(txt_tokens)

        if txt_tokens > HARD_LIMIT:
            txt_over_77.append((idx + 1, img_path, txt_path_str, txt_tokens, txt_prompt))

        if csv_prompt != txt_prompt:
            mismatched_txt.append((idx + 1, img_path, txt_path_str, csv_tokens, txt_tokens))

    # ==========================================
    # 4. TERMINAL REPORT
    # ==========================================
    print(f"\n📊 AUDIT SUMMARY FOR {total_rows} IMAGES:", flush=True)
    print("-" * 95, flush=True)
    print(f"   • CSV Prompts Checked:       {len(csv_token_counts)}/{total_rows}")
    print(f"   • CSV Token Range:           Min={min(csv_token_counts)} | Avg={sum(csv_token_counts)/len(csv_token_counts):.1f} | Max={max(csv_token_counts)} / {HARD_LIMIT}")
    print(f"   • CSV Over-77 Violations:    {len(csv_over_77)}")
    print("-" * 95, flush=True)
    print(f"   • .txt Files Checked:        {len(txt_token_counts)}/{total_rows}")
    if txt_token_counts:
        print(f"   • .txt Token Range:          Min={min(txt_token_counts)} | Avg={sum(txt_token_counts)/len(txt_token_counts):.1f} | Max={max(txt_token_counts)} / {HARD_LIMIT}")
    print(f"   • .txt Over-77 Violations:   {len(txt_over_77)}")
    print(f"   • Missing .txt Files:        {len(missing_txt)}")
    print(f"   • CSV vs .txt Mismatches:    {len(mismatched_txt)}")
    print("=" * 95, flush=True)

    if csv_over_77:
        print(f"\n🚨 CSV PROMPTS EXCEEDING {HARD_LIMIT} TOKENS ({len(csv_over_77)}):", flush=True)
        for row_num, img_p, txt_p, tok_cnt, prm in csv_over_77:
            print(f"   [Row {row_num}] Tokens: {tok_cnt}/77 | Image: {img_p}", flush=True)
            print(f"      ↳ Prompt: {prm}", flush=True)

    if txt_over_77:
        print(f"\n🚨 .TXT FILES EXCEEDING {HARD_LIMIT} TOKENS ({len(txt_over_77)}):", flush=True)
        for row_num, img_p, txt_p, tok_cnt, prm in txt_over_77:
            print(f"   [Row {row_num}] Tokens: {tok_cnt}/77 | Image: {img_p}", flush=True)
            print(f"      ↳ .txt File: {txt_p}", flush=True)
            print(f"      ↳ Prompt:    {prm}", flush=True)

    if missing_txt:
        print(f"\n⚠️ MISSING .TXT FILES ({len(missing_txt)}):", flush=True)
        for row_num, img_p, txt_p in missing_txt:
            print(f"   [Row {row_num}] Image: {img_p} -> Missing: {txt_p}", flush=True)

    if mismatched_txt:
        print(f"\n⚠️ CSV vs .TXT CONTENT MISMATCHES ({len(mismatched_txt)}):", flush=True)
        for row_num, img_p, txt_p, c_tok, t_tok in mismatched_txt:
            print(f"   [Row {row_num}] CSV={c_tok} tok vs .txt={t_tok} tok | Image: {img_p}", flush=True)

    if not csv_over_77 and not txt_over_77 and not missing_txt and not mismatched_txt:
        print("\n🎉 100% PASS! Every single CSV row and .txt file is strictly <= 77 CLIP tokens and identical.", flush=True)

if __name__ == "__main__":
    main()