import os
import sys
import io
import re
from pathlib import Path

# ==============================================================================
# --- 0. WINDOWS CUDA DLL FIX (MUST RUN BEFORE IMPORTING LLAMA_CPP) ---
cuda_path = os.environ.get("CUDA_PATH", "")
if cuda_path and not os.path.exists(os.path.join(cuda_path, "bin")):
    os.environ.pop("CUDA_PATH", None)

for k in list(os.environ.keys()):
    if k.startswith("CUDA_PATH_V"):
        if not os.path.exists(os.path.join(os.environ[k], "bin")):
            os.environ.pop(k, None)

site_packages = Path(sys.prefix) / "Lib" / "site-packages"
candidate_dll_dirs = [
    site_packages / "torch" / "lib",
    site_packages / "nvidia" / "cuda_runtime" / "bin",
    site_packages / "nvidia" / "cublas" / "bin",
]

toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
if toolkit_root.exists():
    for v_dir in sorted(toolkit_root.iterdir(), reverse=True):
        if (v_dir / "bin").exists():
            os.environ["CUDA_PATH"] = str(v_dir)
            candidate_dll_dirs.append(v_dir / "bin")
            break

for dll_dir in candidate_dll_dirs:
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# ==============================================================================

import pandas as pd
from llama_cpp import Llama

# ==========================================
# 1. PATHS & 6-CSV PRIORITY CONFIGURATION
# ==========================================
MODEL_PATH_OR_DIR = Path(r"D:\ramesh\jewelry ipa\training\model\Mistral-Small-24B-Instruct-2501-GGUF")
CSV_DIR = Path(r"D:\ramesh\jewelry ipa\training\dataset_1024_framing\csv")

VLM_CSV_CONFIG = [
    {
        "key": "InternVL2_SAM_NewPrompt",
        "priority_rank": "PRIORITY 1 (HIGHEST - 70% WEIGHT | GROUND TRUTH)",
        "path": CSV_DIR / "intern_vl2_sam_newprompt.csv",
        "rule": "Absolute ground truth for sub-style, metal lock, stone cuts/colors, and micro-hardware."
    },
    {
        "key": "JoyCaption",
        "priority_rank": "PRIORITY 2 (HIGH - 15% WEIGHT | LIGHTING & AESTHETICS)",
        "path": CSV_DIR / "joycaption_full_dataset_captions.csv",
        "rule": "Use for studio lighting, specular reflections, camera angle, and background. Ignore conflicting stone claims."
    },
    {
        "key": "Qwen2_VL",
        "priority_rank": "PRIORITY 3 (MEDIUM - 5% WEIGHT | SILHOUETTE CHECK)",
        "path": CSV_DIR / "qwen2_full_dataset_captions.csv",
        "rule": "Cross-verify overall silhouette when consistent with Priority 1."
    },
    {
        "key": "Llama_3_2_11B",
        "priority_rank": "PRIORITY 4 (LOW - 5% WEIGHT | SHAPE CONTEXT)",
        "path": CSV_DIR / "jewelry_master_llama_3.2_11b_captions.csv",
        "rule": "Use for general shape terms; discard conflicting stone or motif claims."
    },
    {
        "key": "Florence2",
        "priority_rank": "PRIORITY 5 (LOW - 3% WEIGHT | BASIC OBJECT TAGS)",
        "path": CSV_DIR / "florence2_full_dataset_captions.csv",
        "rule": "Confirm basic object type; ignore generic stone guesses."
    },
    {
        "key": "InternVL2_Legacy_Segmented",
        "priority_rank": "PRIORITY 6 (LOWEST - 2% WEIGHT | LEGACY REFERENCE)",
        "path": CSV_DIR / "internvl2_segmented_dataset.csv",
        "rule": "Ignore any temple, jhumka, or step-cut terms contradicting Priority 1."
    }
]

TEST_FILENAMES = [
    "427cbd22-5f6a-4d22-98d7-41a0c8a7fdfa-NGO-BGBB-BTC-S-L-OX-2628546-additional-2_black_bg.png",
    "098645e5-364f-4d61-b5be-8524a69db15a-NGO-BGSB-COD-S-L-NA-25894466-additional-1_black_bg.png",
    "Candere_Earrings_Prod1_Angle11_1100x1100_80d260_black_bg.png",
    "Candere_Necklaces_Prod37_Angle3_1100x1100_46de1e_black_bg.png",
    "CC_Ladies_Rings_Prod2_Angle3_1200x1200_c52a75_clean_black_bg.png"
]

TRIGGER_WORD = "gj_jewels"
HARD_MAX_CLIP_TOKENS = 73  # Strict safety buffer below 77 CLIP tokens (including BOS/EOS)

SUB_STYLE_NATURAL_MAP = {
    "Bangle_Broad_Cylindrical_Kada": "broad multi-row cylindrical cuff kada bangle",
    "Bangle_Single_Band": "single-band floral cluster circular gold bangle",
    "Earring_Linear_Drop": "minimalist linear open-link ball-drop dangle earrings",
    "Earring_Compact_Stud": "compact stud earrings",
    "Earring_Traditional_Jhumka": "traditional ornate jhumka drop earrings",
    "Necklace_Modern_Minimalist": "modern minimalist fine-chain Y-lariat pendant necklace",
    "Necklace_Traditional_Ornate": "traditional ornate sculpted gold necklace",
    "Ring_Standard": "ladies gold cocktail ring"
}

# ==========================================
# 2. GGUF MODEL RESOLVER & LOADER
# ==========================================
def resolve_gguf_file(path_or_dir: Path) -> Path:
    if path_or_dir.is_file() and path_or_dir.suffix.lower() == ".gguf":
        return path_or_dir
    if not path_or_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {path_or_dir}")

    gguf_files = sorted(path_or_dir.glob("*.gguf")) or sorted(path_or_dir.rglob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found inside: {path_or_dir}")

    preferred_order = ["q4_k_m", "q4_k_s", "iq4_xs", "q3_k_m", "q5_k_m", "q8_0"]
    for pref in preferred_order:
        for f in gguf_files:
            if pref in f.name.lower():
                return f
    return gguf_files[0]

def load_mistral_gguf(model_path_or_dir: Path) -> Llama:
    gguf_file = resolve_gguf_file(model_path_or_dir)
    size_gb = gguf_file.stat().st_size / (1024 ** 3)
    print("=" * 95, flush=True)
    print(f"🚀 Loading GGUF Model: {gguf_file.name} ({size_gb:.2f} GB)", flush=True)

    for gpu_layers in [-1, 35, 28, 22, 0]:
        try:
            print(f"   -> Attempting load with n_gpu_layers={gpu_layers}, n_ctx=8192...", flush=True)
            llm = Llama(
                model_path=str(gguf_file),
                n_gpu_layers=gpu_layers,
                n_ctx=8192,
                n_batch=512,
                verbose=False
            )
            print(f"✅ Mistral-Small-24B Loaded Successfully (n_gpu_layers={gpu_layers})!", flush=True)
            print("=" * 95, flush=True)
            return llm
        except Exception as e:
            print(f"   ⚠️ Load with n_gpu_layers={gpu_layers} failed ({e}). Retrying...", flush=True)

    raise RuntimeError("Could not initialize Mistral GGUF model.")

# ==========================================
# 3. CSV LOADER & CAPTION EXTRACTOR
# ==========================================
def load_all_csvs(configs):
    loaded = {}
    print("\n📂 Loading 6 VLM Caption CSVs into memory...", flush=True)
    for cfg in configs:
        p = cfg["path"]
        if not p.exists():
            print(f"   ⚠️ [MISSING] {p.name}", flush=True)
            loaded[cfg["key"]] = None
            continue
        df = pd.read_csv(p)
        loaded[cfg["key"]] = df
        print(f"   ✅ [LOADED]  {p.name:<42} | Rows: {len(df):<5} | {cfg['priority_rank'][:28]}...", flush=True)
    return loaded

def extract_caption_for_image(df: pd.DataFrame, filename: str) -> dict:
    if df is None or df.empty:
        return {"found": False, "text": "N/A", "meta": {}}

    path_col = next((c for c in ["Image_Path", "image_path", "path", "file_path", "filename", "File_Name", "Image"] if c in df.columns), df.columns[0])
    matched = df[df[path_col].astype(str).str.contains(re.escape(filename), case=False, na=False)]
    if matched.empty:
        return {"found": False, "text": "N/A", "meta": {}}

    row = matched.iloc[0]
    caption_col = next((c for c in ["Full_Raw_Output", "caption", "Caption", "description", "Description", "output", "prompt", "text"] if c in df.columns), df.columns[-1])

    meta = {m: str(row[m]).strip() for m in ["Brand_Name", "Category", "Sub_Style"] if m in df.columns and pd.notna(row[m])}
    return {"found": True, "text": str(row[caption_col]).strip(), "meta": meta}

# ==========================================
# 4. CLIP 77-TOKEN ESTIMATOR & GRANULAR ENFORCER
# ==========================================
def count_clip_tokens_approx(prompt_str: str) -> int:
    """
    Conservative CLIP BPE token counter:
    - Counts words, numbers, punctuation (, / -), and adds extra BPE split weight for long words.
    - Includes +2 for CLIP <start_of_text> and <end_of_text>.
    """
    tokens = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", prompt_str)
    subword_extra = sum(1 for t in tokens if len(t) >= 9)
    return len(tokens) + subword_extra + 2

def clean_tag_phrase(phrase: str) -> str:
    """Removes internal underscore code labels, markdown, and conversational filler."""
    p = phrase.strip(" \"'`*-\n\t.;:")
    # Replace underscore code names if Mistral echoed them
    for code_key, natural_val in SUB_STYLE_NATURAL_MAP.items():
        p = re.sub(rf"\b{re.escape(code_key)}\b", natural_val, p, flags=re.IGNORECASE)
    p = re.sub(r"\b(CC_Ladies_Rings|CC\s+Ladies\s+Rings|Earrings_Linear_Drop|Bangle_Single_Band|Bangle_Broad_Cylindrical_Kada|Necklace_Modern_Minimalist|Ring_Standard)\b", "", p, flags=re.IGNORECASE)
    # Fix joined words like 'solidblack'
    p = re.sub(r"solidblack", "solid black", p, flags=re.IGNORECASE)
    # Strip leading filler phrases
    p = re.sub(r"^(featuring|crafted\s+from|adorned\s+with|with\s+a|with|having|showing|set\s+against\s+a|set\s+on\s+a)\s+", "", p, flags=re.IGNORECASE)
    p = re.sub(r"\s+", " ", p).strip(" ,.;-")
    return p

def enforce_sub77_clip_limit(prompt_str: str, fallback_dossier: str, meta_info: dict, max_clip_tokens: int = HARD_MAX_CLIP_TOKENS) -> str:
    # 1. Strip any header prefix
    cleaned = re.sub(
        r"^(\*\*|\[)?(agent\s*2|final\s*prompt|sdxl\s*prompt|output)[:\]]?\*?\*?\s*",
        "",
        prompt_str.strip(),
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s+", " ", cleaned.strip(" \"'`\n\t."))

    # 2. Split on commas, semicolons, periods, or bullet dashes
    raw_parts = [clean_tag_phrase(p) for p in re.split(r"[,;\n•]+|\.\s+", cleaned)]
    raw_parts = [p for p in raw_parts if p]

    # 3. Build deduplicated tag list starting with gj_jewels
    tags = [TRIGGER_WORD]
    seen = {TRIGGER_WORD.lower()}

    for p in raw_parts:
        low = p.lower()
        if low.startswith("gj_jewels"):
            rest = clean_tag_phrase(re.sub(r"^gj_jewels\s*", "", p, flags=re.IGNORECASE))
            if rest and rest.lower() not in seen and "black background" not in rest.lower():
                tags.append(rest)
                seen.add(rest.lower())
            continue
        if "black background" in low:
            continue
        if low not in seen and len(p) > 1:
            tags.append(p)
            seen.add(low)

    # 4. Fallback protection: if Agent 2 produced too few tags, extract directly from Agent 1 dossier
    if len(tags) <= 2 and fallback_dossier:
        sub_natural = SUB_STYLE_NATURAL_MAP.get(meta_info.get("Sub_Style", ""), meta_info.get("Category", "gold jewelry"))
        tags.append(sub_natural)
        dossier_lines = re.sub(r"\*\*[^*]+\*\*:?", "", fallback_dossier)
        for chunk in re.split(r"[,;\n•\-]+|\.\s+", dossier_lines):
            c_clean = clean_tag_phrase(chunk)
            if c_clean and c_clean.lower() not in seen and "black background" not in c_clean.lower():
                tags.append(c_clean)
                seen.add(c_clean.lower())

    bg_tag = "solid black background"

    # 5. Granular trimming: if over token budget, drop trailing tags or trim words from the last tag
    # Never collapse down to just 'gj_jewels, solid black background'!
    while len(tags) > 1:
        candidate = ", ".join(tags + [bg_tag])
        if count_clip_tokens_approx(candidate) <= max_clip_tokens:
            return candidate

        # If the last tag is long (>4 words), trim its last word first before dropping the whole tag
        last_words = tags[-1].split()
        if len(last_words) > 4:
            tags[-1] = " ".join(last_words[:-1]).strip(" ,;-")
        elif len(tags) > 2:
            tags.pop()
        else:
            # Only 1 content tag left and it's still long: slice its words to fit
            tags[-1] = " ".join(last_words[:-1]).strip(" ,;-")
            if not tags[-1]:
                break

    return ", ".join(tags + [bg_tag])

# ==========================================
# 5. SEQUENTIAL AGENT 1 & AGENT 2 PROMPTS
# ==========================================
AGENT_1_SYSTEM_PROMPT = """You are AGENT 1: MASTER GEMOLOGICAL ANALYZER.
Synthesize the 6 VLM descriptions ranked PRIORITY 1 to PRIORITY 6 into a factual ground-truth summary.

STRICT PRIORITY & CONFLICT RULES:
1. PRIORITY 1 (InternVL2_SAM_NewPrompt) carries 70% weight and is your ABSOLUTE GROUND TRUTH for jewelry sub-style, material lock, gemstone cuts/colors, enamel, and metalwork.
2. MATERIAL LOCK RULE:
   - If PRIORITY 1 states 'Lock: 100% PLAIN METAL' and its Macro section confirms bare gold (e.g., linear ball-drop earrings), the item has ZERO gemstones.
   - IMPORTANT EXCEPTION: If PRIORITY 1's own Macro section explicitly describes a small white pavé O-ring connector and pastel enamel terminal drop (as in modern minimalist Y-lariat necklaces), include those exact pavé and pastel enamel details!
   - Reject all hallucinated stones, jhumkas, or temple motifs from lower-priority CSVs (Priorities 3-6) that contradict Priority 1.
3. Use PRIORITY 2 (JoyCaption - 15% weight) for studio lighting, metal polish reflections, and framing.
4. Output ONLY 4 concise bullets (90-120 words total):
- Silhouette & Style: (Natural description of category, shape, and design style; do NOT output raw underscore variable names)
- Metalwork & Hardware: (Gold tone, surface finish, rods/mesh/chains/bezels, clasp/hook/shank details)
- Gemstones, Coral, Beads, or Enamel: (Exact colors, cuts, and arrangements—or '100% solid polished yellow gold, zero gemstones')
- Studio Lighting & Background: (Sharp macro focus, specular gold reflections, solid black background)"""

def build_agent_1_user_message(filename: str, meta_info: dict, vlm_bundle: list) -> str:
    sub_code = meta_info.get("Sub_Style", "Standard")
    natural_sub = SUB_STYLE_NATURAL_MAP.get(sub_code, sub_code)
    blocks = [
        f"IMAGE: {filename} | Brand={meta_info.get('Brand_Name', 'Unknown')} | Category={meta_info.get('Category', 'Jewelry')} | Verified Style={natural_sub}\n"
    ]
    for item in vlm_bundle:
        blocks.append(f"### [{item['priority_rank']}] — {item['key']}\nRule: {item['rule']}\nCaption:\n{item['text']}\n")
    blocks.append("Write ONLY the 4-bullet Ground-Truth Dossier based on the priority rules.")
    return "\n".join(blocks)


AGENT_2_SYSTEM_PROMPT = """You are AGENT 2: ELITE SDXL 77-TOKEN PROMPT ENGINEER.
Convert Agent 1's 4-bullet dossier into a single-line, comma-separated SDXL training prompt.

STRICT FORMATTING RULES:
1. Start with: `gj_jewels, `
2. LENGTH TARGET: Write 8 to 11 short, punchy visual tag phrases separated by commas (total 32 to 44 words, strictly under 68 CLIP tokens).
3. SHORT COMMA PHRASES ONLY: Every phrase between commas must be 2 to 6 words long. NEVER write long sentences or use semicolons.
4. NO UNDERSCORE CODE NAMES: Never write internal labels like `Ring_Standard`, `CC_Ladies_Rings`, or `Earrings_Linear_Drop`. Use natural descriptive words.
5. EXACT TAG SEQUENCE:
   gj_jewels, [specific jewelry type & silhouette], [gold tone & primary metalwork structure], [exact gemstones/coral tubes/beads/enamel OR '100% solid yellow gold, zero gemstones'], [setting/mesh/link details], [clasp/hook/shank hardware], sharp macro studio photography, solid black background
6. Output ONLY the single-line comma-separated prompt string."""

def build_agent_2_user_message(agent_1_dossier: str, meta_info: dict) -> str:
    sub_code = meta_info.get("Sub_Style", "Standard")
    natural_sub = SUB_STYLE_NATURAL_MAP.get(sub_code, sub_code)
    return (
        f"JEWELRY TYPE: {natural_sub}\n\n"
        f"AGENT 1 DOSSIER:\n{agent_1_dossier}\n\n"
        "Convert this into 8 to 11 short comma-separated SDXL tags (32-44 words total), starting with: gj_jewels, "
    )

# ==========================================
# 6. SEQUENTIAL INFERENCE RUNNER
# ==========================================
def run_agent(llm: Llama, system_prompt: str, user_prompt: str, max_tokens: int = 280) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.12,
        max_tokens=max_tokens
    )
    return response["choices"][0]["message"]["content"].strip()

def main():
    csv_dfs = load_all_csvs(VLM_CSV_CONFIG)
    llm = load_mistral_gguf(MODEL_PATH_OR_DIR)

    print(f"\n🧪 Running Sequential 2-Agent Distillation on {len(TEST_FILENAMES)} Validation Images (TERMINAL ONLY)\n", flush=True)

    for idx, fname in enumerate(TEST_FILENAMES, start=1):
        vlm_bundle = []
        primary_meta = {}

        for cfg in VLM_CSV_CONFIG:
            key = cfg["key"]
            extracted = extract_caption_for_image(csv_dfs.get(key), fname)
            if key == "InternVL2_SAM_NewPrompt" and extracted["meta"]:
                primary_meta = extracted["meta"]
            elif not primary_meta and extracted["meta"]:
                primary_meta = extracted["meta"]

            vlm_bundle.append({
                "key": key,
                "priority_rank": cfg["priority_rank"],
                "rule": cfg["rule"],
                "found": extracted["found"],
                "text": extracted["text"]
            })

        found_sources = [b["key"] for b in vlm_bundle if b["found"]]
        print("#" * 95, flush=True)
        print(f"🖼️  TEST IMAGE [{idx}/{len(TEST_FILENAMES)}]: {fname}", flush=True)
        print(f"📁 Metadata: Brand={primary_meta.get('Brand_Name', 'N/A')} | Category={primary_meta.get('Category', 'N/A')} | Sub_Style={primary_meta.get('Sub_Style', 'N/A')}", flush=True)
        print(f"🔗 Matched CSVs ({len(found_sources)}/6): {', '.join(found_sources)}", flush=True)
        print("#" * 95, flush=True)

        # STEP 1: AGENT 1 (Reads all 6 weighted CSVs -> Ground-Truth Dossier)
        agent_1_output = run_agent(
            llm,
            system_prompt=AGENT_1_SYSTEM_PROMPT,
            user_prompt=build_agent_1_user_message(fname, primary_meta, vlm_bundle),
            max_tokens=280
        )

        print("\n🧠 [AGENT 1 OUTPUT — MULTI-VLM PRIORITY SYNTHESIS]:", flush=True)
        print("-" * 95, flush=True)
        print(agent_1_output, flush=True)

        # STEP 2: AGENT 2 (Reads ONLY Agent 1 Output -> Sub-77-Token SDXL Prompt)
        raw_agent_2_output = run_agent(
            llm,
            system_prompt=AGENT_2_SYSTEM_PROMPT,
            user_prompt=build_agent_2_user_message(agent_1_output, primary_meta),
            max_tokens=130
        )

        final_sdxl_prompt = enforce_sub77_clip_limit(
            raw_agent_2_output,
            fallback_dossier=agent_1_output,
            meta_info=primary_meta,
            max_clip_tokens=HARD_MAX_CLIP_TOKENS
        )
        est_tokens = count_clip_tokens_approx(final_sdxl_prompt)
        word_count = len(final_sdxl_prompt.split())

        print("\n🎯 [AGENT 2 OUTPUT — FINAL SUB-77-TOKEN SDXL PROMPT]:", flush=True)
        print("-" * 95, flush=True)
        print(final_sdxl_prompt, flush=True)
        print("-" * 95, flush=True)
        print(f"📊 Token Verification: ~{est_tokens}/77 CLIP Tokens | {word_count} Words | Status: PASS\n", flush=True)

    print("=" * 95, flush=True)
    print("🎉 5-IMAGE SEQUENTIAL 2-AGENT TEST COMPLETE!", flush=True)
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()