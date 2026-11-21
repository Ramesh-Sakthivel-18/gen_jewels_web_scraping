import os
import sys
import io
import re
import time
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

# Optional Exact SDXL CLIP-L Tokenizer Loader
CLIP_TOKENIZER = None
try:
    from transformers import CLIPTokenizer
    CLIP_TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
except Exception:
    CLIP_TOKENIZER = None

# ==========================================
# 1. PATHS & 6-CSV PRIORITY CONFIGURATION
# ==========================================
MODEL_PATH_OR_DIR = Path(r"D:\ramesh\jewelry ipa\training\model\Mistral-Small-24B-Instruct-2501-GGUF")
DATASET_DIR = Path(r"D:\ramesh\jewelry ipa\training\dataset_1024_framing")
CSV_DIR = DATASET_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

MASTER_OUTPUT_CSV = CSV_DIR / "mistral_2agent_sdxl_master.csv"

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

TRIGGER_WORD = "gj_jewels"
HARD_MAX_CLIP_TOKENS = 75 if CLIP_TOKENIZER is not None else 73

SUB_STYLE_NATURAL_MAP = {
    "Bangle_Broad_Cylindrical_Kada": "broad multi-row cylindrical cuff kada bangle",
    "Bangle_Single_Band": "single-band floral cluster circular gold bangle",
    "Earring_Linear_Drop": "minimalist linear open-link ball-drop dangle earrings",
    "Earring_Compact_Stud": "compact stud earrings",
    "Earring_Traditional_Jhumka": "traditional ornate jhumka drop earrings",
    "Necklace_Modern_Minimalist": "modern minimalist fine-chain Y-lariat pendant necklace",
    "Necklace_Traditional_Ornate": "traditional ornate sculpted gold necklace",
    "Ring_Standard": "ladies gold cocktail ring",
    "Pendant_Standard": "gold pendant"
}

# ==========================================
# 2. 100% GPU OFFLOAD MODEL LOADER
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

def load_mistral_gguf_full_gpu(model_path_or_dir: Path) -> Llama:
    gguf_file = resolve_gguf_file(model_path_or_dir)
    size_gb = gguf_file.stat().st_size / (1024 ** 3)
    print("=" * 95, flush=True)
    print(f"🚀 Loading GGUF Model (FULL GPU OFFLOAD): {gguf_file.name} ({size_gb:.2f} GB)", flush=True)
    print("   -> Setting n_gpu_layers=-1 (All layers on GPU), n_ctx=8192, n_batch=512...", flush=True)

    llm = Llama(
        model_path=str(gguf_file),
        n_gpu_layers=-1,   # Strictly 100% GPU offload as requested
        n_ctx=8192,        # Full 8K context window for all 6 VLM descriptions
        n_batch=512,
        verbose=False
    )
    print("✅ Mistral-Small-24B Loaded Successfully with n_gpu_layers=-1 (100% GPU Offload)!", flush=True)
    if CLIP_TOKENIZER is not None:
        print("✅ Exact SDXL CLIPTokenizer (openai/clip-vit-large-patch14) Active! (Hard Limit: <=77 tokens)", flush=True)
    else:
        print("ℹ️ Using Conservative Regex CLIP Token Counter (Hard Limit: <=73 tokens)", flush=True)
    print("=" * 95, flush=True)
    return llm

# ==========================================
# 3. FAST O(1) CSV INDEXER & DATASET SCANNER
# ==========================================
def load_and_index_all_csvs(configs):
    indexed_sources = {}
    ordered_paths_from_p1 = []

    print("\n📂 Loading & Indexing 6 VLM Caption CSVs into memory...", flush=True)
    for cfg in configs:
        key = cfg["key"]
        p = cfg["path"]
        if not p.exists():
            print(f"   ⚠️ [MISSING] {p.name}", flush=True)
            indexed_sources[key] = {}
            continue

        df = pd.read_csv(p)
        path_col = next(
            (c for c in ["Image_Path", "image_path", "path", "file_path", "filename", "File_Name", "Image"] if c in df.columns),
            df.columns[0]
        )
        caption_col = next(
            (c for c in ["Full_Raw_Output", "caption", "Caption", "description", "Description", "output", "prompt", "text"] if c in df.columns),
            df.columns[-1]
        )

        lookup = {}
        for _, row in df.iterrows():
            raw_path_str = str(row[path_col]).strip()
            if not raw_path_str or raw_path_str.lower() == "nan":
                continue
            fname_key = Path(raw_path_str).name.lower()
            caption_text = str(row[caption_col]).strip() if pd.notna(row[caption_col]) else "N/A"
            meta = {
                m: str(row[m]).strip()
                for m in ["Brand_Name", "Category", "Sub_Style"]
                if m in df.columns and pd.notna(row[m])
            }
            lookup[fname_key] = {
                "raw_path": raw_path_str,
                "text": caption_text,
                "meta": meta
            }
            if key == "InternVL2_SAM_NewPrompt":
                ordered_paths_from_p1.append(Path(raw_path_str))

        indexed_sources[key] = lookup
        print(f"   ✅ [INDEXED] {p.name:<42} | Indexed Rows: {len(lookup):<5} | {cfg['priority_rank'][:28]}...", flush=True)

    return indexed_sources, ordered_paths_from_p1

def collect_all_dataset_images(dataset_dir: Path, p1_paths: list) -> list:
    seen = set()
    final_list = []

    for p in p1_paths:
        if p.exists():
            norm = str(p.resolve()).lower()
            if norm not in seen:
                seen.add(norm)
                final_list.append(p)

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in sorted(dataset_dir.rglob(ext)):
            if "output_with_sam" in p.parts or "csv" in p.parts:
                continue
            norm = str(p.resolve()).lower()
            if norm not in seen:
                seen.add(norm)
                final_list.append(p)

    return final_list

# ==========================================
# 4. EXACT CLIP 77-TOKEN COUNTER & ENFORCER
# ==========================================
def count_clip_tokens(prompt_str: str) -> int:
    """Uses exact CLIPTokenizer if available; otherwise uses conservative BPE regex estimator."""
    if CLIP_TOKENIZER is not None:
        return len(CLIP_TOKENIZER.encode(prompt_str, truncation=False))
    tokens = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", prompt_str)
    subword_extra = sum(1 for t in tokens if len(t) >= 9)
    return len(tokens) + subword_extra + 2

def clean_tag_phrase(phrase: str) -> str:
    p = phrase.strip(" \"'`*-\n\t.;:")
    for code_key, natural_val in SUB_STYLE_NATURAL_MAP.items():
        p = re.sub(rf"\b{re.escape(code_key)}\b", natural_val, p, flags=re.IGNORECASE)
    p = re.sub(
        r"\b(CC_Ladies_Rings|CC\s+Ladies\s+Rings|Earrings_Linear_Drop|Bangle_Single_Band|Bangle_Broad_Cylindrical_Kada|Necklace_Modern_Minimalist|Necklace_Traditional_Ornate|Earring_Compact_Stud|Earring_Traditional_Jhumka|Ring_Standard|Pendant_Standard)\b",
        "",
        p,
        flags=re.IGNORECASE
    )
    p = re.sub(r"solidblack", "solid black", p, flags=re.IGNORECASE)
    p = re.sub(r"smoothgold", "smooth gold", p, flags=re.IGNORECASE)
    p = re.sub(r"whitegold", "white gold", p, flags=re.IGNORECASE)
    p = re.sub(r"yellowgold", "yellow gold", p, flags=re.IGNORECASE)
    p = re.sub(r"rosegold", "rose gold", p, flags=re.IGNORECASE)
    p = re.sub(
        r"^(featuring|crafted\s+from|adorned\s+with|with\s+a|with|having|showing|set\s+against\s+a|set\s+on\s+a)\s+",
        "",
        p,
        flags=re.IGNORECASE
    )
    p = re.sub(r"\s+", " ", p).strip(" ,.;-")
    return p

def normalize_and_dedup_tags(prompt_str: str, fallback_dossier: str, meta_info: dict) -> list:
    cleaned = re.sub(
        r"^(\*\*|\[)?(agent\s*2|agent\s*3|final\s*prompt|sdxl\s*prompt|output)[:\]]?\*?\*?\s*",
        "",
        prompt_str.strip(),
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s+", " ", cleaned.strip(" \"'`\n\t."))

    raw_parts = [clean_tag_phrase(p) for p in re.split(r"[,;\n•]+|\.\s+", cleaned)]
    raw_parts = [p for p in raw_parts if p]

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

    if len(tags) <= 2 and fallback_dossier:
        sub_natural = SUB_STYLE_NATURAL_MAP.get(meta_info.get("Sub_Style", ""), meta_info.get("Category", "gold jewelry"))
        tags.append(sub_natural)
        dossier_lines = re.sub(r"\*\*[^*]+\*\*:?", "", fallback_dossier)
        for chunk in re.split(r"[,;\n•\-]+|\.\s+", dossier_lines):
            c_clean = clean_tag_phrase(chunk)
            if c_clean and c_clean.lower() not in seen and "black background" not in c_clean.lower():
                tags.append(c_clean)
                seen.add(c_clean.lower())

    return tags

def enforce_sub77_clip_limit(prompt_str: str, fallback_dossier: str, meta_info: dict, max_clip_tokens: int = HARD_MAX_CLIP_TOKENS) -> str:
    tags = normalize_and_dedup_tags(prompt_str, fallback_dossier, meta_info)
    bg_tag = "solid black background"

    while len(tags) > 1:
        candidate = ", ".join(tags + [bg_tag])
        candidate = re.sub(r",\s*", ", ", candidate)
        if count_clip_tokens(candidate) <= max_clip_tokens:
            return candidate

        last_words = tags[-1].split()
        if len(last_words) > 4:
            tags[-1] = " ".join(last_words[:-1]).strip(" ,;-")
        elif len(tags) > 2:
            tags.pop()
        else:
            tags[-1] = " ".join(last_words[:-1]).strip(" ,;-")
            if not tags[-1]:
                break

    final_str = ", ".join(tags + [bg_tag])
    return re.sub(r",\s*", ", ", final_str).strip()

# ==========================================
# 5. SEQUENTIAL AGENT PROMPTS (AGENT 1 -> AGENT 2 -> AGENT 3 AUDITOR)
# ==========================================
AGENT_1_SYSTEM_PROMPT = """You are AGENT 1: MASTER GEMOLOGICAL ANALYZER.
Synthesize the 6 VLM descriptions ranked PRIORITY 1 to PRIORITY 6 into a complete, factual ground-truth summary.

STRICT PRIORITY & CONFLICT RULES:
1. PRIORITY 1 (InternVL2_SAM_NewPrompt) carries 70% weight and is your ABSOLUTE GROUND TRUTH for jewelry sub-style, material lock, gemstone cuts/colors, enamel, and micro-hardware.
2. MATERIAL LOCK RULE:
   - If PRIORITY 1 states 'Lock: 100% PLAIN METAL' and its Macro section confirms bare gold (e.g., linear ball-drop earrings), the item has ZERO gemstones.
   - IMPORTANT EXCEPTION: If PRIORITY 1's own Macro section explicitly describes a small white pavé O-ring connector and pastel enamel terminal drop (as in modern minimalist Y-lariat necklaces), include those exact pavé and pastel enamel details!
   - Reject all hallucinated stones, jhumkas, or temple motifs from lower-priority CSVs (Priorities 3-6) that contradict Priority 1.
3. Use PRIORITY 2 (JoyCaption - 15% weight) for studio lighting, metal polish reflections, and framing.
4. Output ONLY 4 detailed, factual bullets (100-140 words total):
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
2. LENGTH TARGET: Write 8 to 11 short, punchy visual tag phrases separated by commas (total 34 to 44 words, strictly under 68 CLIP tokens).
3. SHORT COMMA PHRASES ONLY: Every phrase between commas must be 2 to 6 words long. Always put a space after every comma. NEVER write long sentences or use semicolons.
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
        "Convert this into 8 to 11 short comma-separated SDXL tags (34-44 words total), starting with: gj_jewels, "
    )

AGENT_3_AUDITOR_SYSTEM_PROMPT = """You are AGENT 3: SDXL 77-TOKEN PRECISION AUDITOR.
The draft SDXL prompt exceeded the 75 CLIP token limit or contained long phrases.
Rewrite and tighten the draft prompt against Agent 1's Dossier so that:
1. It starts with `gj_jewels, ` and ends with `, sharp macro studio photography, solid black background`.
2. It preserves 100% of the key jewelry silhouette, gold finish, gemstones/coral/beads/enamel (or '100% solid yellow gold, zero gemstones'), and clasp/shank hardware.
3. It removes redundant words so the total length is strictly 32 to 40 words (under 66 CLIP tokens), with every tag separated by a comma and space.
Output ONLY the tightened single-line comma-separated prompt."""

# ==========================================
# 6. DETERMINISTIC INFERENCE & FULL DATASET RUNNER
# ==========================================
def run_agent(llm: Llama, system_prompt: str, user_prompt: str, max_tokens: int = 380) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,      # Deterministic greedy decoding for maximum factual accuracy
        top_p=1.0,
        repeat_penalty=1.12,
        max_tokens=max_tokens
    )
    return response["choices"][0]["message"]["content"].strip()

def main():
    # 1. Load & O(1) Index all 6 CSVs
    indexed_csvs, p1_image_paths = load_and_index_all_csvs(VLM_CSV_CONFIG)

    # 2. Collect full dataset image paths
    all_image_paths = collect_all_dataset_images(DATASET_DIR, p1_image_paths)
    total_images = len(all_image_paths)

    print(f"\n📁 Total Dataset Images Discovered: {total_images}", flush=True)
    print(f"📄 Master CSV Output Path:         {MASTER_OUTPUT_CSV}", flush=True)
    print(f"📝 Sidecar .txt Output Mode:       Saved alongside each image (<image_stem>.txt = Agent 2 SDXL prompt only)", flush=True)

    # 3. Auto-Resume check
    results = []
    processed_map = {}
    if MASTER_OUTPUT_CSV.exists():
        try:
            existing_df = pd.read_csv(MASTER_OUTPUT_CSV)
            results = existing_df.to_dict("records")
            for row in results:
                img_p = str(row.get("Image_Path", "")).strip()
                if img_p:
                    processed_map[img_p.lower()] = row
            print(f"🔄 Resuming: {len(processed_map)} images already recorded in Master CSV.", flush=True)
        except Exception as e:
            print(f"⚠️ Could not read existing Master CSV ({e}). Starting fresh.", flush=True)

    # 4. Load Mistral-Small-24B GGUF with 100% GPU Offload (n_gpu_layers=-1)
    llm = load_mistral_gguf_full_gpu(MODEL_PATH_OR_DIR)

    start_time = time.time()
    newly_processed = 0

    for idx, img_path in enumerate(all_image_paths, start=1):
        img_str_path = str(img_path)
        img_key_lower = img_str_path.lower()
        fname = img_path.name
        fname_lower = fname.lower()

        txt_path = img_path.with_suffix(".txt")
        sub_folder = img_path.parent.name
        main_folder = img_path.parent.parent.name
        short_id = f"{main_folder}/{sub_folder}/{fname}"

        # Skip if already processed in CSV (and ensure .txt exists on disk)
        if img_key_lower in processed_map:
            existing_prompt = str(processed_map[img_key_lower].get("Agent_2_SDXL_Prompt", "")).strip()
            if existing_prompt and not txt_path.exists():
                txt_path.write_text(existing_prompt, encoding="utf-8")
            print(f"⏭️  [{idx}/{total_images}][SKIP - ALREADY DONE] {short_id}", flush=True)
            continue

        try:
            img_t0 = time.time()
            vlm_bundle = []
            primary_meta = {
                "Brand_Name": main_folder,
                "Category": sub_folder,
                "Sub_Style": "Standard"
            }

            for cfg in VLM_CSV_CONFIG:
                key = cfg["key"]
                source_dict = indexed_csvs.get(key, {})
                entry = source_dict.get(fname_lower)

                if entry:
                    if key == "InternVL2_SAM_NewPrompt" and entry["meta"]:
                        primary_meta.update(entry["meta"])
                    elif primary_meta.get("Sub_Style") == "Standard" and entry["meta"].get("Sub_Style"):
                        primary_meta["Sub_Style"] = entry["meta"]["Sub_Style"]

                    vlm_bundle.append({
                        "key": key,
                        "priority_rank": cfg["priority_rank"],
                        "rule": cfg["rule"],
                        "found": True,
                        "text": entry["text"]
                    })
                else:
                    vlm_bundle.append({
                        "key": key,
                        "priority_rank": cfg["priority_rank"],
                        "rule": cfg["rule"],
                        "found": False,
                        "text": "N/A"
                    })

            matched_count = sum(1 for b in vlm_bundle if b["found"])
            print("-" * 95, flush=True)
            print(
                f"🖼️  [{idx}/{total_images}] START IMAGE: {short_id}\n"
                f"   ├─ Brand: {primary_meta.get('Brand_Name')} | Category: {primary_meta.get('Category')} | "
                f"Sub_Style: {primary_meta.get('Sub_Style')} | Matched CSVs: {matched_count}/6",
                flush=True
            )

            # STEP 1: Run Agent 1 (Multi-VLM Ground-Truth Synthesis)
            print(f"   ├─ 🧠 [AGENT 1] Synthesizing 6 VLM inputs (Priority 1 = InternVL2_SAM)...", end="", flush=True)
            t_a1 = time.time()
            agent_1_output = run_agent(
                llm,
                system_prompt=AGENT_1_SYSTEM_PROMPT,
                user_prompt=build_agent_1_user_message(fname, primary_meta, vlm_bundle),
                max_tokens=380
            )
            a1_sec = time.time() - t_a1
            a1_preview = " ".join(agent_1_output.splitlines())[:110]
            print(f" Done ({a1_sec:.1f}s)", flush=True)
            print(f"   │     ↳ Agent 1 Preview: {a1_preview}...", flush=True)

            # STEP 2: Run Agent 2 (Sub-77-Token SDXL Prompt Distiller)
            print(f"   ├─ 🎯 [AGENT 2] Distilling sub-77-token SDXL prompt from Agent 1...", end="", flush=True)
            t_a2 = time.time()
            raw_agent_2_output = run_agent(
                llm,
                system_prompt=AGENT_2_SYSTEM_PROMPT,
                user_prompt=build_agent_2_user_message(agent_1_output, primary_meta),
                max_tokens=150
            )
            a2_sec = time.time() - t_a2

            # Check if raw Agent 2 draft exceeds token limit; if so, invoke Agent 3 Auditor to intelligently tighten it
            pre_tags = normalize_and_dedup_tags(raw_agent_2_output, agent_1_output, primary_meta)
            pre_candidate = re.sub(r",\s*", ", ", ", ".join(pre_tags + ["solid black background"]))
            pre_tokens = count_clip_tokens(pre_candidate)

            if pre_tokens > HARD_MAX_CLIP_TOKENS:
                print(f" Draft={pre_tokens} tokens -> Running Agent 3 Auditor...", end="", flush=True)
                auditor_user_msg = (
                    f"AGENT 1 DOSSIER:\n{agent_1_output}\n\n"
                    f"OVER-LENGTH DRAFT ({pre_tokens} tokens):\n{pre_candidate}\n\n"
                    "Tighten this prompt to 32-40 words while preserving all structural, stone, and hardware facts:"
                )
                raw_agent_2_output = run_agent(
                    llm,
                    system_prompt=AGENT_3_AUDITOR_SYSTEM_PROMPT,
                    user_prompt=auditor_user_msg,
                    max_tokens=130
                )
                a2_sec = time.time() - t_a2

            final_sdxl_prompt = enforce_sub77_clip_limit(
                raw_agent_2_output,
                fallback_dossier=agent_1_output,
                meta_info=primary_meta,
                max_clip_tokens=HARD_MAX_CLIP_TOKENS
            )

            est_tokens = count_clip_tokens(final_sdxl_prompt)
            word_count = len(final_sdxl_prompt.split())
            print(f" Done ({a2_sec:.1f}s | {est_tokens}/77 CLIP Tokens | {word_count} Words)", flush=True)
            print(f"   │     ↳ SDXL Prompt: {final_sdxl_prompt}", flush=True)

            # SAVE 1: Write ONLY Agent 2 SDXL prompt to the image's sidecar .txt file
            txt_path.write_text(final_sdxl_prompt, encoding="utf-8")
            print(f"   ├─ 📝 [SAVED .TXT] {txt_path}", flush=True)

            # SAVE 2: Append both Agent 1 and Agent 2 outputs to the Master CSV
            row_record = {
                "Image_Path": img_str_path,
                "Txt_Path": str(txt_path),
                "Brand_Name": primary_meta.get("Brand_Name", main_folder),
                "Category": primary_meta.get("Category", sub_folder),
                "Sub_Style": primary_meta.get("Sub_Style", "Standard"),
                "Agent_1_Synthesis": agent_1_output,
                "Agent_2_SDXL_Prompt": final_sdxl_prompt,
                "Est_CLIP_Tokens": est_tokens,
                "Word_Count": word_count
            }
            results.append(row_record)
            processed_map[img_key_lower] = row_record
            newly_processed += 1

            pd.DataFrame(results).to_csv(MASTER_OUTPUT_CSV, index=False, quoting=1, encoding="utf-8")

            total_img_sec = time.time() - img_t0
            elapsed = time.time() - start_time
            avg_sec = elapsed / max(1, newly_processed)
            print(
                f"   └─ 💾 [SAVED CSV]  Row #{len(results)} saved to {MASTER_OUTPUT_CSV.name} "
                f"(Image Time: {total_img_sec:.1f}s | Avg: {avg_sec:.1f}s/img)",
                flush=True
            )

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user! Progress up to the last completed image is safely saved in CSV and .txt.", flush=True)
            break
        except Exception as e:
            print(f"\n   └─ ❌ [ERROR] {short_id} | {e}", flush=True)

    print("=" * 95, flush=True)
    print(f"🎉 RUN COMPLETE! Total Saved: {len(results)}/{total_images} images.", flush=True)
    print(f"📄 Master CSV (Agent 1 + Agent 2): {MASTER_OUTPUT_CSV}", flush=True)
    print(f"📝 Sidecar .txt Files (Agent 2):   Saved directly inside each dataset subfolder.", flush=True)
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()