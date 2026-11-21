import sys
import io
import pandas as pd
from pathlib import Path

# Clean terminal UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MASTER_CSV = Path(r"D:\ramesh\jewelry ipa\training\dataset_1024_framing\csv\mistral_2agent_sdxl_master.csv")

TEST_FILENAMES = [
    "427cbd22-5f6a-4d22-98d7-41a0c8a7fdfa-NGO-BGBB-BTC-S-L-OX-2628546-additional-2_black_bg.png",
    "098645e5-364f-4d61-b5be-8524a69db15a-NGO-BGSB-COD-S-L-NA-25894466-additional-1_black_bg.png",
    "Candere_Earrings_Prod1_Angle11_1100x1100_80d260_black_bg.png",
    "Candere_Necklaces_Prod37_Angle3_1100x1100_46de1e_black_bg.png",
    "CC_Ladies_Rings_Prod2_Angle3_1200x1200_c52a75_clean_black_bg.png"
]

if not MASTER_CSV.exists():
    print(f"❌ Master CSV not found at: {MASTER_CSV}")
    sys.exit(1)

df = pd.read_csv(MASTER_CSV)

print("=" * 95)
print(f"📄 Master CSV Loaded: {MASTER_CSV} (Total Rows: {len(df)})")
print("🔍 Fetching Final Agent 2 SDXL Prompts & Verifying Sidecar .txt Files for 5 Sample Images...")
print("=" * 95)

found_count = 0
for idx, fname in enumerate(TEST_FILENAMES, start=1):
    matched = df[df["Image_Path"].astype(str).str.contains(fname, case=False, na=False)]
    if matched.empty:
        print(f"\n⚠️ [{idx}/5] Not found in CSV: {fname}")
        continue

    found_count += 1
    row = matched.iloc[0]
    txt_path = Path(str(row["Txt_Path"]))
    txt_exists = txt_path.exists()
    txt_content = txt_path.read_text(encoding="utf-8").strip() if txt_exists else "MISSING .TXT FILE"

    print(f"\n[{found_count}/5] 🖼️  IMAGE: {fname}")
    print(f"📁 Brand: {row['Brand_Name']} | Category: {row['Category']} | Sub-Style: {row['Sub_Style']}")
    print(f"📊 Tokens: {row['Est_CLIP_Tokens']}/77 | Words: {row['Word_Count']} | .txt Exists: {txt_exists}")
    print("-" * 95)
    print("🎯 FINAL AGENT 2 SDXL PROMPT (CSV):")
    print(row["Agent_2_SDXL_Prompt"])
    print("-" * 95)
    print(f"📝 SIDECAR .TXT CONTENT ({txt_path}):")
    print(txt_content)
    print("=" * 95)

print(f"\n🎉 Verified {found_count}/5 sample images from the completed 4,273-image run!")