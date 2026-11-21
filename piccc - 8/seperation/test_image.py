import pandas as pd
from pathlib import Path
import sys
import io

# Fix terminal encoding for Windows PowerShell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

csv_path = Path(r"D:\ramesh\jewelry ipa\training\dataset_1024_framing\csv\intern_vl2_sam_newprompt.csv")

if not csv_path.exists():
    print(f"❌ CSV file not found at: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

# The 5 test image filenames
test_filenames = [
    "427cbd22-5f6a-4d22-98d7-41a0c8a7fdfa-NGO-BGBB-BTC-S-L-OX-2628546-additional-2_black_bg.png",
    "098645e5-364f-4d61-b5be-8524a69db15a-NGO-BGSB-COD-S-L-NA-25894466-additional-1_black_bg.png",
    "Candere_Earrings_Prod1_Angle11_1100x1100_80d260_black_bg.png",
    "Candere_Necklaces_Prod37_Angle3_1100x1100_46de1e_black_bg.png",
    "CC_Ladies_Rings_Prod2_Angle3_1200x1200_c52a75_clean_black_bg.png"
]

print("=" * 90)
print(f"📄 Scanning CSV: {csv_path}")
print(f"🔍 Searching for the 5 validation test images...")
print("=" * 90)

found_count = 0
for idx, fname in enumerate(test_filenames, start=1):
    matched_row = df[df['Image_Path'].str.contains(fname, case=False, na=False)]
    
    if not matched_row.empty:
        found_count += 1
        row = matched_row.iloc[0]
        print(f"\n[{found_count}/5] 🖼️  IMAGE: {fname}")
        print(f"📁 Brand: {row['Brand_Name']} | Category: {row['Category']} | Sub-Style: {row['Sub_Style']}")
        print("-" * 90)
        print(row['Full_Raw_Output'])
        print("=" * 90)
    else:
        print(f"\n⚠️ Warning: '{fname}' not found in CSV yet (Make sure the full run processed it).")

print(f"\n🎉 Done! Printed {found_count} test image captions from the CSV.")