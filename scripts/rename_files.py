import os

# ─────────────────────────────────────────────
# Renames all files in RIAWELC folder
# Replaces [ and ] with underscores
# Example: bam5_Img2_A80_S5_[3][10].png
#       → bam5_Img2_A80_S5__3__10_.png
# ─────────────────────────────────────────────

RIAWELC_DIR = r"C:\Users\Sam\Desktop\weld-defect-detection\data\reviewed\RIAWELC"

renamed_count = 0
skipped_count = 0

for root, dirs, files in os.walk(RIAWELC_DIR):
    for fname in files:
        if "[" in fname or "]" in fname:
            old_path = os.path.join(root, fname)
            new_name = fname.replace("[", "_").replace("]", "_")
            new_path = os.path.join(root, new_name)

            os.rename(old_path, new_path)
            renamed_count += 1
        else:
            skipped_count += 1

print(f"Done.")
print(f"Renamed : {renamed_count} files")
print(f"Skipped : {skipped_count} files (no brackets)")
