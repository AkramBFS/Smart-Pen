import os
import csv

TARGET_DIR = "recordings"  # <-- change if needed

for filename in os.listdir(TARGET_DIR):
    if not filename.endswith(".csv"):
        continue

    if "_medium_" not in filename:
        continue

    old_path = os.path.join(TARGET_DIR, filename)

    # ---------- Rename file ----------
    new_filename = filename.replace("_medium_", "_good_")
    new_path = os.path.join(TARGET_DIR, new_filename)

    print(f"Renaming: {filename} -> {new_filename}")
    os.rename(old_path, new_path)

    # ---------- Update CSV contents ----------
    rows = []

    with open(new_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # header stays unchanged
            if row and row[0] == "student":
                rows.append(row)
                continue

            # safety check
            if len(row) >= 6:
                if row[4] == "medium":
                    row[4] = "good"

            rows.append(row)

    with open(new_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

print("✅ Done: all medium → good conversions completed.")
