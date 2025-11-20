import pandas as pd
import os
import shutil

# ---------- CONFIGURE THESE ----------
metadata_file = "metadata.xlsx"    # or "metadata.csv"
images_folder = "images/"          # folder where all original images exist
output_folder = "diabetic_images/" # folder to store filtered images
image_column = "image_name"        # column in metadata that contains image file names
d_column = "D"                     # column with 0/1 values
# ------------------------------------

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the metadata file
if metadata_file.endswith(".xlsx"):
    df = pd.read_excel(metadata_file)
else:
    df = pd.read_csv(metadata_file)

# Filter rows where D = 1
df_filtered = df[df[d_column] == 1]

print(f"Total images with D=1: {len(df_filtered)}")

# Copy images
count = 0
for img_name in df_filtered[image_column]:
    src_path = os.path.join(images_folder, img_name)
    dst_path = os.path.join(output_folder, img_name)

    # Copy only if file exists
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        count += 1
    else:
        print(f"⚠️ Image not found: {src_path}")

print(f"✅ Copied {count} images to '{output_folder}'")
