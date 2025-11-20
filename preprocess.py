import cv2
import numpy as np
import os
import glob
import csv

# --- CONFIGURATION ---

# 1. Set your input directory.
#    This is the "drive folder" you mentioned.
#    Create a folder named 'fundus_images' in the same directory as this script
#    and place all your images (or folders of images) inside it.
INPUT_DIR = 'fundus_images'

# 2. Set your output directory.
#    This script will create a folder named 'processed_images' for you.
OUTPUT_DIR = 'processed_images_training'

# 3. Set your log file name.
#    A CSV file will be created to log the results of each operation.
LOG_FILE = 'processing_log.csv'

# 4. Set the target dimensions for your final images.
#    (512, 512) is a good default for many models.
TARGET_SIZE = (512, 512)

# 5. Set the image extensions you want to look for.
IMG_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')

# --- END CONFIGURATION ---


def preprocess_image(filepath, output_sub_dir, target_size):
    """
    Applies the full preprocessing pipeline to a single fundus image.
    1. Reads image
    2. Crops and masks the circular retina region
    3. Applies CLAHE to the L-channel of the LAB color space
    4. Resizes to target dimensions
    5. Saves the processed image
    """
    try:
        # 1. Read the image
        img = cv2.imread(filepath)
        if img is None:
            return False, None, "Could not read image file"

        # 2. Crop and Mask Retina Region
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to create a binary mask.
        # We use a low threshold (10) to separate the black background from the
        # illuminated retina area.
        _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None, "No contours found (is this a valid fundus image?)"
            
        # Find the largest contour (which should be the retina)
        c = max(contours, key=cv2.contourArea)
        
        # Get the bounding box (x, y, width, height) of the largest contour
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Crop the original image to this bounding box
        img_cropped = img[y:y+h, x:x+w]
        
        # --- Create a mask for the cropped image ---
        
        # We need a new, smaller mask that fits the cropped image
        mask = np.zeros(img_cropped.shape[:2], dtype="uint8")
        
        # We need to shift the contour coordinates to be relative to the cropped image
        # New coordinates = (original_coords - [x_offset, y_offset])
        c_shifted = c - [x, y]
        
        # Draw the shifted contour onto the new mask (filled)
        cv2.drawContours(mask, [c_shifted], -1, 255, cv2.FILLED)
        
        # Apply the mask to the cropped image
        img_masked = cv2.bitwise_and(img_cropped, img_cropped, mask=mask)

        # Check for empty crop
        (h_crop, w_crop) = img_masked.shape[:2]
        if h_crop == 0 or w_crop == 0:
            return False, None, "Crop resulted in zero-size image"

        # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        # Convert the masked BGR image to LAB color space
        lab = cv2.cvtColor(img_masked, cv2.COLOR_BGR2Lab)
        
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)
        
        # Create a CLAHE object (clipLimit controls contrast limiting, tileGridSize controls block size)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE to the L-channel (Lightness)
        l_clahe = clahe.apply(l)
        
        # Merge the CLAHE-enhanced L-channel back with the original A and B channels
        lab_merged = cv2.merge((l_clahe, a, b))
        
        # Convert the merged LAB image back to BGR color space
        img_clahe = cv2.cvtColor(lab_merged, cv2.COLOR_Lab2BGR)
        
        # Re-apply the mask to remove any artifacts introduced in the black region
        img_clahe_masked = cv2.bitwise_and(img_clahe, img_clahe, mask=mask)

        # 4. Resize and Standardize Dimensions
        img_resized = cv2.resize(img_clahe_masked, target_size, interpolation=cv2.INTER_LINEAR)

        # 5. Save the Result
        
        # Get the original file's basename (e.g., 'image001.jpg')
        basename = os.path.basename(filepath)
        # Get the name without extension (e.g., 'image001')
        filename, _ = os.path.splitext(basename)
        
        # Create a new filename (e.g., 'image001_processed.png')
        # We save as PNG to preserve quality after processing.
        new_filename = f"{filename}_processed.png"
        
        # Create the full output path
        output_path = os.path.join(output_sub_dir, new_filename)
        
        # Save the final image
        cv2.imwrite(output_path, img_resized)
        
        return True, output_path, None

    except Exception as e:
        # Catch any other unexpected errors
        return False, None, str(e)


def main():
    """
    Main function to find images, set up logging, and process all images.
    """
    print("Starting Fundus Image Preprocessing...")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # Create the main output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Find all image files ---
    image_files = []
    print("\nSearching for images...")
    for ext in IMG_EXTENSIONS:
        # Use recursive=True to find images in subdirectories
        pattern = os.path.join(INPUT_DIR, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    if not image_files:
        print(f"Error: No images found in '{INPUT_DIR}'.")
        print("Please create the 'fundus_images' folder and put your images inside it.")
        return

    print(f"Found {len(image_files)} images to process.")

    # --- Set up logging ---
    print(f"Logging results to {LOG_FILE}")
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the CSV header
        writer.writerow(['OriginalFile', 'ProcessedFile', 'Status', 'Message'])
        
        # --- Process each file ---
        print("\nProcessing images...")
        success_count = 0
        fail_count = 0
        
        for i, filepath in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {filepath}")
            
            try:
                # --- Determine output subdirectory to preserve folder structure ---
                
                # Get the relative path from the input directory
                # e.g., 'Normal/image001.jpg' or 'image002.jpg'
                rel_path = os.path.relpath(filepath, INPUT_DIR)
                
                # Get the directory part of that relative path
                # e.g., 'Normal' or ''
                rel_dir = os.path.dirname(rel_path)
                
                # Create the corresponding output subdirectory
                # e.g., 'processed_images/Normal' or 'processed_images/'
                output_sub_dir = os.path.join(OUTPUT_DIR, rel_dir)
                
                # Ensure this output subdirectory exists
                os.makedirs(output_sub_dir, exist_ok=True)

                # Process the image
                success, out_path, error_msg = preprocess_image(filepath, output_sub_dir, TARGET_SIZE)
                
                if success:
                    print(f"  -> SUCCESS: Saved to {out_path}")
                    writer.writerow([filepath, out_path, 'Success', ''])
                    success_count += 1
                else:
                    print(f"  -> FAILED: {error_msg}")
                    writer.writerow([filepath, 'N/A', 'Failed', error_msg])
                    fail_count += 1
            
            except Exception as e:
                # Catch critical errors in the main loop
                print(f"  -> CRITICAL ERROR: {e}")
                writer.writerow([filepath, 'N/A', 'Error', str(e)])
                fail_count += 1

    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process:      {fail_count}")
    print(f"Log file saved to:    {LOG_FILE}")


if __name__ == "__main__":
    main()