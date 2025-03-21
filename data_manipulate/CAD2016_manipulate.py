import os
import shutil

# Define base dataset directory
base_dir = r"D:\Dateset\CAD2016_GT"

# Define base output directory
output_base_dir = r"D:\Dateset\Processed"

# Iterate over all categories (e.g., chameleon, redblue, etc.)
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    gt_path = os.path.join(category_path, "GT")
    frame_path = os.path.join(category_path, "Frame")

    if os.path.isdir(gt_path) and os.path.isdir(frame_path):  # Ensure both folders exist
        # Define output directories for this category
        output_category_dir = os.path.join(output_base_dir, category)
        output_frames_dir = os.path.join(output_category_dir, "Frame")
        output_gt_dir = os.path.join(output_category_dir, "GT")

        # Create subfolders if they don’t exist
        os.makedirs(output_frames_dir, exist_ok=True)
        os.makedirs(output_gt_dir, exist_ok=True)

        gt_files = {}

        # Find all GT files and extract their numeric values
        for gt_file in os.listdir(gt_path):
            if gt_file.endswith("_gt.png"):
                num_part = gt_file.split("_gt.png")[0]  # Extract numeric part
                gt_files[num_part] = gt_file

        # Process corresponding frame files
        for num, gt_filename in gt_files.items():
            frame_filename = f"{category}_{num}.png"
            frame_full_path = os.path.join(frame_path, frame_filename)
            gt_full_path = os.path.join(gt_path, gt_filename)

            if os.path.exists(frame_full_path):  # Ensure the frame exists
                new_name = f"{int(num):05d}.png"  # Format as 5-digit number

                # Copy and rename files to category-specific output directories
                shutil.copy2(frame_full_path, os.path.join(output_frames_dir, new_name))
                shutil.copy2(gt_full_path, os.path.join(output_gt_dir, new_name))

print("Processing complete! GT and Frame files are synchronized, renamed, and stored in subfolders.")
