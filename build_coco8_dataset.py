"""
Build complete coco8 detection dataset with train and test splits.
"""
import subprocess
import os

# Build train split
print("Building train split...")
subprocess.run([
    "python", "-m", "dataset.build_detection_dataset",
    "--input-annotation-file", "data/coco8/annotations_train.json",
    "--input-images-dir", "data/coco8/images/train",
    "--output-dir", "data/coco8-detection-temp-train",
    "--annotation-format", "coco",
    "--subsets", "train",
    "--test-set-name", "val",
    "--num-aug", "10",
    "--max-objects", "20",
    "--num-classes", "80"
], check=True)

# Build val/test split  
print("\nBuilding test split...")
subprocess.run([
    "python", "-m", "dataset.build_detection_dataset",
    "--input-annotation-file", "data/coco8/annotations_val.json",
    "--input-images-dir", "data/coco8/images/val",
    "--output-dir", "data/coco8-detection-temp-test",
    "--annotation-format", "coco",
    "--subsets", "val",
    "--test-set-name", "val",
    "--num-aug", "10",
    "--max-objects", "20",
    "--num-classes", "80"
], check=True)

# Move to final location
print("\nOrganizing dataset...")
os.makedirs("data/coco8-detection", exist_ok=True)

# Move train split
if os.path.exists("data/coco8-detection-temp-train/train"):
    os.rename("data/coco8-detection-temp-train/train", "data/coco8-detection/train")
    os.rmdir("data/coco8-detection-temp-train")

# Move test split
if os.path.exists("data/coco8-detection-temp-test/test"):
    os.rename("data/coco8-detection-temp-test/test", "data/coco8-detection/test")
    os.rmdir("data/coco8-detection-temp-test")

print("\nDataset ready at data/coco8-detection/")
print("  - Train split: data/coco8-detection/train/")
print("  - Test split: data/coco8-detection/test/")
