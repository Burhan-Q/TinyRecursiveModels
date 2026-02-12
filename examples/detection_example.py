"""
Example script demonstrating object detection action verification with TRM.

This script creates a minimal synthetic dataset and shows how to:
1. Build a detection dataset
2. Train a model
3. Evaluate action verification
"""
import os
import json
import numpy as np
from pathlib import Path


def create_synthetic_coco_annotations(output_path: str, num_images: int = 100):
    """
    Create a synthetic COCO-format annotation file for demonstration.
    
    This creates simple synthetic detections with action descriptions.
    In practice, you would use real COCO annotations or custom datasets.
    """
    # Define classes
    categories = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "cup"},
        {"id": 3, "name": "chair"},
        {"id": 4, "name": "table"},
        {"id": 5, "name": "laptop"},
        {"id": 6, "name": "phone"},
        {"id": 7, "name": "book"},
        {"id": 8, "name": "bottle"},
    ]
    
    images = []
    annotations = []
    ann_id = 1
    
    for img_id in range(1, num_images + 1):
        # Create image metadata
        images.append({
            "id": img_id,
            "file_name": f"synthetic_{img_id:04d}.jpg",
            "width": 640,
            "height": 480
        })
        
        # Create random detections (2-5 objects per image)
        num_objects = np.random.randint(2, 6)
        
        for _ in range(num_objects):
            # Random category
            cat_id = np.random.randint(1, len(categories) + 1)
            
            # Random bounding box (ensuring valid coordinates)
            x = np.random.randint(0, 500)
            y = np.random.randint(0, 350)
            w = np.random.randint(50, 140)
            h = np.random.randint(50, 130)
            
            # Ensure box is within image bounds
            if x + w > 640:
                w = 640 - x
            if y + h > 480:
                h = 480 - y
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1
    
    # Create COCO format JSON
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Created synthetic COCO annotations: {output_path}")
    print(f"  - {len(images)} images")
    print(f"  - {len(annotations)} annotations")
    print(f"  - {len(categories)} categories")
    
    return output_path


def main():
    """Run the example pipeline."""
    print("="*70)
    print("Object Detection Action Verification - Example")
    print("="*70)
    print()
    
    # Step 1: Create synthetic data
    print("Step 1: Creating synthetic COCO annotations...")
    output_dir = "data/detection-example"
    os.makedirs(output_dir, exist_ok=True)
    
    train_ann_file = create_synthetic_coco_annotations(
        os.path.join(output_dir, "annotations_train.json"),
        num_images=80
    )
    val_ann_file = create_synthetic_coco_annotations(
        os.path.join(output_dir, "annotations_val.json"),
        num_images=20
    )
    print()
    
    # Step 2: Explain how to build dataset
    print("Step 2: Build TRM-compatible dataset")
    print("-" * 70)
    print("Run the following command to build the dataset:")
    print()
    print("python -m dataset.build_detection_dataset \\")
    print(f"  --input-annotation-file {train_ann_file} \\")
    print(f"  --input-images-dir {output_dir}/images \\")
    print(f"  --output-dir {output_dir}-processed \\")
    print("  --annotation-format coco \\")
    print("  --subsets train \\")
    print("  --test-set-name val \\")
    print("  --num-aug 50 \\")
    print("  --max-objects 20 \\")
    print("  --num-classes 8")
    print()
    print("⚠️  WARNING: This example creates annotations only.")
    print("    You must provide actual images in the input-images-dir.")
    print("    For a working example, see the COCO8 dataset setup in README.md")
    print()
    
    # Step 3: Explain training
    print("Step 3: Train the model")
    print("-" * 70)
    print("Run the following command to train:")
    print()
    print("python pretrain.py \\")
    print("  --config-name cfg_detection \\")
    print(f"  data_paths=\"[{output_dir}-processed]\" \\")
    print("  arch.L_cycles=4 \\")
    print("  arch.H_cycles=3 \\")
    print("  global_batch_size=32 \\")
    print("  epochs=5000 \\")
    print("  eval_interval=500 \\")
    print("  +run_name=\"detection_example\"")
    print()
    
    # Step 4: Explain evaluation
    print("Step 4: Evaluation")
    print("-" * 70)
    print("The model will automatically evaluate during training.")
    print("Metrics include:")
    print("  - Accuracy: Overall classification accuracy")
    print("  - Precision: Precision for positive predictions")
    print("  - Recall: Recall for positive class")
    print("  - F1 Score: Harmonic mean of precision and recall")
    print()
    print("Results will be logged to Weights & Biases and saved in checkpoints/")
    print()
    
    # Summary
    print("="*70)
    print("Example Summary")
    print("="*70)
    print()
    print("This example demonstrates how to:")
    print("  1. Create COCO-format annotations (or use existing ones)")
    print("  2. Build a TRM-compatible detection dataset")
    print("  3. Train the model with recursive reasoning")
    print("  4. Evaluate action verification performance")
    print()
    print("For real-world usage:")
    print("  - Use actual object detection datasets (COCO, custom)")
    print("  - Add real action descriptions from human annotations")
    print("  - Consider adding a vision encoder for raw images")
    print("  - Tune hyperparameters based on your task")
    print()
    print("See DETECTION_README.md for more details!")
    print("="*70)


if __name__ == "__main__":
    main()
