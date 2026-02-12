"""
Convert YOLO format dataset (like coco8) to COCO format for TRM processing.
"""
import os
import json
from pathlib import Path
from PIL import Image


def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format bbox to COCO format.
    
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    COCO format: [x_min, y_min, width, height] (absolute pixels)
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert to absolute coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convert to top-left corner
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    
    return [x_min, y_min, width, height]


def convert_yolo_to_coco(yolo_dir, output_file, split='train', class_names=None):
    """
    Convert YOLO format dataset to COCO format.
    
    Args:
        yolo_dir: Path to YOLO dataset directory (containing images/ and labels/)
        output_file: Output path for COCO JSON file
        split: Dataset split ('train' or 'val')
        class_names: Optional list of class names (if None, uses generic names)
    """
    images_dir = os.path.join(yolo_dir, 'images', split)
    labels_dir = os.path.join(yolo_dir, 'labels', split)
    
    # Get all image files
    image_files = sorted(Path(images_dir).glob('*.jpg'))
    
    # Default COCO class names (80 classes)
    if class_names is None:
        class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    
    # Create categories
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    
    images = []
    annotations = []
    ann_id = 1
    
    for img_id, img_path in enumerate(image_files, start=1):
        # Load image to get dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Add image entry
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height
        })
        
        # Load corresponding label file
        label_file = os.path.join(labels_dir, img_path.stem + '.txt')
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    yolo_bbox = [float(x) for x in parts[1:5]]
                    
                    # Convert to COCO format
                    coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
                    
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": coco_bbox,
                        "area": coco_bbox[2] * coco_bbox[3],
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
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Converted {split} split:")
    print(f"  - Images: {len(images)}")
    print(f"  - Annotations: {len(annotations)}")
    print(f"  - Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    # Convert coco8 dataset
    yolo_dir = "data/coco8"
    
    print("Converting YOLO format (coco8) to COCO format...\n")
    
    # Convert train split
    train_output = convert_yolo_to_coco(
        yolo_dir,
        "data/coco8/annotations_train.json",
        split='train'
    )
    
    print()
    
    # Convert val split
    val_output = convert_yolo_to_coco(
        yolo_dir,
        "data/coco8/annotations_val.json",
        split='val'
    )
    
    print("\nConversion complete!")
    print(f"Train annotations: {train_output}")
    print(f"Val annotations: {val_output}")
