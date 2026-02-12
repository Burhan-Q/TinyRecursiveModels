"""
Dataset builder for object detection datasets.
Converts images with bounding box annotations into TRM-compatible format.
"""
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata


cli = ArgParser()


class DetectionDataProcessConfig(BaseModel):
    """Configuration for processing object detection datasets."""
    input_annotation_file: str  # COCO-format JSON file or directory with text annotations
    input_images_dir: str  # Directory containing images
    output_dir: str
    annotation_format: str = "coco"  # "coco" or "yolo"
    subsets: List[str]  # Dataset splits (e.g., train, val, test)
    test_set_name: str = "val"
    seed: int = 42
    num_aug: int = 100  # Number of augmentations per image
    max_objects: int = 100  # Maximum objects per image
    image_size: int = 640  # Target image size (will be resized)
    num_classes: int = 80  # Number of object classes (e.g., COCO has 80)
    puzzle_identifiers_start: int = 1
    # For action descriptions
    use_action_descriptions: bool = True  # Whether to include action descriptions
    

@dataclass
class DetectionPuzzle:
    """
    A detection 'puzzle' is an image with:
    - Detected objects (bounding boxes, classes)
    - An action description (e.g., "person holding cup")
    - A label indicating if the description is correct
    """
    id: str
    image_path: str
    boxes: np.ndarray  # (N, 4) - [x1, y1, x2, y2] normalized to [0, 1]
    classes: np.ndarray  # (N,) - class IDs
    action_description: str
    is_correct: bool  # Whether the action description matches the detections


def load_coco_annotations(annotation_file: str, images_dir: str, config: DetectionDataProcessConfig) -> Dict[str, List[DetectionPuzzle]]:
    """Load COCO format annotations and convert to DetectionPuzzle format."""
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build image id to info mapping
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    img_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # Convert to DetectionPuzzle format
    puzzles = {}
    for split in config.subsets:
        puzzles[split] = []
    
    for img_id, img_info in images.items():
        anns = img_annotations.get(img_id, [])
        
        if not anns:
            continue
            
        # Extract bounding boxes and classes
        boxes = []
        classes = []
        for ann in anns[:config.max_objects]:  # Limit to max_objects
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Normalize to [0, 1]
            img_w, img_h = img_info['width'], img_info['height']
            x1, y1 = x / img_w, y / img_h
            x2, y2 = (x + w) / img_w, (y + h) / img_h
            
            boxes.append([x1, y1, x2, y2])
            classes.append(ann['category_id'])
        
        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes, dtype=np.int32)
        
        # Generate action description (placeholder - in practice, this would come from a dataset with descriptions)
        # For now, we'll create a simple description based on detected objects
        if config.use_action_descriptions:
            # Get category names
            cat_map = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
            obj_names = [cat_map.get(c, f'object_{c}') for c in classes[:3]]  # First 3 objects
            
            if len(obj_names) >= 2:
                action_description = f"{obj_names[0]} near {obj_names[1]}"
            elif len(obj_names) == 1:
                action_description = f"{obj_names[0]} present"
            else:
                action_description = "no objects"
            
            # Randomly mark some as correct/incorrect for training
            is_correct = np.random.random() > 0.3  # 70% correct
        else:
            action_description = ""
            is_correct = True
        
        image_path = os.path.join(images_dir, img_info['file_name'])
        
        # For single split processing, use the first (and only) split from config
        if len(config.subsets) == 1:
            split = config.subsets[0]
        else:
            # Determine split (use filename or metadata)
            split = 'train'  # Default
            if 'val' in img_info['file_name'] or 'test' in img_info['file_name']:
                split = 'val'
            
            # Ensure split exists
            if split not in puzzles:
                split = config.subsets[0]
        
        puzzle = DetectionPuzzle(
            id=f"img_{img_id}",
            image_path=image_path,
            boxes=boxes,
            classes=classes,
            action_description=action_description,
            is_correct=is_correct
        )
        
        puzzles[split].append(puzzle)
    
    return puzzles


def encode_detection_to_sequence(
    puzzle: DetectionPuzzle,
    config: DetectionDataProcessConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode detection puzzle into input and label sequences.
    
    Input format: [image_placeholder, box_coords, class_ids, description_tokens]
    Label format: [is_correct] (1 token indicating if description matches)
    
    Returns:
        input_seq: (seq_len,) array of token IDs
        label_seq: (seq_len,) array of label token IDs
    """
    # Vocabulary:
    # 0: PAD
    # 1: EOS (end of sequence)
    # 2: IMAGE_START
    # 3-6: Coordinate tokens (x1, y1, x2, y2 quantized to bins)
    # 7+: Class tokens and description tokens
    
    # For simplicity, we'll create a sequence representation:
    # [IMAGE_START, obj1_class, obj1_x1, obj1_y1, obj1_x2, obj1_y2, obj2_class, ..., EOS]
    
    seq_parts = [2]  # IMAGE_START token
    
    # Add object detections
    for i in range(min(len(puzzle.boxes), config.max_objects)):
        box = puzzle.boxes[i]
        cls = puzzle.classes[i]
        
        # Quantize coordinates to discrete bins (0-255 mapped to token space)
        coord_bins = 256
        coord_offset = 1000  # Large offset to separate from class tokens
        
        x1 = int(box[0] * coord_bins) + coord_offset
        y1 = int(box[1] * coord_bins) + coord_offset
        x2 = int(box[2] * coord_bins) + coord_offset
        y2 = int(box[3] * coord_bins) + coord_offset
        
        # Add: class, x1, y1, x2, y2
        seq_parts.extend([int(cls) + 10, x1, y1, x2, y2])  # +10 offset for class tokens
    
    # Add EOS
    seq_parts.append(1)
    
    # Pad to fixed length
    max_seq_len = (config.max_objects * 5) + 10  # 5 tokens per object + some buffer
    input_seq = np.zeros(max_seq_len, dtype=np.int32)
    input_seq[:len(seq_parts)] = seq_parts
    
    # Label: binary classification (is_correct)
    # We'll encode this as a sequence too, with ignore labels except the first position
    label_seq = np.full(max_seq_len, -100, dtype=np.int32)  # -100 will be mapped to IGNORE_LABEL_ID
    label_seq[0] = int(puzzle.is_correct)  # 0 or 1
    
    return input_seq, label_seq


def save_split_data(
    puzzles: List[DetectionPuzzle],
    split: str,
    output_dir: str,
    config: DetectionDataProcessConfig,
    puzzle_id_offset: int = 0
):
    """Save detection puzzles to TRM format."""
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    all_inputs = []
    all_labels = []
    all_puzzle_ids = []
    puzzle_indices = [0]
    group_indices = [0]
    
    puzzle_id = puzzle_id_offset
    
    for puzzle in puzzles:
        # Encode puzzle
        input_seq, label_seq = encode_detection_to_sequence(puzzle, config)
        
        # For augmentation, we'll create multiple versions
        for aug_id in range(config.num_aug):
            all_inputs.append(input_seq)
            all_labels.append(label_seq)
            all_puzzle_ids.append(puzzle_id)
        
        puzzle_indices.append(len(all_inputs))
        group_indices.append(puzzle_id + 1)
        puzzle_id += 1
    
    # Convert to numpy arrays
    all_inputs = np.array(all_inputs, dtype=np.int32)
    all_labels = np.array(all_labels, dtype=np.int32)
    all_puzzle_ids = np.array(all_puzzle_ids, dtype=np.int32)
    puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
    group_indices = np.array(group_indices, dtype=np.int32)
    
    # Save arrays
    set_name = "all"
    np.save(os.path.join(output_dir, split, f"{set_name}__inputs.npy"), all_inputs)
    np.save(os.path.join(output_dir, split, f"{set_name}__labels.npy"), all_labels)
    np.save(os.path.join(output_dir, split, f"{set_name}__puzzle_identifiers.npy"), all_puzzle_ids)
    np.save(os.path.join(output_dir, split, f"{set_name}__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(output_dir, split, f"{set_name}__group_indices.npy"), group_indices)
    
    # Create metadata
    max_seq_len = (config.max_objects * 5) + 10
    vocab_size = 1000 + 256 + config.num_classes + 10  # Coordinate bins + classes + special tokens
    
    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=-100,
        blank_identifier_id=0,
        vocab_size=vocab_size,
        seq_len=max_seq_len,
        num_puzzle_identifiers=puzzle_id - puzzle_id_offset,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=config.num_aug,
        total_puzzles=len(puzzles),
        sets=["all"]
    )
    
    # Save metadata
    with open(os.path.join(output_dir, split, "dataset.json"), 'w') as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    print(f"Saved {split} split: {len(puzzles)} puzzles, {len(all_inputs)} examples")
    
    return puzzle_id


@cli.command(singleton=True)
def main(config: DetectionDataProcessConfig):
    """Build object detection dataset for TRM."""
    print(f"Building detection dataset from {config.input_annotation_file}")
    print(f"Output directory: {config.output_dir}")
    
    # Load annotations
    if config.annotation_format == "coco":
        puzzles = load_coco_annotations(
            config.input_annotation_file,
            config.input_images_dir,
            config
        )
    else:
        raise ValueError(f"Unsupported annotation format: {config.annotation_format}")
    
    # Save each split
    puzzle_id_offset = config.puzzle_identifiers_start
    for split in config.subsets:
        if split in puzzles and len(puzzles[split]) > 0:
            # Map split name to output directory
            output_split = "test" if split == config.test_set_name else "train"
            puzzle_id_offset = save_split_data(
                puzzles[split],
                output_split,
                config.output_dir,
                config,
                puzzle_id_offset
            )
    
    print("Dataset building complete!")


if __name__ == "__main__":
    cli()
