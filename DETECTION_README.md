# Object Detection Action Verification

This extension adapts TinyRecursiveModels (TRM) to work with image object detection data, specifically for verifying whether action descriptions match the detected objects in images.

## Important Note

**This implementation uses detection metadata (bounding boxes + class IDs), not raw image pixels:**
- The current approach encodes object detections as discrete tokens
- Image paths are stored in the dataset but **not used for feature extraction**
- No image embeddings or vision encoder are included in this version
- This design maintains compatibility with TRM's token-based architecture
- To use actual image features, see [Extending to Real-World Use Cases](#extending-to-real-world-use-cases)

## Overview

The object detection adaptation allows TRM to:
- Process object detection predictions (bounding boxes and class labels)
- Verify if action descriptions correctly describe what's detected in the scene
- Use recursive reasoning to improve classification accuracy

## Task Description

Given:
- Object detection results: bounding boxes + class labels (e.g., person at [0.1, 0.2, 0.5, 0.6], cup at [0.3, 0.4, 0.4, 0.5])
- A text description of an action (e.g., "person holding cup")

The model predicts:
- Whether the description correctly describes the detected objects (binary classification: correct/incorrect)

**Note**: The model does not process the raw image - only the detection metadata.

## Dataset Format

The object detection dataset uses COCO-format annotations converted to TRM's sequence format:

### Input Sequence Format
```
[IMAGE_START, obj1_class, obj1_x1, obj1_y1, obj1_x2, obj1_y2, obj2_class, ..., EOS]
```

Where:
- `IMAGE_START`: Special token (2) marking the start
- `obj_class`: Object class ID (offset by 10 to avoid collision with special tokens)
- `x1, y1, x2, y2`: Bounding box coordinates quantized to 256 bins (offset by 1000)
- `EOS`: End-of-sequence marker (1)

### Label Format
```
[is_correct, IGNORE, IGNORE, ...]
```

Where:
- `is_correct`: Binary label (0 or 1) indicating if the action description matches
- `IGNORE`: Ignore labels (-100) for remaining positions

## Building a Detection Dataset

### From COCO Annotations

```bash
python -m dataset.build_detection_dataset \
  --input-annotation-file path/to/coco/annotations/instances_train2017.json \
  --input-images-dir path/to/coco/images/train2017 \
  --output-dir data/detection-action-aug-100 \
  --annotation-format coco \
  --subsets train val \
  --test-set-name val \
  --num-aug 100 \
  --max-objects 50 \
  --num-classes 80
```

Parameters:
- `--input-annotation-file`: Path to COCO JSON annotation file
- `--input-images-dir`: Directory containing images
- `--output-dir`: Output directory for processed dataset
- `--annotation-format`: Annotation format (currently supports "coco")
- `--subsets`: Dataset splits to process (e.g., train, val)
- `--test-set-name`: Which subset to use as test set
- `--num-aug`: Number of augmentations per image
- `--max-objects`: Maximum number of objects to encode per image
- `--num-classes`: Number of object classes (80 for COCO)

### Custom Annotations

For custom action verification datasets, prepare COCO-format JSON with:
```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "cup"}
  ]
}
```

Note: Action descriptions are currently generated automatically. For real action verification tasks, extend the dataset builder to load descriptions from an additional annotation file.

## Training

### Basic Training

```bash
python pretrain.py \
  --config-name cfg_detection \
  data_paths="[data/detection-action-aug-100]" \
  arch.L_cycles=4 \
  arch.H_cycles=3 \
  +run_name="detection_baseline"
```

### Advanced Options

```bash
# With EMA (Exponential Moving Average)
python pretrain.py \
  --config-name cfg_detection \
  data_paths="[data/detection-action-aug-100]" \
  ema=True \
  ema_rate=0.999

# Multi-GPU training
torchrun --nproc-per-node 4 pretrain.py \
  --config-name cfg_detection \
  data_paths="[data/detection-action-aug-100]" \
  global_batch_size=256
```

## Evaluation

The model is evaluated using:
- **Accuracy**: Percentage of correctly classified descriptions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

Results are logged to Weights & Biases and saved to checkpoint directory.

## Model Architecture

The model uses the same TRM architecture as the original paper, but:
- Input sequences encode object detections instead of grid patterns
- Vocabulary size is adjusted to accommodate object classes and coordinate bins
- Output is binary classification instead of grid reconstruction

### Key Components

1. **Sparse Embedding Layer**: Embeds discrete tokens (object classes, coordinates)
2. **Recursive Reasoning Module**: Iteratively refines predictions through H_cycles × L_cycles
3. **Classification Head**: Outputs binary logits for action verification

## Extending to Real-World Use Cases

The current implementation uses **detection metadata only** (no image pixels). To adapt this for production object detection with actual images:

1. **Add Vision Encoder**: 
   - Integrate a CNN (ResNet, EfficientNet) or Vision Transformer (ViT, CLIP)
   - Extract image features (e.g., 2048-dim vector from ResNet)
   - Replace or augment the `IMAGE_START` token with image embedding
   - Update model to accept both discrete tokens and continuous features

2. **Use Real Detections**: 
   - Integrate with YOLO, Faster R-CNN, DETR, or other detectors
   - Run detector on raw images to get bboxes + classes
   - Feed detection results into TRM as discrete tokens (current approach)

3. **Action Dataset**: 
   - Use datasets with human-annotated action descriptions:
     - HICO-DET (Human-Object Interaction Detection)
     - V-COCO (Verbs in COCO)
     - Custom annotations linking detections to actions
   - Replace synthetic descriptions with real annotations

4. **Multi-label Classification**: 
   - Extend beyond binary to predict multiple simultaneous actions
   - Support complex descriptions: "person holding cup and sitting on chair"

5. **Temporal Reasoning**: 
   - Process video sequences frame-by-frame
   - Track objects and actions over time
   - Add temporal tokens to encode frame ordering

### Example Integration with Vision Encoder

```python
# Pseudocode for adding image features
class DetectionModelWithVision(nn.Module):
    def __init__(self):
        self.vision_encoder = torchvision.models.resnet50(pretrained=True)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.fusion = nn.Linear(2048 + hidden_dim, hidden_dim)
        self.trm = TinyRecursiveModel(...)
    
    def forward(self, image, detection_tokens):
        # Extract image features
        img_features = self.vision_encoder(image)  # (B, 2048)
        
        # Embed detection tokens
        token_embeds = self.token_embeddings(detection_tokens)  # (B, seq_len, hidden_dim)
        
        # Fuse image features with first token
        token_embeds[:, 0] = self.fusion(
            torch.cat([img_features, token_embeds[:, 0]], dim=-1)
        )
        
        # Run TRM
        return self.trm(token_embeds)
```
3. **Action Dataset**: Use datasets like HICO-DET, V-COCO, or create custom annotations
4. **Multi-label Classification**: Extend to predict multiple action types
5. **Temporal Reasoning**: Add video support for action recognition over time

## Example Use Cases

- **Safety Monitoring**: Verify if detected actions match safety protocols
- **Image Captioning Validation**: Check if generated captions accurately describe scenes
- **Video Understanding**: Verify action descriptions in video frames
- **Robotic Vision**: Validate object interactions for manipulation tasks

## Performance Notes

- Binary classification is simpler than grid puzzles, so smaller models may suffice
- Consider reducing H_cycles and L_cycles for faster inference
- The recursive reasoning helps correct initial misclassifications
- Augmentation is crucial for generalization with limited training data

## Citation

If you use this object detection adaptation, please cite the original TRM paper:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```
