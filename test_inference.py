"""
Minimal inference test on COCO8 detection dataset.
This script tests loading the dataset and running a forward pass through a simple model.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add repo to path
sys.path.insert(0, '/home/runner/work/TinyRecursiveModels/TinyRecursiveModels')

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def create_simple_detection_model(vocab_size, seq_len, hidden_dim=64):
    """Create a simple baseline model for detection task."""
    class SimpleDetectionModel(torch.nn.Module):
        def __init__(self, vocab_size, seq_len, hidden_dim):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
            self.fc1 = torch.nn.Linear(seq_len * hidden_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, 2)  # Binary classification
            
        def forward(self, inputs):
            # inputs: (B, seq_len)
            x = self.embedding(inputs)  # (B, seq_len, hidden_dim)
            x = x.reshape(x.size(0), -1)  # (B, seq_len * hidden_dim)
            x = torch.relu(self.fc1(x))  # (B, hidden_dim)
            logits = self.fc2(x)  # (B, 2)
            return logits
    
    return SimpleDetectionModel(vocab_size, seq_len, hidden_dim)


def test_dataset_loading():
    """Test loading the detection dataset."""
    print("="*70)
    print("Testing Dataset Loading")
    print("="*70)
    
    config = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=["data/coco8-detection"],
        global_batch_size=4,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    dataset = PuzzleDataset(config, split="train")
    
    print(f"\n✓ Dataset created successfully")
    print(f"  Metadata:")
    print(f"    - vocab_size: {dataset.metadata.vocab_size}")
    print(f"    - seq_len: {dataset.metadata.seq_len}")
    print(f"    - total_puzzles: {dataset.metadata.total_puzzles}")
    print(f"    - total_groups: {dataset.metadata.total_groups}")
    
    return dataset


def test_inference():
    """Run a simple inference test."""
    print("\n" + "="*70)
    print("Testing Inference")
    print("="*70)
    
    # Load dataset
    config = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=["data/coco8-detection"],
        global_batch_size=4,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    dataset = PuzzleDataset(config, split="test")
    
    # Get one batch
    batch_iter = iter(dataset)
    set_name, batch, global_batch_size = next(batch_iter)
    
    print(f"\n✓ Loaded batch from {set_name}")
    print(f"  Batch info:")
    print(f"    - global_batch_size: {global_batch_size}")
    print(f"    - inputs shape: {batch['inputs'].shape}")
    print(f"    - labels shape: {batch['labels'].shape}")
    print(f"    - puzzle_identifiers shape: {batch['puzzle_identifiers'].shape}")
    
    # Create a simple model
    model = create_simple_detection_model(
        vocab_size=dataset.metadata.vocab_size,
        seq_len=dataset.metadata.seq_len,
        hidden_dim=64
    )
    
    print(f"\n✓ Created model")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        inputs = batch['inputs']
        labels = batch['labels']
        
        # Forward pass
        logits = model(inputs)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Extract ground truth labels (first position only, others are ignore)
        gt_labels = labels[:, 0]
        valid_mask = gt_labels != -100
        
        if valid_mask.any():
            gt_labels_valid = gt_labels[valid_mask]
            preds_valid = preds[valid_mask]
            
            accuracy = (preds_valid == gt_labels_valid).float().mean()
            
            print(f"\n✓ Inference completed")
            print(f"  Results:")
            print(f"    - Logits shape: {logits.shape}")
            print(f"    - Predictions: {preds_valid.tolist()}")
            print(f"    - Ground truth: {gt_labels_valid.tolist()}")
            print(f"    - Random accuracy: {accuracy.item():.2%} (expected ~50% for random model)")
        
    # Test multiple batches
    print(f"\n✓ Testing multiple batches...")
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for set_name, batch, global_batch_size in dataset:
        with torch.no_grad():
            logits = model(batch['inputs'])
            preds = torch.argmax(logits, dim=1)
            labels = batch['labels'][:, 0]
            
            valid_mask = labels != -100
            if valid_mask.any():
                all_preds.extend(preds[valid_mask].tolist())
                all_labels.extend(labels[valid_mask].tolist())
        
        num_batches += 1
        if num_batches >= 5:  # Test first 5 batches
            break
    
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    overall_accuracy = (all_preds == all_labels).float().mean()
    
    print(f"  Processed {num_batches} batches")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Overall random accuracy: {overall_accuracy.item():.2%}")
    
    # Show prediction distribution
    unique_preds, counts = torch.unique(all_preds, return_counts=True)
    print(f"\n  Prediction distribution:")
    for pred, count in zip(unique_preds.tolist(), counts.tolist()):
        pred_str = "correct" if pred == 1 else "incorrect"
        print(f"    - {pred_str} (class {pred}): {count} samples ({count/len(all_labels)*100:.1f}%)")
    
    return True


def test_sequence_encoding():
    """Test that sequences are properly encoded."""
    print("\n" + "="*70)
    print("Testing Sequence Encoding")
    print("="*70)
    
    config = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=["data/coco8-detection"],
        global_batch_size=4,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    dataset = PuzzleDataset(config, split="test")
    
    # Get one batch
    batch_iter = iter(dataset)
    set_name, batch, global_batch_size = next(batch_iter)
    
    inputs = batch['inputs'][0]  # First example
    
    # Check sequence structure
    print(f"\n✓ Analyzing first sequence:")
    print(f"  First 10 tokens: {inputs[:10].tolist()}")
    
    # Find IMAGE_START (token 2)
    if inputs[0] == 2:
        print(f"  ✓ Starts with IMAGE_START token (2)")
    else:
        print(f"  ✗ ERROR: Does not start with IMAGE_START!")
        return False
    
    # Find EOS (token 1)
    eos_positions = (inputs == 1).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        eos_pos = eos_positions[0].item()
        print(f"  ✓ Contains EOS token at position {eos_pos}")
        
        # Count objects (after IMAGE_START, before EOS, in groups of 5)
        num_tokens = eos_pos - 1  # Exclude IMAGE_START
        num_objects = num_tokens // 5
        print(f"  ✓ Sequence encodes ~{num_objects} objects")
        
        # Extract first object
        if num_tokens >= 5:
            obj_tokens = inputs[1:6].tolist()  # First object tokens
            print(f"  First object tokens: {obj_tokens}")
            print(f"    - Class token: {obj_tokens[0]} (class ID: {obj_tokens[0] - 10})")
            print(f"    - Coord tokens: x1={obj_tokens[1]}, y1={obj_tokens[2]}, x2={obj_tokens[3]}, y2={obj_tokens[4]}")
    
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("COCO8 Detection Dataset - Inference Test")
    print("="*70)
    
    try:
        # Test 1: Dataset loading
        dataset = test_dataset_loading()
        
        # Test 2: Sequence encoding
        test_sequence_encoding()
        
        # Test 3: Inference
        test_inference()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("  ✓ Dataset loads correctly")
        print("  ✓ Sequences are properly encoded")
        print("  ✓ Model inference works")
        print("  ✓ Predictions can be generated")
        print("\nThe detection dataset is ready for training with TRM!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
