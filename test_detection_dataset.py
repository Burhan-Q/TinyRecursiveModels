"""
Test script to validate the detection dataset can be loaded correctly.
This doesn't require training, just checks the dataset structure.
"""
import numpy as np
import json
import os


def test_dataset_loading(dataset_path, split='train'):
    """Test loading a detection dataset split."""
    split_dir = os.path.join(dataset_path, split)
    
    print(f"\nTesting {split} split at {split_dir}")
    print("="*60)
    
    # Load metadata
    with open(os.path.join(split_dir, 'dataset.json'), 'r') as f:
        metadata = json.load(f)
    
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Load data arrays
    inputs = np.load(os.path.join(split_dir, 'all__inputs.npy'))
    labels = np.load(os.path.join(split_dir, 'all__labels.npy'))
    puzzle_ids = np.load(os.path.join(split_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(split_dir, 'all__puzzle_indices.npy'))
    group_indices = np.load(os.path.join(split_dir, 'all__group_indices.npy'))
    
    print(f"\nArray shapes:")
    print(f"  inputs: {inputs.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  puzzle_ids: {puzzle_ids.shape}")
    print(f"  puzzle_indices: {puzzle_indices.shape}")
    print(f"  group_indices: {group_indices.shape}")
    
    # Sample first example
    print(f"\nFirst example:")
    print(f"  input tokens (first 20): {inputs[0, :20]}")
    print(f"  label tokens (first 10): {labels[0, :10]}")
    print(f"  puzzle_id: {puzzle_ids[0]}")
    
    # Decode first input
    print(f"\nDecoding first input:")
    input_seq = inputs[0]
    
    # Find IMAGE_START token (2)
    if input_seq[0] == 2:
        print(f"  ✓ Starts with IMAGE_START token (2)")
    
    # Find EOS token (1)
    eos_idx = np.where(input_seq == 1)[0]
    if len(eos_idx) > 0:
        print(f"  ✓ Contains EOS token (1) at position {eos_idx[0]}")
        num_tokens = eos_idx[0]
        print(f"  ✓ Sequence length: {num_tokens} tokens")
    
    # Count objects (each object has 5 tokens: class, x1, y1, x2, y2)
    # After IMAGE_START (token 0), we have objects
    non_pad = input_seq[input_seq != 0]
    non_eos = non_pad[non_pad != 1]
    num_obj_tokens = len(non_eos) - 1  # Minus IMAGE_START
    num_objects = num_obj_tokens // 5
    print(f"  ✓ Estimated {num_objects} objects detected")
    
    # Check label
    print(f"\nFirst label:")
    label_seq = labels[0]
    non_ignore = label_seq[label_seq != -100]
    if len(non_ignore) > 0:
        print(f"  Binary classification label: {non_ignore[0]} ({'correct' if non_ignore[0] == 1 else 'incorrect'})")
    
    # Validate consistency
    print(f"\nValidation:")
    assert inputs.shape[0] == labels.shape[0], "Mismatch: inputs and labels"
    assert inputs.shape[0] == puzzle_ids.shape[0], "Mismatch: inputs and puzzle_ids"
    assert inputs.shape[1] == metadata['seq_len'], f"Mismatch: seq_len {inputs.shape[1]} != {metadata['seq_len']}"
    print(f"  ✓ All shapes match metadata")
    print(f"  ✓ {metadata['total_puzzles']} puzzles with {metadata['mean_puzzle_examples']} examples each")
    print(f"  ✓ Total examples: {inputs.shape[0]}")
    
    return True


def main():
    print("="*60)
    print("Detection Dataset Validation Test")
    print("="*60)
    
    dataset_path = "data/coco8-detection"
    
    # Test both splits
    for split in ['train', 'test']:
        try:
            test_dataset_loading(dataset_path, split)
            print(f"\n✓ {split.upper()} SPLIT: PASSED")
        except Exception as e:
            print(f"\n✗ {split.upper()} SPLIT: FAILED")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! Dataset is ready for training.")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
