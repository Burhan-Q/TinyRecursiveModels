"""
Evaluator for object detection action verification task.
"""
import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import torch


class DetectionActionEvaluator:
    """
    Evaluator for action description verification in object detection.
    
    This evaluator computes:
    - Accuracy: Percentage of correct action description classifications
    - Precision/Recall: For positive predictions
    - F1 Score: Harmonic mean of precision and recall
    """
    
    # Small constant to avoid division by zero
    EPSILON = 1e-8
    
    def __init__(
        self,
        data_path: str,
        eval_metadata: Any,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Args:
            data_path: Path to dataset
            eval_metadata: Metadata from the dataset
            threshold: Classification threshold for predictions
        """
        self.data_path = data_path
        self.eval_metadata = eval_metadata
        self.threshold = threshold
        
        # Accumulate predictions and labels
        self.all_preds = []
        self.all_labels = []
        self.all_logits = []
        
        # Required outputs from model
        self.required_outputs = ["logits", "pred"]
    
    def begin_eval(self):
        """Reset evaluation state."""
        self.all_preds = []
        self.all_labels = []
        self.all_logits = []
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Update with a batch of predictions.
        
        Args:
            batch: Dictionary with 'labels' tensor (B, seq_len)
            preds: Dictionary with 'pred' tensor (B, seq_len) and optionally 'logits'
        """
        # Extract labels (only first position matters - binary classification)
        labels = batch['labels'][:, 0].cpu().numpy()  # (B,)
        
        # Extract predictions (first position)
        if 'pred' in preds:
            pred = preds['pred'][:, 0].cpu().numpy()  # (B,)
        else:
            # If no pred, use logits
            logits = preds.get('logits', None)
            if logits is not None:
                # Assume binary classification, take class 1 probability
                logits = logits[:, 0].cpu().numpy()  # (B,)
                pred = (logits > self.threshold).astype(np.int32)
            else:
                raise ValueError("No predictions available")
        
        # Filter out ignore labels
        valid_mask = labels != -100
        
        self.all_labels.extend(labels[valid_mask].tolist())
        self.all_preds.extend(pred[valid_mask].tolist())
        
        if 'logits' in preds:
            logits = preds['logits'][:, 0].cpu().numpy()
            self.all_logits.extend(logits[valid_mask].tolist())
    
    def result(
        self,
        save_path: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        group: Optional[Any] = None
    ) -> Optional[Dict[str, float]]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metrics if rank == 0, else None
        """
        if rank != 0:
            return None
        
        if len(self.all_labels) == 0:
            print("Warning: No predictions to evaluate")
            return None
        
        # Convert to numpy
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        
        # Compute metrics
        accuracy = (preds == labels).mean()
        
        # Compute precision, recall, F1 for positive class
        true_positives = ((preds == 1) & (labels == 1)).sum()
        false_positives = ((preds == 1) & (labels == 0)).sum()
        false_negatives = ((preds == 0) & (labels == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives + self.EPSILON)
        recall = true_positives / (true_positives + false_negatives + self.EPSILON)
        f1 = 2 * precision * recall / (precision + recall + self.EPSILON)
        
        metrics = {
            "detection/accuracy": float(accuracy),
            "detection/precision": float(precision),
            "detection/recall": float(recall),
            "detection/f1": float(f1),
            "detection/num_samples": len(labels),
            "detection/positive_ratio": float((labels == 1).mean()),
        }
        
        # Save detailed results if requested
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            
            results = {
                "metrics": metrics,
                "predictions": {
                    "labels": labels.tolist(),
                    "preds": preds.tolist(),
                }
            }
            
            if self.all_logits:
                results["predictions"]["logits"] = self.all_logits
            
            with open(os.path.join(save_path, "results.json"), 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Saved evaluation results to {save_path}")
        
        # Print metrics
        print("\n" + "="*50)
        print("Detection Action Verification Results")
        print("="*50)
        for key, value in metrics.items():
            if "num_samples" not in key:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("="*50 + "\n")
        
        return metrics
