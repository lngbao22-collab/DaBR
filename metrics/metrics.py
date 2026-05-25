# coding: utf-8
"""
Metrics module for both link prediction and triple classification tasks.
Includes ranking metrics and classification metrics.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score


class Metrics:
    """Metrics calculator for knowledge graph embedding models."""

    @staticmethod
    def _to_python_number(value):
        """Convert NumPy scalar types into native Python numbers."""
        if isinstance(value, np.generic):
            return value.item()
        return value
    
    @staticmethod
    def calculate_ranking_metrics(ranks, hits):
        """
        Calculate ranking metrics for link prediction task.
        
        Args:
            ranks: list of ranking positions (1-indexed)
            hits: dict with hit@k keys (e.g., {'hit@1': count, 'hit@3': count, 'hit@10': count})
        
        Returns:
            dict with MR, MRR, Hit@1, Hit@3, Hit@10
        """
        ranks = np.array(ranks)
        n = len(ranks)
        
        mr = np.mean(ranks)
        mrr = np.mean(1.0 / ranks)
        
        metrics = {
            'MR': float(mr),
            'MRR': float(mrr),
            'Hit@1': float(hits.get('hit@1', 0) / n) if n > 0 else 0.0,
            'Hit@3': float(hits.get('hit@3', 0) / n) if n > 0 else 0.0,
            'Hit@10': float(hits.get('hit@10', 0) / n) if n > 0 else 0.0,
        }
        
        return metrics
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_scores=None):
        """
        Calculate classification metrics for triple classification task.
        
        Args:
            y_true: ground truth labels (0 or 1)
            y_pred: predicted labels (0 or 1)
            y_scores: prediction scores/probabilities for AUC metrics (optional)
        
        Returns:
            dict with Accuracy, Precision, Recall, F1-Score, PR-AUC, ROC-AUC
        """
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Handle precision/recall for binary classification
        if len(np.unique(y_true)) > 1:
            metrics['Precision'] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
            metrics['Recall'] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
            metrics['F1-Score'] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
        else:
            # If only one class in labels, return 0
            metrics['Precision'] = 0.0
            metrics['Recall'] = 0.0
            metrics['F1-Score'] = 0.0
        
        # AUC metrics (requires prediction scores)
        if y_scores is not None:
            y_scores = np.array(y_scores)
            try:
                metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_scores))
            except:
                metrics['ROC-AUC'] = 0.0
            
            try:
                metrics['PR-AUC'] = float(average_precision_score(y_true, y_scores))
            except:
                metrics['PR-AUC'] = 0.0
        else:
            metrics['ROC-AUC'] = 0.0
            metrics['PR-AUC'] = 0.0
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics_dict):
        """
        Format metrics dictionary into a readable string.
        
        Args:
            metrics_dict: dictionary of metrics
        
        Returns:
            formatted string representation
        """
        lines = []
        for key, value in metrics_dict.items():
            value = Metrics._to_python_number(value)
            if isinstance(value, (float, np.floating)):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        return " | ".join(lines)


class RankingMetricsCalculator:
    """Calculates ranking metrics during link prediction evaluation."""
    
    def __init__(self):
        self.ranks_head = []
        self.ranks_tail = []
        self.hits_head = {'hit@1': 0, 'hit@3': 0, 'hit@10': 0}
        self.hits_tail = {'hit@1': 0, 'hit@3': 0, 'hit@10': 0}
    
    def add_rank(self, rank, entity_type='head'):
        """
        Add a ranking result.
        
        Args:
            rank: ranking position (1-indexed)
            entity_type: 'head' or 'tail'
        """
        if entity_type == 'head':
            self.ranks_head.append(rank)
            if rank <= 1:
                self.hits_head['hit@1'] += 1
            if rank <= 3:
                self.hits_head['hit@3'] += 1
            if rank <= 10:
                self.hits_head['hit@10'] += 1
        else:
            self.ranks_tail.append(rank)
            if rank <= 1:
                self.hits_tail['hit@1'] += 1
            if rank <= 3:
                self.hits_tail['hit@3'] += 1
            if rank <= 10:
                self.hits_tail['hit@10'] += 1
    
    def get_metrics(self):
        """Calculate and return all ranking metrics."""
        # Head metrics
        head_metrics = Metrics.calculate_ranking_metrics(
            self.ranks_head, self.hits_head
        )
        
        # Tail metrics
        tail_metrics = Metrics.calculate_ranking_metrics(
            self.ranks_tail, self.hits_tail
        )
        
        # Combined metrics (average)
        combined_metrics = {}
        for key in head_metrics:
            combined_metrics[key] = (head_metrics[key] + tail_metrics[key]) / 2.0
        
        return {
            'head': head_metrics,
            'tail': tail_metrics,
            'combined': combined_metrics
        }


class ClassificationMetricsCalculator:
    """Calculates classification metrics during triple classification evaluation."""
    
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
    
    def add_prediction(self, true_label, pred_label, pred_score=None):
        """
        Add a prediction result.
        
        Args:
            true_label: ground truth label (0 or 1)
            pred_label: predicted label (0 or 1)
            pred_score: prediction score/probability (optional)
        """
        self.y_true.append(true_label)
        self.y_pred.append(pred_label)
        if pred_score is not None:
            self.y_scores.append(pred_score)
    
    def get_metrics(self):
        """Calculate and return all classification metrics."""
        y_scores = self.y_scores if self.y_scores else None
        return Metrics.calculate_classification_metrics(
            self.y_true, self.y_pred, y_scores
        )
