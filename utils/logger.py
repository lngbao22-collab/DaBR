# coding: utf-8
"""
Logging utilities for DaBR training and testing.
"""

import os
import logging
import json
from datetime import datetime


def setup_logger(name, log_dir='logs', level=logging.INFO):
    """
    Setup a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file


class ResultReporter:
    """Reports and saves training results."""
    
    def __init__(self, result_dir):
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        self.results = {}
    
    def add_metric(self, key, value):
        """Add a metric to the results."""
        self.results[key] = value
    
    def add_metrics(self, metrics_dict):
        """Add multiple metrics."""
        self.results.update(metrics_dict)
    
    def save_json(self, filename='results.json'):
        """Save results to JSON file."""
        filepath = os.path.join(self.result_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        return filepath
    
    def save_text(self, filename='results.txt'):
        """Save results to text file."""
        filepath = os.path.join(self.result_dir, filename)
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            for key, value in self.results.items():
                if isinstance(value, float):
                    f.write(f"{key:.<40} {value:.6f}\n")
                elif isinstance(value, int):
                    f.write(f"{key:.<40} {value}\n")
                else:
                    f.write(f"{key:.<40} {value}\n")
            
            f.write("=" * 80 + "\n")
        return filepath
    
    def print_results(self):
        """Print results to console."""
        print("=" * 80)
        print("EXPERIMENT RESULTS")
        print("=" * 80)
        for key, value in self.results.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:.6f}")
            else:
                print(f"{key:.<40} {value}")
        print("=" * 80)
