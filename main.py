#!/usr/bin/env python
# coding: utf-8
"""
Unified training and testing script for DaBR model.
Uses JSON configuration files for dataset-specific settings.
"""

import os
import sys
import json
import time
import torch
import logging
from argparse import ArgumentParser
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from models.dabr import DaBR
from utils.logger import setup_logger, ResultReporter


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def apply_config(con, config_dict, args):
    """Apply configuration dictionary to Config object."""
    benchmark_path = config_dict.get('benchmark_path', f"./benchmarks/{config_dict['dataset_name']}/")
    con.set_in_path(benchmark_path)
    con.set_work_threads(config_dict.get('work_threads', 8))
    con.set_train_times(config_dict.get('num_epochs', 10000))
    con.set_nbatches(config_dict.get('nbatches', 100))
    con.set_alpha(config_dict.get('learning_rate', 0.1))
    con.set_bern(1)
    con.set_dimension(config_dict.get('hidden_size', 500))
    con.set_lmbda(config_dict.get('lmbda', 0.5))
    con.set_lmbda_two(config_dict.get('lmbda2', 0.01))
    con.set_margin(1.0)
    con.set_ent_neg_rate(config_dict.get('neg_num', 5))
    con.set_opt_method(config_dict.get('optim', 'adagrad'))
    con.set_save_steps(config_dict.get('save_steps', 400))
    con.set_valid_steps(config_dict.get('valid_steps', 400))
    con.set_early_stopping_patience(config_dict.get('early_stopping_patience', 10))
    con.set_test_link(config_dict.get('test_link', True))
    con.set_test_triple(config_dict.get('test_triple', False))

    if config_dict.get('test_triple', False):
        labels_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.normpath(benchmark_path)),
                f"{config_dict['dataset_name']}_w_labels",
            )
        )
        if os.path.isdir(labels_path):
            con.set_triple_classification_path(os.path.join(labels_path, ""))
    
    # Setup checkpoint and result directories
    dataset_name = config_dict['dataset_name']
    out_dir = os.path.abspath(os.path.join("logs", dataset_name))
    os.makedirs(out_dir, exist_ok=True)
    
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    
    # Generate model name
    if args.model_name is None or len(args.model_name.strip()) == 0:
        model_name = (
            f"{dataset_name}_lda-{config_dict.get('lmbda', 0.5)}_"
            f"nneg-{config_dict.get('neg_num', 5)}_"
            f"hs-{config_dict.get('hidden_size', 500)}_"
            f"lr-{config_dict.get('learning_rate', 0.1)}_"
            f"nepochs-{config_dict.get('num_epochs', 10000)}"
        )
    else:
        model_name = args.model_name
    
    result_dir = os.path.abspath(os.path.join(checkpoint_dir, model_name))
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    con.set_checkpoint_dir(checkpoint_dir)
    con.set_result_dir(result_dir)
    
    return result_dir


def main():
    parser = ArgumentParser("DaBR - Unified Training Script")
    parser.add_argument("--dataset", default="WN18RR", 
                        choices=["WN18RR", "FB15K237"],
                        help="Name of the dataset")
    parser.add_argument("--mode", default="train", 
                        choices=["train", "predict"],
                        help="Mode: train or predict")
    parser.add_argument("--model_name", default=None, 
                        help="Custom model name (optional)")
    parser.add_argument("--checkpoint_path", default=None, 
                        help="Path to checkpoint for prediction mode")
    parser.add_argument("--test_file", default="", 
                        help="Custom test file path")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device id to use")
    parser.add_argument("--config", default=None,
                        help="Path to custom config file (overrides dataset config)")
    
    args = parser.parse_args()
    
    # Setup logging
    logs_dir = os.path.join("logs", args.dataset)
    logger, log_file = setup_logger(f"DaBR-{args.dataset}", logs_dir)
    logger.info("="*80)
    logger.info(f"DaBR Training/Testing Started")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Mode: {args.mode}")
    logger.info("="*80)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(args.gpu_id)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.backends.cudnn.is_available():
            logger.info("cuDNN available: Yes")
            logger.info(f"cuDNN benchmark enabled: {torch.backends.cudnn.benchmark}")
    
    # Load configuration from JSON
    if args.config:
        config_path = args.config
    else:
        config_path = f"./config/{args.dataset}.json"
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config_dict = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    logger.info("Configuration details:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize Config object
    con = Config()
    
    # Apply configuration
    result_dir = apply_config(con, config_dict, args)
    logger.info(f"Result directory: {result_dir}")
    
    # Handle test file path
    if args.test_file != "":
        test_file_path = os.path.join(config_dict.get('benchmark_path', f"./benchmarks/{args.dataset}/"), 
                                      args.test_file)
    else:
        test_file_path = ""
    con.set_test_file_path(test_file_path)
    
    # Initialize backend
    logger.info("Initializing data...")
    con.init()
    logger.info(f"Entity total: {con.entTotal}")
    logger.info(f"Relation total: {con.relTotal}")
    logger.info(f"Train total: {con.trainTotal}")
    logger.info(f"Valid total: {con.validTotal}")
    logger.info(f"Test total: {con.testTotal}")
    
    # Create result reporter
    reporter = ResultReporter(result_dir)
    reporter.add_metrics({
        "dataset": args.dataset,
        "mode": args.mode,
        "entity_count": con.entTotal,
        "relation_count": con.relTotal,
        "train_triple_count": con.trainTotal,
        "valid_triple_count": con.validTotal,
        "test_triple_count": con.testTotal,
    })
    
    # Training mode
    if args.mode == "train":
        logger.info("="*80)
        logger.info("Starting training...")
        logger.info("="*80)
        start_time = time.time()
        
        con.set_train_model(DaBR)
        logger.info("Model initialized")
        
        best_model = con.training_model()
        test_metrics = con.last_test_metrics
        
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f}s)")
        
        # Add timing info to reporter
        reporter.add_metrics({
            "training_time_seconds": con.train_time,
            "validation_time_seconds": con.valid_time,
            "test_time_seconds": con.test_time,
            "total_time_seconds": con.total_time,
            "best_epoch": con.best_epoch,
            "best_mrr": con.best_mrr,
        })
        if test_metrics:
            reporter.add_metrics(test_metrics)
        
    # Prediction mode
    else:
        logger.info("="*80)
        logger.info("Starting prediction...")
        logger.info("="*80)
        if args.checkpoint_path is None:
            args.checkpoint_path = os.path.join(result_dir, DaBR.__name__ + ".ckpt")
        
        logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
        con.set_test_model(DaBR, args.checkpoint_path)
        
        test_start = time.time()
        test_metrics = con.test()
        test_time = time.time() - test_start
        logger.info(f"Test time: {test_time:.2f}s")
        
        reporter.add_metrics({
            "test_time_seconds": test_time,
        })
        if test_metrics:
            reporter.add_metrics(test_metrics)
    
    # Save results
    logger.info("="*80)
    logger.info("Saving results...")
    logger.info("="*80)
    
    json_path = reporter.save_json('results.json')
    logger.info(f"Results saved to: {json_path}")
    
    logger.info("="*80)
    logger.info("DaBR completed successfully!")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

