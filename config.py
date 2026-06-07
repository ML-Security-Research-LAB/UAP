"""
Configuration module for UAP experiments.
Consolidates argument parsing and preprocessing.
"""
import argparse
import os
import torch
import random
import numpy as np

def create_parser():
    """Create argument parser with all configuration options."""
    parser = argparse.ArgumentParser(description='UAP Federated Learning Experiments', conflict_handler='resolve')

    # Dataset and environment settings
    parser.add_argument('--dataset', type=str, default='PACS', help='Dataset name (PACS, VLCS, OfficeHome, etc.)')
    parser.add_argument('--test_env', type=int, default=0, help='Test domain index')
    parser.add_argument('--server_domain', type=int, default=1, help='Server domain index')
    parser.add_argument('--num_domains', type=int, default=4, help='Number of domains')
    parser.add_argument('--dataset_folder', type=str, default='./data', help='Path to dataset folder')
    parser.add_argument('--experiment_path', type=str, default='./all_results/', help='Path to save results')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')

    # Model settings
    parser.add_argument('--method', type=str, default='SSFL', help='Method name (SSFL, UAP)')
    parser.add_argument('--back_bone', type=str, default='resnet18', help='Backbone architecture')
    parser.add_argument('--z_dim', type=int, default=512, help='Feature dimension')

    # Training settings
    parser.add_argument('--optim', type=str, default='SGD', help='Optimizer (SGD, Adam)')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--E', type=int, default=5, help='Number of local updates per round')
    parser.add_argument('--rounds', type=int, default=40, help='Number of federated rounds')

    # Learning rate scheduler settings
    parser.add_argument('--lr_scheduler', default=True, action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'step'], help='Scheduler type')
    parser.add_argument('--lr_reduction_factor', type=float, default=0.1, help='LR reduction factor')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Step size for step scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Gamma for step scheduler')
    parser.add_argument('--min_lr', type=float, default=0, help='Minimum learning rate')

    # Method-specific settings
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha hyperparameter')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta hyperparameter')
    parser.add_argument('--lambda_', type=float, default=0.01, help='Lambda hyperparameter')
    parser.add_argument('--uniform_weights', action='store_true', help='Use uniform aggregation weights')
    parser.add_argument('--skip_bn', default=True, action='store_true', help='Skip batch normalization adaptation')

    # Multi-client settings
    parser.add_argument('--multi_client', action='store_true', help='Use multi-client setting')
    parser.add_argument('--data_distribution', type=str, default='single-domain',
                       choices=['single-domain', 'multi-domain'], help='Data distribution strategy')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients in multi-client setting')
    parser.add_argument('--labeled_data', type=int, default=0, help='Number of labeled samples per client')

    # Other settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0, cuda:1, cpu)')
    parser.add_argument('--verbose', default=True, action='store_true', help='Verbose output')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--ext', type=str, default='', help='Experiment name extension')

    return parser


def get_args(parse=True, print_args=True):
    """
    Get parsed arguments with optional preprocessing.

    Args:
        parse: If True, parse arguments. If False, return parser only.
        print_args: If True, print parsed arguments.

    Returns:
        Parsed arguments namespace or parser object.
    """
    parser = create_parser()

    if not parse:
        return parser

    args = parser.parse_args()

    # Post-process arguments (convert string booleans)
    for name in args.__dict__:
        value = getattr(args, name)
        if value in ['True', 'False', 'None']:
            setattr(args, name, eval(value))
        if callable(value):
            # Handle callable defaults (if any)
            setattr(args, name, value(args.seed))

    if print_args:
        print(args)

    return args


# For backward compatibility - maintain old interface
parser = create_parser()
args = get_args(parse=True, print_args=False)  # Don't auto-print when imported
