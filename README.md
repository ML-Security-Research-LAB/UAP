# UAP (WACV 2026 oral)

Clean, minimal codebase for training and evaluating SSFL (Semi-Supervised Federated Learning) and UAP methods.

## Directory Structure

```
UAP/
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
├── config.py             # Unified configuration module
│
├── methods/              # Model implementations
│   ├── SSFL.py          # Semi-supervised federated learning baseline
│   ├── UAP.py           # UAP method
│   └── base.py          # Base model class
│
├── datasets.py           # Dataset classes (PACS, VLCS, OfficeHome, etc.)
├── multi_client.py       # Multi-client data loading
├── utils.py              # Utility functions (model loading, aggregation, etc.)
├── losses/               # Loss functions (CDD, MMD, covariance)
├── mnist_datasets.py     # MNIST dataset utilities
│
├── requirements.txt      # Python dependencies
└── BATCHNORM_STATS.md   # Documentation on BN stats sharing
```

## Quick Start

### 1. Install Dependencies
```bash
conda create -n uap python=3.8
conda activate uap
pip install -r requirements.txt
```

### 2. Training
```bash
# Train SSFL on PACS
python train.py --dataset PACS --test_env 0 --method SSFL --device cuda:0

# Train UAP on PACS
python train.py --dataset PACS --test_env 0 --method UAP --device cuda:0
```

### 3. Evaluation
```bash
# Evaluate SSFL
python evaluate.py --dataset PACS --test_env 0 --method SSFL --experiment_path data_final --device cuda:0

# Evaluate UAP
python evaluate.py --dataset PACS --test_env 0 --method UAP --experiment_path data_final --device cuda:0
```

## Key Features

### SSFL (Semi-Supervised Federated Learning)
- Server trains on labeled data from one domain
- Clients train on unlabeled data from other domains
- Batch normalization statistics are shared (privacy-preserving)
- Inspired by SemiFL

### UAP (Unified Adversarial Perturbation)
- Extends SSFL with domain-invariant feature learning
- Uses adversarial training for better generalization

## Configuration

### New Way (Recommended)
```python
from config import get_args
args = get_args()
```

```

## Important Arguments

- `--dataset`: Dataset name (PACS, VLCS, OfficeHome)
- `--test_env`: Target test domain (0-3 for PACS)
- `--server_domain`: Server labeled domain (0-3)
- `--method`: Method name (SSFL, UAP)
- `--multi_client`: Enable multi-client setting
- `--num_clients`: Number of clients in multi-client mode
- `--rounds`: Number of federated rounds
- `--E`: Local epochs per round
- `--device`: CUDA device (cuda:0, cuda:1, cpu)

## Datasets

Required datasets:
- PACS: 4 domains (art_painting, cartoon, photo, sketch)
- VLCS: 4 domains
- OfficeHome: 4 domains

Place datasets in `./data/` directory.

## Results

Checkpoints and logs are saved to:
```
results/{experiment_path}/{dataset}/{dataset}/{method}/target_{test_env}/server_{server_domain}/seed_{seed}/
```

Each checkpoint includes:
- `checkpoint.pt`: Model weights and optimizer state
- `target_accs.csv`: Target domain accuracies per round

## Documentation

- `BATCHNORM_STATS.md`: Explains batch normalization statistics sharing and privacy preservation

## Module Dependencies

### train.py
```
train.py
  ├── config.py (or argument.py)
  ├── utils.py
  │   └── methods/ (SSFL, UAP)
  │       └── base.py
  ├── datasets.py
  └── multi_client.py
```

### evaluate.py
```
evaluate.py
  ├── prepare.py (or config.py)
  ├── utils.py
  ├── datasets.py
  └── losses/
```

