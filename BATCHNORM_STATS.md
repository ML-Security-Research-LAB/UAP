# Batch Normalization Statistics and Test Model

## `make_batchnorm_stats()`
Creates a copy of the trained model (`test_model`) and recalculates batch normalization running statistics using data from both server and clients. This enables domain-adaptive normalization by computing BN statistics on mixed-domain data instead of single-domain data.

## `test_model`
A deep copy of the trained model with recalculated BN statistics. Used during evaluation to provide better domain generalization. Only BN layer statistics (running mean/variance) differ from the original model; learned weights remain identical.

## Privacy Preservation
Only BN statistics (running mean and variance) are computed and shared between server and clients. No raw data is exchanged. Clients compute statistics locally on their data, and only aggregate statistical moments are used.

## Source
This approach is inspired by **SemiFL** (Semi-supervised Federated Learning), which uses BN statistics sharing for domain adaptation without violating privacy constraints.
