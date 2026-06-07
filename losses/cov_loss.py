import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

def get_vne(H):
    Z = torch.nn.functional.normalize(H, dim=1)
    sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
    eig_val = sing_val ** 2
    return - (eig_val * torch.log(eig_val)).nansum()

def compute_covariance(feats):
    """
    Compute the empirical covariance matrix from a batch of feature vectors.

    Args:
        feats (torch.Tensor): A tensor of shape (B, d) containing B feature vectors.

    Returns:
        torch.Tensor: A (d, d) covariance matrix.
    """
    B = feats.size(0)
    # Compute the mean of the features (shape: (1, d))
    mean_feats = torch.mean(feats, dim=0, keepdim=True)
    # Center the features by subtracting the mean
    centered_feats = feats - mean_feats
    # Compute the covariance matrix; using (B-1) for an unbiased estimator
    cov_matrix = (centered_feats.T @ centered_feats) / (B - 1)
    return cov_matrix

def eigenvalue_variance_penalty(cov, gamma=1.0):
    """
    Computes a penalty proportional to the variance of the eigenvalues of a 
    symmetric (covariance) matrix 'cov' without doing an explicit eigen-decomposition.
    
    Parameters:
      cov (torch.Tensor): A symmetric matrix of shape (n, n).
      gamma (float): Regularization strength.
    
    Returns:
      torch.Tensor: The penalty value.
    """
    n = cov.shape[0]
    trace = torch.trace(cov)
    frob_norm_sq = torch.sum(cov * cov)  # Equivalent to torch.norm(cov, 'fro')**2
    # Compute the sum of squared deviations of eigenvalues (unnormalized variance)
    variance_sum = frob_norm_sq - (trace**2) / n
    # Optionally, you might want to normalize by n to get the average variance.
    return gamma * variance_sum / n

# class CovLoss(nn.Module):

#     def __init__(self, batch_size, num_features, device):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_features = num_features
#         self.I = torch.eye(num_features).to(device) 
        
#     def forward(self, x):
#         # x = F.normalize(x, dim=0)
#         # cov = compute_covariance(x)
#         # loss = cov.norm(p=2)
#         # loss = torch.norm(cov, p='fro')
#         # loss = eigenvalue_variance_penalty(cov)
#         # loss = get_vne(x)
#         x = x - x.mean(dim=0)
#         cov_x = (x.T @ x) / (self.batch_size - 1)
#         # loss = eigenvalue_variance_penalty(cov_x)
#         loss = (cov_x - self.I).pow_(2).sum().div(self.num_features)
#         # print(self.I.abs().sum(), self.I.shape[0] * 0.01)
#         return loss


class CovLoss(nn.Module):

    def __init__(self, batch_size, num_features, device):
        super().__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        self.I = torch.eye(num_features).to(device) 
        
    def forward(self, x):
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (self.batch_size - 1)
        loss = (cov_x - self.I).pow_(2).sum().div(self.num_features)
        return loss


# class CovLoss(nn.Module):

#     def __init__(self, batch_size, num_features, device):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_features = num_features
#         self.I = torch.eye(num_features).to(device)
        
#     def forward(self, x):
#         x = x - x.mean(dim=0)
#         cov_x = (x.T @ x) / (self.batch_size - 1)
#         # Compute the squared differences, then mask out the diagonal.
#         loss = (((cov_x - self.I)**2) * (1 - self.I)).sum().div(self.num_features)
#         return loss
    

# class CovLoss(nn.Module):

#     def __init__(self, batch_size, num_features, device):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_features = num_features
#         self.I = torch.eye(num_features).to(device) 
#         self.lambda_min = 1.0 * 512
        
#     def forward(self, x):
#         cov = compute_covariance(x)

#         # Compute sum of diagonal
#         diag_sum = cov.diag().sum()

#         # Apply penalty if sum is below lambda_min
#         penalty = self.penalty_coeff * torch.relu(self.lambda_min - diag_sum)

#         # Loss: Minimize sum of diagonal while ensuring it is above lambda_min
#         loss = diag_sum + penalty

#         return loss
    
class CovLossWithTargets(nn.Module):

    def __init__(self, batch_size, num_features, device):
        super().__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        
    def forward(self, x, target_covs):
        # x = F.normalize(x, dim=0)
        cov = compute_covariance(x)
        # loss = cov.norm(p=2)
        # loss = torch.norm(cov, p='fro')
        loss = [(cov - target_cov).pow_(2).sum().div(self.num_features) for target_cov in target_covs]
        loss = sum(loss) / len(loss)
        # loss = get_vne(x)
        # x = x - x.mean(dim=0)
        # cov_x = (x.T @ x) / (self.batch_size - 1)
        # loss = eigenvalue_variance_penalty(cov_x)
        # loss = (cov_x - self.I).pow_(2).sum().div(self.num_features)
        # print(self.I.abs().sum(), self.I.shape[0] * 0.01)
        return loss


# class CovLoss(nn.Module):

#     def __init__(self, batch_size, num_features, device):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_features = num_features
#         self.I = torch.eye(num_features).to(device)
        
#     def forward(self, x):
#         x = x - x.mean(dim=0)
#         cov_x = (x.T @ x) / (x.shape[0] - 1)
#         mask = torch.ones_like(cov_x) - self.I
#         off_diag_loss = (mask * cov_x).pow_(2).sum().div(self.num_features * (self.num_features - 1))
#         return off_diag_loss