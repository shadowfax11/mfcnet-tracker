import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 

def get_loss(outputs, targets, loss_fns, loss_wts, args):
    loss_dict = {} 
    total_loss = 0.0
    for loss_fn, loss_wt in zip(loss_fns, loss_wts):
        if loss_fn == 'mse':
            loss = LossMSE()(outputs, targets)
        elif loss_fn == 'nll':
            loss = LossNLL(class_weights=args.class_weights, num_classes=args.num_classes)(outputs, targets)
        elif loss_fn == 'soft_jaccard':
            loss = LossSoftJaccard(num_classes=args.num_classes)(outputs, targets)
        else: 
            raise ValueError(f'Loss function {loss_fn} not implemented')
        total_loss += loss_wt * loss
        loss_dict['loss_' + loss_fn] = loss.item()
    loss_dict['loss_total'] = total_loss.item()
    return total_loss, loss_dict

class LossMSE:
    def __init__(self):
        self.mse_loss = nn.MSELoss()
    
    def __call__(self, outputs, targets):
        loss = self.mse_loss(outputs, targets)
        return loss

class LossNLL: 
    def __init__(self, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32))
            nll_weight = nll_weight.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.num_classes = num_classes
    
    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        return loss

class LossSoftJaccard:
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.eps = 1e-15  # Small constant to avoid division by zero

    def __call__(self, outputs, targets):
        loss = 0.0  # Initialize total loss
        for cls in range(1, self.num_classes):  # Exclude background class
            # Create binary masks for the current class
            jaccard_target = (targets == cls).float()
            jaccard_output = outputs[:, cls].exp()  # Assuming outputs are logits, use exp to get probabilities
            # Compute intersection and union
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum() - intersection
            # Compute Jaccard loss for the current class and accumulate
            jaccard_loss = -torch.log((intersection + self.eps) / (union + self.eps))
            loss += jaccard_loss
        # Return the average loss over all classes
        return loss / self.num_classes

class LossWassersteinDistance(nn.Module):  # Inherit from nn.Module to use register_buffer
    def __init__(self, num_classes, image_size, normalize=True):
        super(LossWassersteinDistance, self).__init__()
        self.num_classes = num_classes
        self.eps = 1e-15  # Small constant to avoid division by zero
        self.normalize = normalize
        self.image_size = image_size  # Fixed image size (height, width)

        # Compute the cost matrix based on the fixed image size and register it as a buffer
        cost_matrix = self.compute_cost_matrix(*image_size)
        self.register_buffer('cost_matrix', cost_matrix)  # Register the cost matrix as a buffer

    def compute_cost_matrix(self, height, width):
        """
        Computes the cost matrix, which is the pairwise distance between each pixel location.
        This matrix represents the cost of transporting mass between pixels.
        """
        # Create a grid of coordinates for each pixel in the mask
        x = torch.arange(width).float()
        y = torch.arange(height).float()
        X, Y = torch.meshgrid(x, y)

        # Flatten the grid and calculate pairwise Euclidean distances
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        dist_matrix = torch.cdist(coords, coords, p=2)  # p=2 for Euclidean distance
        return dist_matrix

    def forward(self, outputs, targets):
        """
        Compute the Wasserstein distance loss between predicted and target masks.
        """
        loss = 0.0  # Initialize total loss

        batch_size, _, height, width = outputs.size()

        # Use the stored cost matrix buffer
        cost_matrix = self.cost_matrix

        for cls in range(self.num_classes):
            # Extract soft predictions (probabilities) and create binary target masks for the current class
            target_mask = (targets == cls).float().view(batch_size, -1)  # Flatten the target mask
            pred_mask = outputs[:, cls].exp().view(batch_size, -1)  # Flatten the predicted mask (assume logits passed)

            # Normalize the masks to sum to 1 (valid probability distributions)
            if self.normalize:
                target_mask = target_mask / (target_mask.sum(dim=1, keepdim=True) + self.eps)
                pred_mask = pred_mask / (pred_mask.sum(dim=1, keepdim=True) + self.eps)

            # Compute Wasserstein distance (using the cost matrix and mass transport)
            wasserstein_distance = torch.sum(cost_matrix * (target_mask - pred_mask).abs(), dim=[1, 2])

            # Accumulate the Wasserstein distance for all classes
            loss += wasserstein_distance.mean()  # Average over batch

        return loss / self.num_classes  # Return average loss over classes
