import torch
import torch.nn as nn


# Maximum value for predicted optical flow magnitude
# => used to exclude extremely large displacements
MAX_FLOW = 400.0


class RaftLoss(nn.Module):
    """ Original loss, as defined in the paper. 
    
    Computes the L1 distance between each intermediate prediction and the ground truth.
    Final loss is weighted sum of intermediate losses, with exponentially increasing weights (parameterized by gamma).
    Early predictions are given less weight than later predictions.

    Also computes the following metrics for the last prediction:
        - EPE (end-point error): L2 distance between prediction and GT
        - <K>px: percentage of pixels for which EPE is smaller than K pixel(s)
    """

    def __init__(self, gamma=0.8, max_flow=MAX_FLOW):
        """ Initialize loss.

        Args:
            gamma (float, optional): Weighting factor for sequence loss. Defaults to 0.8.
            max_flow (float, optional): Maximum flow magnitude. Defaults to MAX_FLOW.        
        """
        super(RaftLoss, self).__init__()
        self.gamma = gamma
        self.max_flow = max_flow


    def forward(self, flow_preds, flow_gt, valid_mask):
        """ Compute loss.

        Args:
            flow_preds (list(torch.Tensor)): List of intermediate flow predictions.
            flow_gt (torch.Tensor): Flow ground truth.
            valid_mask (torch.Tensor): Flow validity mask; used to compute loss only for valid GT positions.
            
        Returns:
            (flow loss, additional metrics dict)
        """
        num_predictions = len(flow_preds)
        flow_loss = 0.0

        # Exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid_mask = (valid_mask >= 0.5) & (mag < self.max_flow)

        # Compute L1 loss for each prediction in the sequence
        for i in range(num_predictions):
            i_weight = self.gamma ** (num_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid_mask[:, None] * i_loss).mean()

        # Compute EPE and other metrics for most recent prediction
        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid_mask.view(-1)]

        metrics = {
            'flow_loss': flow_loss.item(),
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics


class RaftSemanticLoss(nn.Module):
    """ Similar to RaftLoss, with an additional term that uses the semantic labels. 
    """

    def __init__(self, gamma=0.8, max_flow=MAX_FLOW):
        """ Initialize loss.

        Args:
            gamma (float, optional): Weighting factor for sequence loss. Defaults to 0.8.
            max_flow (float, optional): Maximum flow magnitude. Defaults to MAX_FLOW.        
        """
        super(RaftLoss, self).__init__()
        self.gamma = gamma
        self.max_flow = max_flow


    def forward(self, flow_preds, flow_gt, valid_mask, semseg_gt):
        """ Compute loss.

        Args:
            flow_preds (list(torch.Tensor)): List of intermediate flow predictions.
            flow_gt (torch.Tensor): Flow ground truth.
            valid_mask (torch.Tensor): Flow validity mask; used to compute loss only for valid GT positions.
            semseg_gt (torch.Tensor): Semantic segmentation ground.

        Returns:
            (flow loss, additional metrics dict)
        """
        num_predictions = len(flow_preds)
        flow_loss = 0.0

        # Exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid_mask = (valid_mask >= 0.5) & (mag < self.max_flow)

        # Compute L1 loss for each prediction in the sequence
        for i in range(num_predictions):
            i_weight = self.gamma ** (num_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid_mask[:, None] * i_loss).mean()

        # Compute EPE and other metrics for most recent prediction
        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid_mask.view(-1)]

        metrics = {
            'flow_loss': flow_loss.item(),
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics

