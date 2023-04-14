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

    def __init__(self, gamma=0.8, max_flow=MAX_FLOW, w_smooth=0.5):
        """ Initialize loss.

        Args:
            gamma (float, optional): Weighting factor for sequence loss. Defaults to 0.8.
            max_flow (float, optional): Maximum flow magnitude. Defaults to MAX_FLOW.        
            w_semantic (float, optiona): Weighting factor for semantic smoothness loss. Defaults to 0.5
        """
        super(RaftSemanticLoss, self).__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.w_smooth = w_smooth

    @staticmethod
    def image_grads(img):
        """ Compute vertical and horizontal image gradients for a batch of images.

        Args:
            img (torch.Tensor): Input image batch, shape [B, H, W]

        Returns:
            tuple of vertical gradients (shape [B, H, W-1]) and horizontal gradients (shape [B, H-1, W])
        """
        # Vertical gradients
        gx = img[:, :, :-1] - img[:, :, 1:]

        # Horizontal gradients
        gy = img[:, :-1, :] - img[:, 1:, :]
        return gx, gy

    def get_semantic_smoothness_loss(self, flow_pred, semseg_gt, valid_mask):
        """ Compute semantic smoothness loss, that ensures flow discontinuities are 
        correlated with semantic discontinuities.

        Args:
            flow_pred (torch.Tensor): Flow prediction, shape [B, 2, H, W]
            semseg_gt (torch.Tensor): Semantic GT, shape [B, 3, H, W]
            valid_mask (torch.Tensor): Flow validity mask, shape [B, 1, H, W]

        Return:
            semantic smoothness loss (float)
        """

        # Flow gradients
        u_flow_grad_x, u_flow_grad_y = self.image_grads(flow_pred[:, 0, :, :])
        v_flow_grad_x, v_flow_grad_y = self.image_grads(flow_pred[:, 1, :, :])

        # Semseg gradients
        semseg_grad_x, semseg_grad_y = self.image_grads(semseg_gt)

        # If semseg gradient is zero => penalize flow discontinuities
        # If semseg gradient is not zero => no penalty
        semseg_weight_x = (semseg_grad_x.abs() < 1e-5).float()
        semseg_weight_y = (semseg_grad_y.abs() < 1e-5).float()

        loss_x = semseg_weight_x * torch.abs(u_flow_grad_x) + semseg_weight_x * torch.abs(v_flow_grad_x)
        loss_y = semseg_weight_y * torch.abs(u_flow_grad_y) + semseg_weight_y * torch.abs(v_flow_grad_y)

        # Crop to same dimensions
        # loss_x [B, H, W-1] -> [B, H-1, W-1]
        # loss_y [B, H-1, W] -> [B, H-1, W-1]
        semantic_loss = loss_x[:, 1:, :] + loss_y[:, :, 1:]

        # Convert from [B, H, W] to [B, 1, H, W]
        semantic_loss = semantic_loss[:, None]

        # Crop valid mask [B, H, W] -> [B, H-1, W-1]
        valid_mask = valid_mask[:, :, 1:, 1:]

        return (valid_mask * semantic_loss).mean()

    def forward(self, flow_preds, flow_gt, valid_mask, semseg_gt_1, semseg_gt_2):
        """ Compute loss.

        Args:
            flow_preds (list(torch.Tensor)): List of intermediate flow predictions, each with shape [B, 2, H, W].
            flow_gt (torch.Tensor): Flow ground truth, shape [B, 2, H, W]
            valid_mask (torch.Tensor): Flow validity mask; used to compute loss only for valid GT positions, shape [B, H, W].
            semseg_gt_1 (torch.Tensor): Semantic segmentation ground truth for first image, shape [B, 3, H, W].
            semseg_gt_2 (torch.Tensor): Semantic segmentation ground truth for second image, shape [B, 3, H, W].

        Returns:
            (total loss, additional metrics dict)
        """
        num_predictions = len(flow_preds)
        total_flow_loss = 0.0
        total_semantic_loss = 0.0

        # Exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid_mask = (valid_mask >= 0.5) & (mag < self.max_flow)

        # Convert from [B, H, W] to [B, 1, H, W]
        valid_mask = valid_mask[:, None]

        # Compute L1 flow loss and semantic smoothness loss for each prediction in the sequence
        for i in range(num_predictions):
            weight = self.gamma ** (num_predictions - i - 1)
            flow_loss = (flow_preds[i] - flow_gt).abs()
            total_flow_loss += weight * (valid_mask * flow_loss).mean()
            semantic_loss = self.get_semantic_smoothness_loss(flow_preds[i], semseg_gt_1, valid_mask)
            total_semantic_loss += weight * semantic_loss

        total_loss = total_flow_loss + self.w_smooth * total_semantic_loss

        # Compute EPE and other metrics for most recent prediction
        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid_mask.view(-1)]

        metrics = {
            'flow_loss': total_flow_loss.item(),
            'semantic_loss': total_semantic_loss.item(),
            'total_loss': total_loss.item(),
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return total_loss, metrics

