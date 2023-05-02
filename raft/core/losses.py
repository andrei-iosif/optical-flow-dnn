import math

import torch
import torch.nn as nn


# Maximum value for predicted optical flow magnitude
# => used to exclude extremely large displacements
MAX_FLOW = 400.0


def l1_loss(flow_pred, flow_gt, valid_mask=None):
    loss_img = torch.sum((flow_pred - flow_gt).abs(), dim=1)
    if valid_mask is not None:
        return (valid_mask * loss_img).mean()
    else:
        return loss_img.mean()

def l1_loss_fixed(flow_pred, flow_gt, valid_mask=None):
    loss_img = torch.sum((flow_pred - flow_gt).abs(), dim=1)
    if valid_mask is not None:
        return torch.sum(loss_img * valid_mask) / torch.sum(valid_mask)
    else:
        return loss_img.mean()


def endpoint_error(flow_pred, flow_gt, valid_mask=None):
    """ Compute EPE metric between predicted flow and GT flow.

    Args:
        flow_pred (torch.Tensor): Predicted flow, shape [B, 2, H, W]
        flow_gt (torch.Tensor): GT flow, shape [B, 2, H, W]
        valid_mask (torch.Tensor, optional): Flow validity mask, shape [B, 1, H, W]. Defaults to None.

    Returns:
        EPE for each pixel, flattened to shape [B*H*W]
    """
    epe = torch.sum((flow_pred - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid_mask.view(-1)]
    return epe


class RaftLoss(nn.Module):
    """ Original loss, as defined in the paper. 
    
    Computes the L1 distance between each intermediate prediction and the ground truth.
    Final loss is weighted sum of intermediate losses, with exponentially increasing weights (parameterized by gamma).
    Early predictions are given less weight than later predictions.

    Also computes the following metrics for the last prediction:
        - EPE (end-point error): L2 distance between prediction and GT
        - <K>px: percentage of pixels for which EPE is smaller than K pixel(s)
    """

    def __init__(self, gamma=0.8, max_flow=MAX_FLOW, debug_iter=False):
        """ Initialize loss.

        Args:
            gamma (float, optional): Weighting factor for sequence loss. Defaults to 0.8.
            max_flow (float, optional): Maximum flow magnitude. Defaults to MAX_FLOW.        
        """
        super(RaftLoss, self).__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.debug = debug_iter


    def forward(self, flow_preds, flow_gt, valid_mask):
        """ Compute loss.

        Args:
            flow_preds (list(torch.Tensor)): List of intermediate flow predictions.
            flow_gt (torch.Tensor): Flow ground truth.
            valid_mask (torch.Tensor): Flow validity mask; used to compute loss only for valid GT positions.
            debug_iter (bool, optional): If True, save metrics for intermediate flow refinement iterations. Defaults to False
            
        Returns:
            (flow loss, additional metrics dict)
        """
        num_predictions = len(flow_preds)
        total_flow_loss = 0.0

        # Exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid_mask = (valid_mask >= 0.5) & (mag < self.max_flow)

        # DEBUG MODE
        if self.debug:
            flow_loss_list = []
            epe_list = []

        # Compute L1 loss for each prediction in the sequence
        for i in range(num_predictions):
            weight = self.gamma ** (num_predictions - i - 1)
            flow_loss = l1_loss_fixed(flow_preds[i], flow_gt, valid_mask)
            total_flow_loss += weight * flow_loss

            # DEBUG MODE
            if self.debug:
                flow_loss_list.append(flow_loss.item())
                epe_list.append(endpoint_error(flow_preds[i], flow_gt, valid_mask).mean().item())

        # Compute EPE and other metrics for most recent prediction
        epe = endpoint_error(flow_preds[-1], flow_gt, valid_mask) 

        metrics = {
            'flow_loss': total_flow_loss.item(),
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        # DEBUG MODE
        # Save additional metrics
        if self.debug:
            for i in range(num_predictions - 1):
                metrics[f'flow_loss_iter_{i}'] = flow_loss_list[i]
                metrics[f'epe_iter_{i}'] = epe_list[i]

        return total_flow_loss, metrics


class RaftSemanticLoss(nn.Module):
    """ Similar to RaftLoss, with an additional term that uses the semantic labels. 
    """

    def __init__(self, gamma=0.8, max_flow=MAX_FLOW, w_smooth=0.5, debug_iter=False):
        """ Initialize loss.

        Args:
            gamma (float, optional): Weighting factor for sequence loss. Defaults to 0.8.
            max_flow (float, optional): Maximum flow magnitude. Defaults to MAX_FLOW.        
            w_semantic (float, optional): Weighting factor for semantic smoothness loss. Defaults to 0.5
            debug_iter (bool, optional): If True, save metrics for intermediate flow refinement iterations. Defaults to False
        """
        super(RaftSemanticLoss, self).__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.w_smooth = w_smooth
        self.debug = debug_iter

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

    def get_semantic_smoothness_loss(self, flow_pred, semseg_gt_1, semseg_gt_2, flow_valid_mask):
        """ Compute semantic smoothness loss, that ensures flow discontinuities are 
        correlated with semantic discontinuities.

        Args:
            flow_pred (torch.Tensor): Flow prediction, shape [B, 2, H, W]
            semseg_gt_1 (torch.Tensor): Semantic GT for first image, shape [B, H, W]
            semseg_gt_2 (torch.Tensor): Semantic GT for second image, shape [B, H, W]
            flow_valid_mask (torch.Tensor): Flow validity mask, shape [B, 1, H, W]

        Return:
            semantic smoothness loss (float)
        """

        # Flow gradients
        u_flow_grad_x, u_flow_grad_y = self.image_grads(flow_pred[:, 0, :, :])
        v_flow_grad_x, v_flow_grad_y = self.image_grads(flow_pred[:, 1, :, :])

        # Semseg gradients
        semseg_grad_x, semseg_grad_y = self.image_grads(semseg_gt_1)

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

        # Crop flow valid mask [B, H, W] -> [B, H-1, W-1]
        flow_valid_mask = flow_valid_mask[:, :, 1:, 1:]

        # Compute non-occlusion mask from semseg GT for both frames
        non_occlusion_mask = ((semseg_gt_1 - semseg_gt_2).abs() < 1e-5).float()
        non_occlusion_mask = non_occlusion_mask[:, 1:, 1:]

        loss_mask = non_occlusion_mask * flow_valid_mask
        return torch.sum(loss_mask * semantic_loss) / torch.sum(loss_mask)

    def forward(self, flow_preds, flow_gt, valid_mask, semseg_gt_1, semseg_gt_2):
        """ Compute loss.

        Args:
            flow_preds (list(torch.Tensor)): List of intermediate flow predictions, each with shape [B, 2, H, W].
            flow_gt (torch.Tensor): Flow ground truth, shape [B, 2, H, W]
            valid_mask (torch.Tensor): Flow validity mask; used to compute loss only for valid GT positions, shape [B, H, W].
            semseg_gt_1 (torch.Tensor): Semantic segmentation ground truth for first image, shape [B, H, W].
            semseg_gt_2 (torch.Tensor): Semantic segmentation ground truth for second image, shape [B, H, W].

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

        # DEBUG MODE
        if self.debug:
            flow_loss_list = []
            semantic_loss_list = []
            epe_list = []

        # Compute L1 flow loss and semantic smoothness loss for each prediction in the sequence
        for i in range(num_predictions):
            weight = self.gamma ** (num_predictions - i - 1)
            flow_loss = l1_loss_fixed(flow_preds[i], flow_gt, valid_mask)
            total_flow_loss += weight * flow_loss
            semantic_loss = self.get_semantic_smoothness_loss(flow_preds[i], semseg_gt_1, semseg_gt_2, valid_mask)
            total_semantic_loss += weight * semantic_loss

            # DEBUG MODE
            if self.debug:
                flow_loss_list.append(flow_loss.item())
                semantic_loss_list.append(semantic_loss.item())
                epe_list.append(endpoint_error(flow_preds[i], flow_gt, valid_mask).mean().item())

        total_loss = total_flow_loss + self.w_smooth * total_semantic_loss

        # Compute EPE and other metrics for most recent prediction
        epe = endpoint_error(flow_preds[-1], flow_gt, valid_mask)

        metrics = {
            'flow_loss': total_flow_loss.item(),
            'semantic_loss': total_semantic_loss.item(),
            'total_loss': total_loss.item(),
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        # DEBUG MODE
        # Save additional metrics
        if self.debug:
            for i in range(num_predictions - 1):
                metrics[f'flow_loss_iter_{i}'] = flow_loss_list[i]
                metrics[f'semantic_loss_iter_{i}'] = semantic_loss_list[i]
                metrics[f'epe_iter_{i}'] = epe_list[i]

        return total_loss, metrics
        

class RaftUncertaintyLoss(nn.Module):
    def __init__(self, gamma=0.8, max_flow=MAX_FLOW, min_variance=1e-4):
        super(RaftUncertaintyLoss, self).__init__()
        self.gamma = gamma 
        self.max_flow = max_flow
        self.min_variance = min_variance

    def negative_log_likelihood_loss(self, flow_pred, flow_gt, flow_valid_mask):
        """ Compute negative log-likelihood loss for Gaussian predictions.

        Loss is defined as loss = log(sigma) + ((gt - mean)^2 / sigma)^(1/2)
        See https://arxiv.org/pdf/1805.11327.pdf

        Args:
            flow_pred (tuple(torch.Tensor)): Flow mean and variance, each with shape [B, 2, H, W]
            flow_gt (torch.Tensor): Flow ground truth, with shape [B, 2, H, W]
            flow_valid_mask (torch.Tensor): Flow validity mask, shape [B, 1, H, W]

        Returns:
            Loss value (float)
        """
        # pred_mean, pred_variance = flow_pred
        # pred_variance = pred_variance + self.min_variance
        # nll_loss = torch.abs(flow_gt - pred_mean) / pred_variance + torch.log(pred_variance)

        pred_mean, pred_log_variance = flow_pred
        nll_loss = torch.abs(flow_gt - pred_mean) * torch.exp(-pred_log_variance) + pred_log_variance

        return (flow_valid_mask * nll_loss).mean()

    def nll_loss_v3(self, flow_pred, flow_gt, flow_valid_mask):
        # https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html

        pred_mean, pred_variance = flow_pred
        pred_variance = torch.clamp(pred_variance, min=self.min_variance)
        nll_loss = 0.5 * (torch.abs(flow_gt - pred_mean) / pred_variance + torch.log(pred_variance))
        nll_loss += 0.5 * math.log(2 * math.pi)

        return (flow_valid_mask * nll_loss).mean()

    def endpoint_error(self, flow_pred, flow_gt):
        flow_pred_mean, _ = flow_pred
        return torch.sum((flow_pred_mean[-1] - flow_gt) ** 2, dim=1).sqrt()

    def forward(self, flow_preds, flow_gt, flow_valid_mask):
        num_predictions = len(flow_preds)
        total_loss = 0.0

        # Exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid_mask = (flow_valid_mask >= 0.5) & (mag < self.max_flow)

        # Convert from [B, H, W] to [B, 1, H, W]
        valid_mask = valid_mask[:, None]

        for i in range(num_predictions):
            loss_weight = self.gamma ** (num_predictions - i - 1)
            loss = self.nll_loss_v3(flow_preds[i], flow_gt, valid_mask)

            # Final loss is weighted sum of losses for each flow refinement iteration
            total_loss += loss_weight * loss

        epe = self.endpoint_error(flow_preds[-1], flow_gt)
        epe = epe.view(-1)[valid_mask.view(-1)]

        metrics = {
            'flow_loss': total_loss.item(),
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }
        return total_loss, metrics
