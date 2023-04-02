import torch
import torch.nn as nn


class RaftSequenceLoss(nn.Module):
    def forward(self, flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)

        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics


def elementwise_laplacian(flow_pred, flow_gt):
    predictions_mean, predictions_variance = flow_pred

    log_var = torch.sum(torch.log(predictions_variance), dim=1, keepdim=True)
    abs_diff = (flow_gt - predictions_mean).abs()

    weighted_epe = torch.sqrt(
        torch.sum(abs_diff / predictions_variance, dim=1, keepdim=True))

    return weighted_epe + log_var


def endpoint_error(flow_pred, flow_gt):
    flow_pred_mean, _ = flow_pred
    return torch.sum((flow_pred_mean[-1] - flow_gt) ** 2, dim=1).sqrt()


class LaplacianLogLikelihoodLoss(nn.Module):
    def __init__(self, gamma=0.8, max_flow=400):
        super(LaplacianLogLikelihoodLoss, self).__init__()
        self.gamma = gamma 
        self.max_flow = max_flow

    def forward(self, flow_preds, flow_gt, flow_valid_mask):
        n_predictions = len(flow_preds)
        total_loss = 0.0

        # Exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid = (flow_valid_mask >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            loss_weight = self.gamma ** (n_predictions - i - 1)
            loss = elementwise_laplacian(flow_preds[i], flow_gt)

            # Final loss is weighted sum of losses for each flow refinement iteration
            total_loss += loss_weight * (valid[:, None] * loss).mean()

        epe = endpoint_error(flow_preds[-1], flow_gt)
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'flow_loss': total_loss.mean().item(),
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return total_loss, metrics