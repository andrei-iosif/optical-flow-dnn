import torch


def EPE(pred_flow, gt_flow):
    """
    Endpoint error (EPE) metric for flow estimation. Equal to Euclidean distance between predicted flow and ground truth
    flow, averaged over all pixels.
    :param pred_flow: predicted flow; shape B x 2 x H x W
    :param gt_flow: ground truth flow; shape B x 2 x H x W
    :return: EPE value
    """
    return torch.linalg.norm(gt_flow - pred_flow, ord=2, dim=1).mean()
