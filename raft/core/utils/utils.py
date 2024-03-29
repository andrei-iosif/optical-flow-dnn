import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht), indexing="ij")

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Bilinear sampling. Wrapper over grid_sample.

    Args:
        img (torch.Tensor): Image to sample from, shape [B, C, H_in, W_in]
        coords (torch.Tensor): Sampling coordinates, shape [B, H_out, W_out, 2]
        mode (str, optional): Sampling mode. Defaults to 'bilinear'.
        mask (bool, optional): If true, return a mask containing 1 if sampling coordinates are inside input image, and 0 otherwise. Defaults to False.

    Returns:
        Sampled image, with shape [B, C, H_out, W_out]
    """

    H, W = img.shape[-2:]

    # Normalize sampling coordinates to [-1, 1]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W-1) - 1
    ygrid = 2 * ygrid / (H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode=mode)

    # Compute mask for sampling coords inside/outside input image
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    """ Create coordinate grids for a batch of image pairs.

    Args:
        batch (int): Batch size
        ht (int): Image height
        wd (int): Image width
        device (str): Target device for created tensor

    Returns:
        Coordinate grids with shape [B, 2, H, W]
    """
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def endpoint_error_numpy(flow_pred, flow_gt, valid_mask=None):
    """ Compute EPE metric between predicted flow and GT flow.

    Args:
        flow_pred (np.ndarray): Predicted flow, shape [2, H, W]
        flow_gt (np.ndarray): GT flow, shape [2, H, W]
        valid_mask (np.ndarray, optional): Flow validity mask, shape [H, W]. Defaults to None.

    Returns:
        EPE for each pixel, flattened to shape [H*W]
    """
    epe = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=0))

    if valid_mask is not None:
        return epe * valid_mask
    return epe
