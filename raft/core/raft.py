import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import BasicUpdateBlock, SmallUpdateBlock
from core.extractor import BasicEncoder, SmallEncoder
from core.corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        # Parameters for correlation module and update module
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # Create feature encoder, context encoder and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        """ Freeze BatchNorm layers. """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Forward flow is represented as difference between two coordinate grids: flow = coords1 - coords0 """
        N, _, H, W = img.shape
        coords_0 = coords_grid(N, H//8, W//8, device=img.device)
        coords_1 = coords_grid(N, H//8, W//8, device=img.device)

        return coords_0, coords_1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image_1, image_2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames.

        Args:
            image_1 (torch.Tensor): First image, shape [B, 3, H, W]
            image_2 (torch.Tensor): Second image, shape [B, 3, H, W]
            iters (int, optional): Number of iterations for flow refinement. Defaults to 12.
            flow_init (torch.Tensor, optional): Initial optical flow (for 'warm start' training). Defaults to None.
            upsample (bool, optional): Whether to upsample the predicted optical flow back to input dimensions. Defaults to True.
            test_mode (bool, optional): If in test mode, return only the latest prediction. Defaults to False.

        Returns:
            In train mode, return a list of flow predictions. In test mode, return only the latest prediction.
        """

        # Normalize RGB images
        image_1 = 2 * (image_1 / 255.0) - 1.0
        image_2 = 2 * (image_2 / 255.0) - 1.0

        image_1 = image_1.contiguous()
        image_2 = image_2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # Run the feature encoder
        with autocast(enabled=self.args.mixed_precision):
            fmap_1, fmap_2 = self.fnet([image_1, image_2])        
        
        # Build correlation volume
        fmap_1 = fmap_1.float()
        fmap_2 = fmap_2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap_1, fmap_2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap_1, fmap_2, radius=self.args.corr_radius)

        # Run the context encoder
        # => outputs context features and initial hidden state for ConvGRU
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image_1)
            net, context_fmap = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            context_fmap = torch.relu(context_fmap)

        # Initialize flow
        coords_0, coords_1 = self.initialize_flow(image_1)
        if flow_init is not None:
            coords_1 = coords_1 + flow_init

        # Initialize flow variance
        N, _, H, W = image_1.shape
        flow_variance = torch.zeros((N, 2, H//8, W//8), device=image_1.device)

        # Iterative refinement
        flow_predictions = []
        for _ in range(iters):
            # Correlation volume look-up => motion features
            coords_1 = coords_1.detach()
            corr = corr_fn(coords_1) 

            # Run refinement step
            flow = coords_1 - coords_0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, flow_out = self.update_block(net, context_fmap, corr, flow)

            if self.args.uncertainty:
                # delta_flow, flow_variance = flow_out
                delta_flow, delta_flow_variance = flow_out
            else:
                delta_flow = flow_out

            # Update flow estimation
            # F(t+1) = F(t) + \Delta(t)
            coords_1 = coords_1 + delta_flow

            # Update flow variance estimation
            flow_variance = flow_variance + delta_flow_variance

            # Upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords_1 - coords_0)
                if self.args.uncertainty:
                    flow_variance_up = upflow8(flow_variance)
            else:
                flow_up = self.upsample_flow(coords_1 - coords_0, up_mask)
                # TODO: maybe need different upsampling weights for flow variance
                if self.args.uncertainty:
                    flow_variance_up = self.upsample_flow(flow_variance, up_mask)
            
            if self.args.uncertainty:
                flow_predictions.append((flow_up, flow_variance_up))
            else:
                flow_predictions.append(flow_up)

        if test_mode:
            if self.args.uncertainty:
                return flow_variance_up, flow_up
            else:
                return _, flow_up
            
        return flow_predictions


if __name__ == "__main__":
    from torchinfo import summary

    import argparse
    args = argparse.Namespace()
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    args.uncertainty = True

    model = RAFT(args)
    summary(model, input_size=((1, 3, 288, 960), (1, 3, 288, 960)))