import clearml
import torch

import core.utils.visu.visu as visu
from core.utils.utils import InputPadder

def upload_debug_visu(model, data_sample, iters, train_step):
    """ Create a debug visu to be uploaded to ClearML dashboard.

    Args:
        model (nn.Model): RAFT model
        data_sample (tuple): debug sample from dataset
        iters (int): number of iterations used for refinement
        train_step (int): epoch/iteration of training
    """
    with torch.no_grad():
        # Unpack data sample 
        image_1, image_2, flow_gt, valid_mask = data_sample
        
        # Convert input from (3, H, W) to (1, 3, H, W)
        image_1 = image_1[None].cuda()
        image_2 = image_2[None].cuda()

        # Apply padding so that input dimensions are divisible by 8
        padder = InputPadder(image_1.shape)
        image_1, image_2 = padder.pad(image_1, image_2)

        # Run inference on image pair (test mode)
        _, flow_pred = model(image_1, image_2, iters=iters, test_mode=True)

        # Remove batch dimension and padding
        flow_pred = padder.unpad(flow_pred[0])
        
        # Ensure all tensors are copied from GPU
        image_1 = image_1[0].cpu().numpy()
        flow_pred = flow_pred.cpu().numpy()
        flow_gt = flow_gt.cpu().numpy()

        # Create visu
        fig = visu.predictions_visu(image_1, flow_gt, flow_pred, sample_id=0, output_path=None)

        # Upload to ClearML
        logger = clearml.Logger.current_logger()
        logger.report_matplotlib_figure(
            title="Flow prediction visu", series="", iteration=train_step, 
            figure=fig, report_image=True, report_interactive=True)