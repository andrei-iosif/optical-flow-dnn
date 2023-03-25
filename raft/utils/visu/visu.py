import os

import matplotlib.pyplot as plt
import numpy as np

from visu.flow_visu import flow_to_color


def inputs_visu(img_1, gt_flow, valid_flow_mask=None, semseg_gt=None, sample_idx=-1, output_path=None):
    """
    Create visualization of input images, ground truth optical flow, and optionally, valid mask and/or
    semantic segmentation ground truth.

    Args:
        img1 (np.ndarray): First input image, with shape [3, H, W]
        gt_flow (np.ndarray): Ground truth optical flow, with shape [2, H, W]
        valid_flow_mask (np.ndarray): Optical flow validity mask, with shape [1, H, W]
        semseg_gt (np.ndarray): Semantic segmentation ground truth, RGB format, shape [3, H, W]
        sample_idx (int): Frame number
        output_path (str): Path where the visu is saved
    """
    if valid_flow_mask is not None:
        gt_flow = gt_flow * valid_flow_mask

    flow_img = flow_to_color(gt_flow, channels_last=False)

    img_1 = img_1.transpose(1, 2, 0).astype(np.uint8)
    height, width = img_1.shape[0], img_1.shape[1]
    aspect_ratio = width / height

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].imshow(img_1)
    axes[0, 0].set_title(f'Image 1', fontsize=25)

    axes[0, 1].imshow(flow_img)
    axes[0, 1].set_title(f'Flow GT', fontsize=25)

    if valid_flow_mask is not None:
        axes[1, 0].imshow(valid_flow_mask, cmap='gray')
        axes[1, 0].set_title('Valid flow mask', fontsize=25)
    else:
        axes[1, 0].axis('off')

    if semseg_gt is not None:
        semseg_gt = semseg_gt.transpose(1, 2, 0).astype(np.uint8)
        axes[1, 1].imshow(semseg_gt)
        axes[1, 1].set_title('Semseg GT', fontsize=25)
    else:
        axes[1, 1].axis('off')

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.set_size_inches(aspect_ratio * 25, 25)
        fig.savefig(os.path.join(output_path, str(sample_idx) + '.jpeg'), bbox_inches="tight", pad_inches=0.3, dpi=100)
        plt.close()
        print(f"Saved frame {sample_idx}")
    else:
        plt.show()


def predictions_visu(img_1, gt_flow, pred_flow, sample_idx, output_path, additional_info=""):
    """
    Create visualization of input images, predicted optical flow and ground truth optical flow.

    Args:
        img1 (np.ndarray): First input image, with shape [3, H, W]
        gt_flow (np.ndarray): Ground truth optical flow, with shape [2, H, W]
        pred_flow (np.ndarray): Predicted optical flow, with shape [2, H, W]
        sample_idx (int): Frame number
        output_path (str): Path where the visu is saved
        additional_info (str, optional): Additional label to be added to output file names. Defaults to "".
    """
    gt_flow_img = flow_to_color(gt_flow, channels_last=False)
    pred_flow_img = flow_to_color(pred_flow, channels_last=False)

    img_1 = img_1.transpose(1, 2, 0).astype(np.uint8)

    height, width = img_1.shape[0], img_1.shape[1]
    aspect_ratio = width / height

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].imshow(img_1)
    axes[0, 0].set_title(f'Image 1', fontsize=25)

    axes[0, 1].imshow(gt_flow_img)
    axes[0, 1].set_title(f'Flow GT', fontsize=25)

    axes[1, 0].imshow(pred_flow_img)
    axes[1, 0].set_title(f'Predicted Flow', fontsize=25)

    axes[1, 1].axis('off')

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.set_size_inches(aspect_ratio * 25, 25)
        fig.savefig(os.path.join(output_path, str(sample_idx) + "_" + additional_info + '.jpeg'), bbox_inches="tight", pad_inches=0.3, dpi=100)
        plt.close()
        print(f"Saved frame {sample_idx}")
    else:
        plt.show()
