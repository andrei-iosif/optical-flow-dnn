import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from flownet.visu.flow_visu import flow_to_color


def inputs_visu(img1, img2, flow, sample_idx, valid_flow_mask=None, output_path=None):
    matplotlib.use('TkAgg')

    flow_img = flow_to_color(flow)
    height, width = img1.shape[0], img1.shape[1]
    aspect_ratio = width / height

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(f'Image 1', fontsize=25)

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title(f'Image 2', fontsize=25)

    axes[1, 0].imshow(flow_img)
    axes[1, 0].set_title(f'GT Flow', fontsize=25)

    if valid_flow_mask is not None:
        axes[1, 1].imshow(valid_flow_mask, cmap='gray')
        axes[1, 1].set_title('Valid flow mask', fontsize=25)
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


def predictions_visu(img1, img2, gt_flow, pred_flow, sample_idx, output_path=None, additional_info=""):
    gt_flow_img = flow_to_color(gt_flow, channels_last=False)
    pred_flow_img = flow_to_color(pred_flow, channels_last=False)

    img1 = img1.transpose(1, 2, 0).astype(np.uint8)
    img2 = img2.transpose(1, 2, 0).astype(np.uint8)

    height, width = img1.shape[0], img1.shape[1]
    aspect_ratio = width / height

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(f'Image 1', fontsize=25)

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title(f'Image 2', fontsize=25)

    axes[1, 0].imshow(gt_flow_img)
    axes[1, 0].set_title(f'GT Flow', fontsize=25)

    axes[1, 1].imshow(pred_flow_img)
    axes[1, 1].set_title(f'Predicted Flow', fontsize=25)

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.set_size_inches(aspect_ratio * 25, 25)
        fig.savefig(os.path.join(output_path, str(sample_idx) + "_" + additional_info + '.jpeg'), bbox_inches="tight", pad_inches=0.3, dpi=100)
        plt.close()
        print(f"Saved frame {sample_idx}")
    else:
        plt.show()
