import os
import re
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

__all__ = (
    "KittiFlow",
    "Sintel",
    "FlyingChairs"
)

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg

SINTEL_IMG_SIZE = 436, 1024
SINTEL_IMG_SIZE_CROPPED = 384, 768


class FlowDataset(ABC, VisionDataset):
    # Some datasets like Kitti have a built-in valid_flow_mask, indicating which flow values are valid
    # For those we return (img1, img2, flow, valid_flow_mask), and for the rest we return (img1, img2, flow),
    # and it's up to whatever consumes the dataset to decide what valid_flow_mask should be.
    _has_builtin_flow_mask = False

    def __init__(self, root, transforms=None):
        super().__init__(root=root)
        self.transforms = transforms

        self._flow_list = []
        self._image_list = []

    def _read_img(self, file_name):
        """
        Read RGB image from file, shape HxWx3.
        Needs to be reshaped to 3xHxW before being input to the net.
        :param file_name: image file name
        :return: np.array with shape HxWx3
        """

        img = Image.open(file_name)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    @abstractmethod
    def _read_flow(self, file_name):
        # Return the flow or a tuple with the flow and the valid_flow_mask if _has_builtin_flow_mask is True
        pass

    def __getitem__(self, index):
        img1 = self._read_img(self._image_list[index][0])
        img2 = self._read_img(self._image_list[index][1])

        if self._flow_list:  # it will be empty for some dataset when split="test"
            flow = self._read_flow(self._flow_list[index])
            if self._has_builtin_flow_mask:
                flow, valid_flow_mask = flow
            else:
                valid_flow_mask = None
        else:
            flow = valid_flow_mask = None

        if self.transforms is not None:
            img1, img2, flow, valid_flow_mask = self.transforms(img1, img2, flow, valid_flow_mask)

        if self._has_builtin_flow_mask or valid_flow_mask is not None:
            return {"img1": img1, "img2": img2, "flow": flow, "valid_flow_mask": valid_flow_mask}
        else:
            return {"img1": img1, "img2": img2, "flow": flow}

    def __len__(self):
        return len(self._image_list)


class Sintel(FlowDataset):
    """`Sintel <http://sintel.is.tue.mpg.de/>`_ Dataset for optical flow.
    Original image resolution: 436x1024

    The dataset is expected to have the following structure: ::

        root
            Sintel
                testing
                    clean
                        scene_1
                        scene_2
                        ...
                    final
                        scene_1
                        scene_2
                        ...
                training
                    clean
                        scene_1
                        scene_2
                        ...
                    final
                        scene_1
                        scene_2
                        ...
                    flow
                        scene_1
                        scene_2
                        ...

    Args:
        root (string): Root directory of the Sintel Dataset.
        split (string, optional): The dataset split, either "train" (default) or "test"
        pass_name (string, optional): The pass to use, either "clean" (default), "final", or "both". See link above for
            details on the different passes.
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
            ``valid_flow_mask`` is expected for consistency with other datasets which
            return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
    """

    def __init__(self, root, split="train", pass_name="clean", transforms=None):
        super().__init__(root=root, transforms=transforms)

        verify_str_arg(split, "split", valid_values=("train", "test"))
        verify_str_arg(pass_name, "pass_name", valid_values=("clean", "final", "both"))
        passes = ["clean", "final"] if pass_name == "both" else [pass_name]

        root = Path(root) / "Sintel"
        flow_root = root / "training" / "flow"

        for pass_name in passes:
            split_dir = "training" if split == "train" else split
            image_root = root / split_dir / pass_name
            for scene in os.listdir(image_root):
                image_list = sorted(glob(str(image_root / scene / "*.png")))
                for i in range(len(image_list) - 1):
                    self._image_list += [[image_list[i], image_list[i + 1]]]

                if split == "train":
                    self._flow_list += sorted(glob(str(flow_root / scene / "*.flo")))

    def __getitem__(self, index):
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3-tuple with ``(img1, img2, flow)``.
            The flow is a numpy array of shape (2, H, W) and the images are PIL images.
            ``flow`` is None if ``split="test"``.
            If a valid flow mask is generated within the ``transforms`` parameter,
            a 4-tuple with ``(img1, img2, flow, valid_flow_mask)`` is returned.
        """
        return super().__getitem__(index)

    def _read_flow(self, file_name):
        return _read_flo(file_name)


class KittiFlow(FlowDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow>`__ dataset for optical flow (2015).
    Original image resolution: 375x1242

    The dataset is expected to have the following structure: ::

        root
            KittiFlow
                testing
                    image_2
                training
                    image_2
                    flow_occ

    Args:
        root (string): Root directory of the KittiFlow Dataset.
        split (string, optional): The dataset split, either "train" (default) or "test"
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
    """

    _has_builtin_flow_mask = True

    def __init__(self, root, split="train", transforms=None):
        super().__init__(root=root, transforms=transforms)

        verify_str_arg(split, "split", valid_values=("train", "test"))

        root = Path(root) / "KittiFlow" / (split + "ing")
        images1 = sorted(glob(str(root / "image_2" / "*_10.png")))
        images2 = sorted(glob(str(root / "image_2" / "*_11.png")))

        if not images1 or not images2:
            raise FileNotFoundError(
                "Could not find the Kitti flow images. Please make sure the directory structure is correct."
            )

        for img1, img2 in zip(images1, images2):
            self._image_list += [[img1, img2]]

        if split == "train":
            self._flow_list = sorted(glob(str(root / "flow_occ" / "*_10.png")))

    def __getitem__(self, index):
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 4-tuple with ``(img1, img2, flow, valid_flow_mask)``
            where ``valid_flow_mask`` is a numpy boolean mask of shape (H, W)
            indicating which flow values are valid. The flow is a numpy array of
            shape (2, H, W) and the images are PIL images. ``flow`` and ``valid_flow_mask`` are None if
            ``split="test"``.
        """
        return super().__getitem__(index)

    def _read_flow(self, file_name):
        return _read_kitti_flow_and_valid_mask(file_name)


class FlyingChairs(FlowDataset):
    """`FlyingChairs <https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs>`_ Dataset for optical flow.
    Original image resolution: 384x512

    The dataset is expected to have the following structure: ::

        root
            FlyingChairs
                data
                    00001_flow.flo
                    00001_img1.ppm
                    00001_img2.ppm
                    ...
                FlyingChairs_train_val.txt


    Args:
        root (string): Root directory of the FlyingChairs Dataset.
        split (string, optional): The dataset split, either "train" (default) or "val"
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
            ``valid_flow_mask`` is expected for consistency with other datasets which
            return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
    """

    def __init__(self, root, split="train", transforms=None):
        super().__init__(root=root, transforms=transforms)

        verify_str_arg(split, "split", valid_values=("train", "val"))

        root = Path(root) / "FlyingChairs"
        images = sorted(glob(str(root / "data" / "*.ppm")))
        flows = sorted(glob(str(root / "data" / "*.flo")))

        split_file_name = "FlyingChairs_train_val.txt"

        if not os.path.exists(root / split_file_name):
            raise FileNotFoundError(
                "The FlyingChairs_train_val.txt file was not found - please download it from the dataset page (see docstring)."
            )

        split_list = np.loadtxt(str(root / split_file_name), dtype=np.int32)
        for i in range(len(flows)):
            split_id = split_list[i]
            if (split == "train" and split_id == 1) or (split == "val" and split_id == 2):
                self._flow_list += [flows[i]]
                self._image_list += [[images[2 * i], images[2 * i + 1]]]

    def __getitem__(self, index):
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3-tuple with ``(img1, img2, flow)``.
            The flow is a numpy array of shape (2, H, W) and the images are PIL images.
            ``flow`` is None if ``split="val"``.
            If a valid flow mask is generated within the ``transforms`` parameter,
            a 4-tuple with ``(img1, img2, flow, valid_flow_mask)`` is returned.
        """
        return super().__getitem__(index)

    def _read_flow(self, file_name):
        return _read_flo(file_name)


def _read_flo(file_name):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    # Everything needs to be in little Endian according to
    # https://vision.middlebury.edu/flow/code/flow-code/README.txt
    with open(file_name, "rb") as f:
        magic = np.fromfile(f, "c", count=4).tobytes()
        if magic != b"PIEH":
            raise ValueError("Magic number incorrect. Invalid .flo file")

        w = int(np.fromfile(f, "<i4", count=1))
        h = int(np.fromfile(f, "<i4", count=1))
        data = np.fromfile(f, "<f4", count=2 * w * h)
        return data.reshape(h, w, 2)

def _read_kitti_flow_and_valid_mask(file_name):
    """
    Read flow (HxWx2) and valid mask (HxWx1) from image with 3 channels (16 bit).
    Function modified from: https://github.com/open-mmlab/mmflow/blob/master/mmflow/datasets/utils/flow_io.py
    :param file_name: flow input file
    :return: flow and valid map
    """
    flow = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    # Flow shape = (H, W, 2); valid shape = (H, W)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    return flow, valid


def _read_pfm(file_name):
    """Read flow in .pfm format"""

    with open(file_name, "rb") as f:
        header = f.readline().rstrip()
        if header != b"PF":
            raise ValueError("Invalid PFM file")

        dim_match = re.match(rb"^(\d+)\s(\d+)\s$", f.readline())
        if not dim_match:
            raise Exception("Malformed PFM header.")
        w, h = (int(dim) for dim in dim_match.groups())

        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.fromfile(f, dtype=endian + "f")

    data = data.reshape(h, w, 3).transpose(2, 0, 1)
    data = np.flip(data, axis=1)  # flip on h dimension
    data = data[:2, :, :]
    return data.astype(np.float32)
