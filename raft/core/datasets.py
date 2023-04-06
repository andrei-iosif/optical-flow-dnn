# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import random
import re
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, read_semseg=False, from_npz=False, from_png=False, crop_bottom=False):
        """ Constructor for FlowDataset class.

        Args:
            aug_params (_type_, optional): _description_. Defaults to None.
            sparse (bool, optional): If true, read sparse flow in KITTI format. Defaults to False.
            read_semseg (bool, optional): If true, read also semantic segmentation GT. Defaults to False.
            from_npz (bool, optional): If true, read flow from npz files (VIPER format). Defaults to False.
            from_png (bool, optional): If true, read flow from png files (Virtual KITTI format). Defaults to False.
            crop_bottom (bool, optional): If true, crop bottom part of input images. Defaults to False.
        """
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.read_semseg = read_semseg
        self.semseg_list = []

        self.from_npz = from_npz
        self.from_png = from_png
        self.crop_bottom = crop_bottom

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        # Read flow GT
        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        elif self.from_npz:
            flow, valid = frame_utils.read_flow_from_npz(self.flow_list[index])
        elif self.from_png:
            flow, valid = frame_utils.read_vkitti_png_flow(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        # Read semseg GT
        semseg_1, semseg_2 = None, None
        if self.read_semseg:
            semseg_1 = np.array(frame_utils.read_gen(self.semseg_list[index][0]).convert('RGB')).astype(np.uint8)
            semseg_2 = np.array(frame_utils.read_gen(self.semseg_list[index][1]).convert('RGB')).astype(np.uint8)
  
        # Read input images
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # Crop from the bottom of images and flow GT (used for VIPER dataset)
        if self.crop_bottom:
            img1 = img1[:-150, :, :]
            img2 = img2[:-150, :, :]
            flow = flow[:-150, :, :]
            if valid is not None:
                valid = valid[:-150, :]
            if semseg_1 is not None:
                semseg_1 = semseg_1[:-150, :]
                semseg_2 = semseg_2[:-150, :]

        if self.augmentor is not None:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        # Create tensors from numpy arrays and permute dimensions => channel-first
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        if self.read_semseg:
            semseg_1 = torch.from_numpy(semseg_1).permute(2, 0, 1).float()
            semseg_2 = torch.from_numpy(semseg_2).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.read_semseg:
            return img1, img2, flow, valid.float(), semseg_1, semseg_2
        else:
            return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while True:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


class FlyingThingsSubset(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D_subset'):
        super(FlyingThingsSubset, self).__init__(aug_params)

        for view in ['left', 'right']:
            image_root = osp.join(root, 'train/image_clean', view)
            flow_root_forward = osp.join(root, 'train/flow', view, 'into_future')
            flow_root_backward = osp.join(root, 'train/flow/', view, 'into_past')

            image_list = sorted(os.listdir(image_root))
            flow_forward = set(os.listdir(flow_root_forward))
            flow_backward = set(os.listdir(flow_root_backward))

            for i in range(len(image_list)-1):
                img1 = image_list[i]
                img2 = image_list[i+1]

                image_path1 = osp.join(image_root, img1)
                image_path2 = osp.join(image_root, img2)

                if img1.replace('.png', '.flo') in flow_forward:
                    self.image_list += [ [image_path1, image_path2] ]
                    self.flow_list += [ osp.join(flow_root_forward, img1.replace('.png', '.flo')) ]

                if img2.replace('.png', '.flo') in flow_backward:
                    self.image_list += [ [image_path2, image_path1] ]
                    self.flow_list += [ osp.join(flow_root_backward, img2.replace('.png', '.flo')) ]


class VIPER(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/VIPER', split='train'):
        super(VIPER, self).__init__(aug_params, sparse=False, from_npz=True, crop_bottom=False, read_semseg=False)

        seq_idx = 1

        while True:
            flow_paths = sorted(glob(os.path.join(root, split, "flow", "%03d" % seq_idx, "%03d_*.npz" % seq_idx)))
            image_paths = sorted(glob(os.path.join(root, split, "img", "%03d" % seq_idx, "%03d_*.jpg" % seq_idx)))
            if self.read_semseg:
                semseg_paths = sorted(glob(os.path.join(root, split, "cls", "%03d" % seq_idx, "%03d_*.png" % seq_idx)))
                assert len(semseg_paths) == len(image_paths), f"Mismatch between number of semseg maps ({len(semseg_paths)}) and number of images ({len(image_paths)})"

            if len(flow_paths) == 0:
                break

            # assert len(flow_paths) * 2 == len(image_paths), f"Mismatch between number of flow maps ({len(flow_paths)}) and number of images ({len(image_paths)})"

            for i in range(len(flow_paths) - 1):
                frame_id = os.path.basename(flow_paths[i]).split('.')[0]
                next_frame_id = f"{frame_id[:-1]}1"

                for j in range(len(image_paths) - 1):
                    if re.search(frame_id, image_paths[j]) is not None and re.search(next_frame_id, image_paths[j + 1]) is not None:
                        self.flow_list += [flow_paths[i]]
                        self.image_list += [[image_paths[j], image_paths[j + 1]]]
                        if self.read_semseg:
                            self.semseg_list += [[semseg_paths[j], semseg_paths[j + 1]]]
                        break

            seq_idx += 1


class VirtualKITTI(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/VirtualKITTI'):
        super(VirtualKITTI, self).__init__(aug_params, from_png=True, read_semseg=False)

        self.scene_ids = [1, 2, 6, 18, 20]
        self.variants = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone',
            'fog', 'morning', 'overcast', 'rain', 'sunset']

        for scene_id in self.scene_ids:
            for variant in self.variants:
                image_paths = sorted(glob(os.path.join(root, f"Scene{scene_id:02d}", variant, "frames", "rgb", "Camera_0", "rgb_*.jpg")))
                flow_paths = sorted(glob(os.path.join(root, f"Scene{scene_id:02d}", variant, "frames", "forwardFlow", "Camera_0", "flow_*.png")))
                if self.read_semseg:
                    semseg_paths = sorted(glob(os.path.join(root, f"Scene{scene_id:02d}", variant, "frames", "classSegmentation", "Camera_0", "classgt_*.png")))

                # TODO: improve this; do not read RGB and semseg twice
                for i in range(len(flow_paths) - 1):
                    self.flow_list += [flow_paths[i]]
                    self.image_list += [[image_paths[i], image_paths[i + 1]]]
                    if self.read_semseg:
                        self.semseg_list += [[semseg_paths[i], semseg_paths[i + 1]]]


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding training set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        # clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        # final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        # train_dataset = clean_dataset + final_dataset
        train_dataset = FlyingThingsSubset(aug_params)

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        # things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        things = FlyingThings3D(aug_params)
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        # Train set = combination of Sintel clean/final, KITTI, HD1K and FlyingThings3D
        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'viper':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}
        virtual_kitti = VirtualKITTI(aug_params)
        viper = VIPER(aug_params)
        train_dataset = virtual_kitti + 2 * viper

    else:
        raise AttributeError(f"Invalid training stage: {args.stage}")

    # Reset seed here to ensure same validation subsets (ex. see evaluate.py for VIPER dataset)
    torch.manual_seed(0)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
