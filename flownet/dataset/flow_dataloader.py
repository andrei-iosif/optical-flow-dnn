import torch
from torch.utils.data import DataLoader, random_split

from flownet.dataset.flow_datasets import KittiFlow, Sintel, SINTEL_IMG_SIZE_CROPPED, SINTEL_IMG_SIZE
from flownet.dataset.transforms import RandomCropFlowSample


def get_kitti_flow_dataloader(args):
    """
    Prepare KITTI Flow dataset. Original "train" split => train split + validation split
    :param args: cmd arguments
    :return: train, val, test dataloaders
    """
    root_dir = args.dataset_path
    kitti_dataset_train = KittiFlow(root_dir, split="train")
    train_set, val_set = random_split(kitti_dataset_train,
                                      [int(len(kitti_dataset_train) * 0.7), int(len(kitti_dataset_train) * 0.3)],
                                      generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False)

    kitti_dataset_test = KittiFlow(root_dir, split="test")
    test_dataloader = DataLoader(kitti_dataset_test, batch_size=int(args.batch_size), shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


def get_sintel_flow_dataloader(args, batch_overfit_mode=False):
    """
    Prepare Sintel dataset. Original "train" split => train split + validation split
    :param args: cmd arguments
    :return: train, val, test dataloaders
    """
    batch_size = args.batch_size
    root_dir = args.dataset_path
    sintel_dataset_train = Sintel(root_dir, split="train", pass_name="clean",
                                  transforms=RandomCropFlowSample(SINTEL_IMG_SIZE, SINTEL_IMG_SIZE_CROPPED))

    if batch_overfit_mode:
        train_set, val_set, test_set = random_split(sintel_dataset_train,
                                                    [batch_size, len(sintel_dataset_train) - 2 * batch_size,
                                                     batch_size],
                                                    generator=torch.Generator().manual_seed(42))
    else:
        # Train-validation split from original FlowNet paper: 908 - train, 133 - val
        # New split (Sintel test does not have flow GT, need to submit results): 625 - train, 208 - val, 208 - test
        train_set, val_set, test_set = random_split(sintel_dataset_train,
                                                    [625, 208, 208],
                                                    generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True)
    val_dataloader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=int(args.batch_size), shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
