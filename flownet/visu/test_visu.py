from torch.utils.data import DataLoader

from flownet.dataset.flow_datasets import Sintel, SINTEL_IMG_SIZE, SINTEL_IMG_SIZE_CROPPED
from flownet.dataset.transforms import RandomCropFlowSample
from flownet.visu.visu import inputs_visu


def test_inputs_visu():
    # root_dir = r"C:\Users\IOA1CLJ\Documents\datasets\kitti2015_flow"
    # output_path = r"C:\Temp\kitti_flow"
    # train_set = KittiFlow(root_dir, split="train")

    root_dir = r"C:\Users\IOA1CLJ\Documents\datasets"
    output_path = r"C:\Temp\sintel_flow_cropped"
    train_set = Sintel(root_dir, split="train", pass_name="clean", transforms=RandomCropFlowSample(SINTEL_IMG_SIZE, SINTEL_IMG_SIZE_CROPPED))

    # output_path = r"C:\Temp\flying_chairs_flow"
    # train_set = FlyingChairs(root_dir, split="train")

    dataloader = DataLoader(train_set, batch_size=1, shuffle=False)

    for batch_id, batch_data in enumerate(dataloader):
        img1 = batch_data["img1"][0].numpy()
        img2 = batch_data["img2"][0].numpy()
        flow = batch_data["flow"][0].numpy()
        valid_flow_mask = None
        if "valid_flow_mask" in batch_data:
            valid_flow_mask = batch_data["valid_flow_mask"][0].numpy()

        inputs_visu(img1, img2, flow, batch_id, valid_flow_mask=valid_flow_mask, output_path=output_path)


if __name__ == "__main__":
    test_inputs_visu()
