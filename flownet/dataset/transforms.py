import random


class RandomCrop:
    def __init__(self, img_size, crop_size):
        """
        Random image crop with given crop size.
        :param img_size: size of image to be cropped
        :param crop_size: output shape (Hc, Wc)

        """
        height, width = img_size[0], img_size[1]

        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]

        self.row = random.randint(0, height - self.crop_height)
        self.col = random.randint(0, width - self.crop_width)

    def __call__(self, img):
        """
        :param img: image to be cropped; shape H x W x C
        :return: cropped image; shape Hc x Wc x C
        """
        return img[self.row:(self.row + self.crop_height), self.col:(self.col + self.crop_width), :]


class RandomCropFlowSample:
    def __init__(self, img_size, crop_size):
        self.random_crop = RandomCrop(img_size, crop_size)

    def __call__(self, img1, img2, flow, valid_flow_mask):
        img1_cropped = self.random_crop(img1)
        img2_cropped = self.random_crop(img2)

        if flow is not None:
            flow_cropped = self.random_crop(flow)
        else:
            flow_cropped = flow

        if valid_flow_mask is not None:
            valid_flow_cropped = self.random_crop(valid_flow_mask)
        else:
            valid_flow_cropped = valid_flow_mask

        return img1_cropped, img2_cropped, flow_cropped, valid_flow_cropped
