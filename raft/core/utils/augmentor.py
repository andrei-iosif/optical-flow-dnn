import numpy as np
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torchvision.transforms import ColorJitter


class FlowAugmentor:
    """ Applies image augmentations for datasets that contain dense flow GT. """

    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # Spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # Flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img_1, img_2):
        """ Photometric augmentation. 
        Can be applied asymmetrically (separate for each input image), or symmetrically (at the same time for both images).
        
        Args:
            img_1 (np.ndarray): First input image, shape [H, W, 3].
            img_2 (np.ndarray): Second input image, shape [H, W, 3].

        Returns:
            pair of augmented images, same dimensions as input images
        """

        # Asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img_1 = np.array(self.photo_aug(Image.fromarray(img_1)), dtype=np.uint8)
            img_2 = np.array(self.photo_aug(Image.fromarray(img_2)), dtype=np.uint8)

        # Symmetric
        else:
            image_stack = np.concatenate([img_1, img_2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img_1, img_2 = np.split(image_stack, 2, axis=0)

        return img_1, img_2

    def eraser_transform(self, img_1, img_2, bounds=[50, 100]):
        """ Occlusion augmentation.
        Replaces a rectangular region of the image with the mean color of that region.

        Args:
            img_1 (np.ndarray): First input image, shape [H, W, 3].
            img_2 (np.ndarray): Second input image, shape [H, W, 3].
            bounds (list, optional): Range from which the dimensions of rectangular region are selected. Defaults to [50, 100].

        Returns:
            pair of augmented images, same dimensions as input images
        """

        ht, wd = img_1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img_1, img_2

    def spatial_transform(self, img_1, img_2, flow, valid_mask=None, semseg_1=None, semseg_2=None):
        """ Spatial image augmentations. 

        Flow valid mask and semantic GT are given as optional inputs, because they are not available for all datasets.

        Args:
            img_1 (np.ndarray): First input image, shape [H, W, 3].
            img_2 (np.ndarray): Second input image, shape [H, W, 3].
            flow (np.ndarray): Optical flow GT, shape [H, W, 2].
            valid_mask (np.ndarray, optional): Optical flow valid mask, shape [H, W, 1]. Defaults to None.
            semseg_1 (np.ndarray, optional): Semantic segmentation GT for first image, shape [H, W, 3]. Defaults to None.
            semseg_2 (np.ndarray, optional): Semantic segmentation GT for first image, shape [H, W, 3]. Defaults to None.

        Returns:
            tuple of augmented images, flow GT, semantic GT, at dimensions specified by crop size
        """
        # Randomly sample scale
        ht, wd = img_1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        # Rescaling
        if np.random.rand() < self.spatial_aug_prob:
            img_1 = cv2.resize(img_1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_2 = cv2.resize(img_2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

            if valid_mask is not None:
                valid_mask = valid_mask.astype(np.float32)
                valid_mask = cv2.resize(valid_mask, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            
            if semseg_1 is not None and semseg_2 is not None:
                semseg_1, semseg_2 = semseg_1.astype(np.float32), semseg_2.astype(np.float32)
                semseg_1 = cv2.resize(semseg_1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
                semseg_2 = cv2.resize(semseg_2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

        # Flipping
        if self.do_flip:
            # Horizontal flip
            if np.random.rand() < self.h_flip_prob:
                img_1 = img_1[:, ::-1]
                img_2 = img_2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

                if valid_mask is not None:
                    valid_mask = valid_mask[:, ::-1]

                if semseg_1 is not None and semseg_2 is not None:
                    semseg_1 = semseg_1[:, ::-1]
                    semseg_2 = semseg_2[:, ::-1]

            # Vertical flip
            if np.random.rand() < self.v_flip_prob:
                img_1 = img_1[::-1, :]
                img_2 = img_2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

                if valid_mask is not None:
                    valid_mask = valid_mask[::-1, :]

                if semseg_1 is not None and semseg_2 is not None:
                    semseg_1 = semseg_1[::-1, :]
                    semseg_2 = semseg_2[::-1, :]

        # Cropping
        y0 = np.random.randint(0, img_1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img_1.shape[1] - self.crop_size[1])
        
        img_1 = img_1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_2 = img_2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if valid_mask is not None:
            valid_mask = valid_mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if semseg_1 is not None and semseg_2 is not None:
            semseg_1 = semseg_1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            semseg_2 = semseg_2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img_1, img_2, flow, valid_mask, semseg_1, semseg_2

    def __call__(self, img_1, img_2, flow, valid_mask=None, semseg_1=None, semseg_2=None):
        img_1, img_2 = self.color_transform(img_1, img_2)
        img_1, img_2 = self.eraser_transform(img_1, img_2)
        img_1, img_2, flow, valid_mask, semseg_1, semseg_2 = self.spatial_transform(
            img_1, img_2, flow, valid_mask=valid_mask, semseg_1=semseg_1, semseg_2=semseg_2)

        img_1 = np.ascontiguousarray(img_1)
        img_2 = np.ascontiguousarray(img_2)
        flow = np.ascontiguousarray(flow)

        if valid_mask is not None:
            valid_mask = np.ascontiguousarray(valid_mask).astype(bool)

        if semseg_1 is not None and semseg_2 is not None:
            semseg_1 = np.ascontiguousarray(semseg_1).astype(np.int32)
            semseg_2 = np.ascontiguousarray(semseg_2).astype(np.int32)

        return img_1, img_2, flow, valid_mask, semseg_1, semseg_2


class SparseFlowAugmentor:
    """ Applies image augmentations for datasets that contain sparse flow GT. """

    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # Spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # Flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img_1, img_2):
        """ Photometric augmentation. 
        Can be applied asymmetrically (separate for each input image), or symmetrically (at the same time for both images).
        
        Args:
            img_1 (np.ndarray): First input image, shape [H, W, 3].
            img_2 (np.ndarray): Second input image, shape [H, W, 3].

        Returns:
            pair of augmented images, same dimensions as input images
        """
        image_stack = np.concatenate([img_1, img_2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img_1, img_2 = np.split(image_stack, 2, axis=0)
        return img_1, img_2

    def eraser_transform(self, img_1, img_2):
        """ Occlusion augmentation.
        Replaces a rectangular region of the image with the mean color of that region.

        Args:
            img_1 (np.ndarray): First input image, shape [H, W, 3].
            img_2 (np.ndarray): Second input image, shape [H, W, 3].
            bounds (list, optional): Range from which the dimensions of rectangular region are selected. Defaults to [50, 100].

        Returns:
            pair of augmented images, same dimensions as input images
        """
        ht, wd = img_1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img_2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img_1, img_2

    def resize_sparse_flow_map(self, flow, valid_mask, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht), indexing="ij")
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid_mask = valid_mask.reshape(-1).astype(np.float32)

        coords0 = coords[valid_mask>=1]
        flow0 = flow[valid_mask>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img_1, img_2, flow, valid_mask):
        """ Spatial image augmentations. 

        Args:
            img_1 (np.ndarray): First input image, shape [H, W, 3].
            img_2 (np.ndarray): Second input image, shape [H, W, 3].
            flow (np.ndarray): Optical flow GT, shape [H, W, 2].
            valid_mask (np.ndarray, optional): Optical flow valid mask, shape [H, W, 1]. Defaults to None.
            
        Returns:
            tuple of augmented images, flow GT at dimensions specified by crop size
        """
        # Randomly sample scale
        ht, wd = img_1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        # Rescaling
        if np.random.rand() < self.spatial_aug_prob:
            img_1 = cv2.resize(img_1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_2 = cv2.resize(img_2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid_mask = self.resize_sparse_flow_map(flow, valid_mask, fx=scale_x, fy=scale_y)

        # Flipping
        if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img_1 = img_1[:, ::-1]
                img_2 = img_2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid_mask = valid_mask[:, ::-1]

        # Cropping
        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img_1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img_1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img_1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img_1.shape[1] - self.crop_size[1])

        img_1 = img_1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_2 = img_2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_mask = valid_mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img_1, img_2, flow, valid_mask


    def __call__(self, img_1, img_2, flow, valid_mask):
        img_1, img_2 = self.color_transform(img_1, img_2)
        img_1, img_2 = self.eraser_transform(img_1, img_2)
        img_1, img_2, flow, valid_mask = self.spatial_transform(img_1, img_2, flow, valid_mask)

        img_1 = np.ascontiguousarray(img_1)
        img_2 = np.ascontiguousarray(img_2)
        flow = np.ascontiguousarray(flow)
        valid_mask = np.ascontiguousarray(valid_mask)

        return img_1, img_2, flow, valid_mask
