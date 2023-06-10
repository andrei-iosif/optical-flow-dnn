import os
import numpy as np
import PIL.Image
# import skimage.transform


class Slice2D:
    """
    Represents the horizontal and vertical ranges of a 2D image slice.
    """

    def __init__(self, rows_slice=slice(0, None), cols_slice=slice(0, None)):
        """
        Constructor
        :param rows_slice: vertical range [min_row, max_row]
        :param cols_slice: horizontal range [min_col, max_col]
        """
        self.rows_slice = rows_slice
        self.cols_slice = cols_slice


class ImageTiler:
    """
    Class used for image tiling. Given a matrix of images, it concatenates them into a large image.
    """

    @staticmethod
    def run(mat_of_images):
        """
        Create tiled image from matrix of images.
        :param mat_of_images: matrix of images to be concatenated together
        :return: tiled image
        """
        # Compute the size of each sub-tile in the final image
        # At the beginning we use the tile of the original images
        tile_sizes = []
        dtype_ = None
        max_color_channels = 1
        for r in range(len(mat_of_images)):
            tile_sizes_c = []
            for c in range(len(mat_of_images[r])):
                img = mat_of_images[r][c]
                if img is None:
                    tile_sizes_c += [[0, 0]]
                else:
                    tile_sizes_c += [[img.shape[0], img.shape[1]]]
                    dtype_ = img.dtype
                    if len(img.shape) == 3:
                        max_color_channels = max(max_color_channels, img.shape[2])
            tile_sizes += [tile_sizes_c]

        # Now we have to align the sizes row-wise and column-wise
        for r in range(len(mat_of_images)):
            max_rows_r = 0
            for c in range(len(mat_of_images[r])):
                max_rows_r = max(max_rows_r, tile_sizes[r][c][0])
            for c in range(len(mat_of_images[r])):
                tile_sizes[r][c][0] = max_rows_r

        for c in range(len(mat_of_images[0])):
            max_cols_c = 0
            for r in range(len(mat_of_images)):
                max_cols_c = max(max_cols_c, tile_sizes[r][c][1])

            for r in range(len(mat_of_images)):
                tile_sizes[r][c][1] = max_cols_c

        # Compute shape of result image
        out_shape = [0, 0]
        for r in range(len(mat_of_images)):
            out_shape[0] += tile_sizes[r][0][0]
        for c in range(len(mat_of_images[0])):
            out_shape[1] += tile_sizes[0][c][1]

        # Initialize result image
        if max_color_channels > 1:
            img_out = np.zeros((out_shape[0], out_shape[1], max_color_channels), dtype=dtype_)
        else:
            img_out = np.zeros((out_shape[0], out_shape[1]), dtype=dtype_)

        # Copy each tile into the result image
        offset_r = 0
        for r in range(len(mat_of_images)):
            offset_c = 0
            for c in range(len(mat_of_images[r])):
                img = mat_of_images[r][c]
                if img is not None:
                    if max_color_channels == 1:  # gray out and all tile images are gray as well
                        img_out[offset_r:offset_r + img.shape[0], offset_c:offset_c + img.shape[1]] = img
                    else:
                        if len(img.shape) == 3:  # rgb out or rgba out and current tile image is rgb or rgba
                            img_out[offset_r:offset_r + img.shape[0], offset_c:offset_c + img.shape[1], :img.shape[2]] = img
                        else:  # rgb out, but current tile image is gray
                            for ch in max_color_channels:
                                img_out[offset_r:offset_r + img.shape[0], offset_c:offset_c + img.shape[1], ch] = img

                offset_c += tile_sizes[r][c][1]
            offset_r += tile_sizes[r][0][0]

        return img_out
    

class ImageTilerDumper:
    """
    Class used for creating tiled images using as input images from multiple folders.
    """

    def __init__(self, dirs, out_dir, crop_slices=Slice2D(), resize_factor=None):
        """
        Constructor with parameters.
        :param dirs: matrix of input image folders; position in matrix corresponds to desired position in result image
        :param out_dir: output folder for tiled images
        :param crop_slices: area of input image that will be cropped and copied to result image
        :param resize_factor: scale factor for input image
        """
        self.dirs = np.array(dirs)
        if len(self.dirs.shape) == 1:
            self.dirs.shape = (1, self.dirs.shape[0])
        assert len(self.dirs.shape) == 2

        self.out_dir = out_dir
        self.crop_slices = crop_slices
        self.extensions = ['.bmp', '.png', '.jpg', '.jpeg', '.PNG']

        # if resize_factor is None:
        #     self.resize_factor = [1, 1]
        # elif not isinstance(resize_factor, list):
        #     self.resize_factor = [resize_factor, resize_factor]
        # else:
        #     self.resize_factor = resize_factor
        self.resize_factor = [1, 1]
        assert isinstance(self.resize_factor, list)
        assert len(self.resize_factor) == 2
        self.resize_factor = np.array(self.resize_factor)

    def run(self):
        dirs = self.dirs
        out_dir = self.out_dir
        crop_slices = self.crop_slices
        resize_factor = self.resize_factor

        os.makedirs(out_dir, exist_ok=True)

        # Make sure the directories have the same number of files
        files_in_dirs = []
        cnts = set()
        for dir_ in dirs.flatten():
            files = None
            if dir_ is not None:
                files = sorted([os.path.join(str(dir_), f.decode('utf-8')) for f in os.listdir(dir_) if os.path.splitext(f)[1].decode('utf-8') in self.extensions])
                cnts.add(len(files))
            files_in_dirs += [files]

        if len(cnts) > 1:
            raise Exception('the input directories must have the same number of files')

        cnts = cnts.pop()
        if cnts == 0:
            raise Exception('the input directories must contain image files')

        for i in range(len(files_in_dirs)):
            if files_in_dirs[i] is None:
                files_in_dirs[i] = cnts * [None]

        for _, pairs in enumerate(zip(*files_in_dirs)):
            out_file_name = []

            mat_of_images = []

            # Collect the image tiles for the current index
            for r in range(dirs.shape[0]):
                list_of_images_r = []
                for c in range(dirs.shape[1]):
                    idx = r * dirs.shape[1] + c
                    file = pairs[idx]

                    if file is None:
                        list_of_images_r += [None]
                        continue

                    out_file_name += [os.path.splitext(os.path.basename(file))[0]]

                    # Read image
                    img = np.array(PIL.Image.open(file))

                    # Crop
                    img = img[crop_slices.rows_slice, crop_slices.cols_slice, :]

                    # Resize
                    # if resize_factor[0] != 1 or resize_factor[1] != 1:
                    #     out_shape = np.int32(resize_factor * img.shape[:2])
                    #     img = np.uint8(skimage.transform.resize(img, out_shape, preserve_range=True))

                    list_of_images_r += [img]
                mat_of_images += [list_of_images_r]

            # Build the output image
            img_tiles = ImageTiler.run(mat_of_images)

            # Save the file
            out_file_name = '_'.join(out_file_name) + '.png'
            print(out_file_name)
            out_file = os.path.join(out_dir, out_file_name)
            PIL.Image.fromarray(img_tiles).save(out_file)


def run_tiling_kitti():
    image_dir = r'/home/mnegru/repos/optical-flow-dnn/dump/visu/kitti/raft_chairs_seed_0/visu/img_1'
    gt_flow_dir = r'/home/mnegru/repos/optical-flow-dnn/dump/visu/kitti/raft_chairs_seed_0/visu/gt_flow'
    pred_flow_dir_1 = r'/home/mnegru/repos/optical-flow-dnn/dump/visu/kitti/raft_chairs_seed_0/visu/pred_flow'
    pred_flow_dir_2 = r'/home/mnegru/repos/optical-flow-dnn/dump/visu/kitti/raft_things_seed_0/visu/pred_flow'
    pred_flow_dir_3 = r'/home/mnegru/repos/optical-flow-dnn/dump/visu/kitti/raft_viper_seed_0_mixed/visu/pred_flow'
    pred_flow_dir_4 = r'/home/mnegru/repos/optical-flow-dnn/dump/visu/kitti/raft_viper_seed_0_mixed_semantic/visu/pred_flow'
    out_dir = r'/home/mnegru/repos/optical-flow-dnn/dump/visu/tiles/raft_baseline_and_semantic'

    ImageTilerDumper(
        [
            [image_dir, gt_flow_dir], 
            [pred_flow_dir_1, pred_flow_dir_2], 
            [pred_flow_dir_3, pred_flow_dir_4]
        ], out_dir).run()


if __name__ == '__main__':
    run_tiling_kitti()
