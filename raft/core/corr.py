import torch
import torch.nn.functional as F

from utils.utils import bilinear_sampler

try:
    import alt_cuda_corr
except ModuleNotFoundError:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    """
    Given 2 feature maps from 2 consecutive images, compute the all-pairs correlation volume.
    """

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        :param fmap1: first feature map
        :param fmap2: second feature map
        :param num_levels: number of levels in correlation pyramid
        :param radius: size of neighborhood used in correlation lookup
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # Compute all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        # Create correlation pyramid
        # <=> for each pixel in image I1, we have a pyramid of matching costs with all pixels from image 2
        # The base correlation volume is downsampled on the last 2 dimensions (corresponding to spatial dimensions
        # of image 2)
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """
        Correlation lookup operator
        :param coords: coordinates grid warped by current flow estimation
        :return: correlation feature vector for each pixel
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]

            # Create local coordinates grid
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1)

            # Compute warped coordinates grid at current pyramid level
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i

            # For each warped pixel, compute the indexes of its local neighborhood
            # => will be sampled from the correlation volume
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """
        Compute dot product between all pairs of feature vectors
        """
        # Flatten spatial dimensions of feature maps (B, C, H, W) => (B, C, HxW)
        # 1 batch element = a C-dimensional feature vector for each of the HxW pixels
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 

        # Compute correlation using matrix multiplication (<=> a lot of dot products)
        # [(HxW) x C] x [C x (HxW)] ==> (HxW) x (HxW)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        # Normalize result (divide by sqrt(C))
        return corr.mul_(1.0 / torch.sqrt(torch.tensor(dim).float()))


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


if __name__ == "__main__":
    r = 4
    dx = torch.linspace(-r, r, 2 * r + 1)
    dy = torch.linspace(-r, r, 2 * r + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), dim=-1)
    delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
    print()
